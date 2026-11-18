# the graph runs the same mapper, pauses when a field is close to the fill policy, and resumes with code

from columnmap.decider import Answer, Response
from columnmap.harness import retry_policy, run_graph
from columnmap.harness.graph import _repair
from columnmap.knowledge import Knowledge, Output

ROWS = [
    {
        "first": "Budi",
        "last": "Halim",
        "fee": 10,
        "charge": 100,
        "pct": 0.1,
        "agent": "A1",
    },
    {
        "first": "Siti",
        "last": "Wijaya",
        "fee": 20,
        "charge": 200,
        "pct": 0.2,
        "agent": "A2",
    },
]
KNOWLEDGE = Knowledge(
    (
        Output("full_name", "string", "Full name, given name then family name"),
        Output("fee_amount", "number", "Fee charged on the transaction"),
        Output("total_charge", "number", "Transaction amount plus fee"),
        Output("agent", "string", "Selling agent identifier"),
    ),
    (),
)
KINDS = {
    "full_name": "join",
    "fee_amount": "copy",
    "total_charge": "arithmetic",
    "agent": "copy",
}
BIND = {
    "full_name": ["first", "last"],
    "fee_amount": ["fee"],
    "total_charge": ["charge", "fee"],
    "agent": ["agent"],
}
SCORES = {c: {f: -3.0 for f in KINDS} for c in ROWS[0]}
for f, cols in BIND.items():
    for c in cols:
        SCORES[c][f] = 3.0


class Fake:
    """Answers every stage from the script: kind, column scores, and one column per role."""

    low = None  # (field, role index, probability) to move one role near the floor

    def __init__(self):
        self.bind_calls = 0
        self.bind_batches = 0

    def decide(self, text, questions):
        if any(q.name.startswith("bind:") for q in questions):
            self.bind_batches += 1
        out = []
        for q in questions:
            if q.name == "field":
                col = text.split("\n")[0].removeprefix("Input column: ")
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        probabilities={},
                        raw_scores=SCORES[col],
                    )
                )
            elif q.name == "kind":
                field = text.split("\n")[0].removeprefix("Output field: ")
                p = {n: (0.9 if n == KINDS[field] else 0.02) for n in KINDS.values()}
                p = {n: p.get(n, 0.01) for n in [c.value for c in q.choices]}
                out.append(
                    Answer("choice", q.name, choice=KINDS[field], probabilities=p)
                )
            elif q.name.startswith("bind:"):
                self.bind_calls += 1
                _, field, i = q.name.split(":")
                i = int(i)
                names = [c.value for c in q.choices]
                target = BIND[field][i]
                p = 0.9
                if self.low and self.low[:2] == (field, i):
                    p = self.low[2]
                probs = {n: (p if n == target else 0.01) for n in names}
                out.append(
                    Answer(
                        "choice",
                        q.name,
                        choice=target,
                        confidence=p,
                        probabilities=probs,
                    )
                )
            else:
                raise AssertionError(f"unexpected question {q.name}")
        return Response("fake", out, {})


class Flaky(Fake):
    """Refuses the first batch of role questions, then answers normally."""

    def __init__(self):
        super().__init__()
        self.refused = False

    def decide(self, text, questions):
        if not self.refused and any(q.name.startswith("bind:") for q in questions):
            self.refused = True
            self.bind_batches += 1
            return Response(
                "flaky",
                [Answer("refusal", q.name, message="not now") for q in questions],
                {},
            )
        return super().decide(text, questions)


def test_the_graph_runs_every_field_and_commits_the_converted_table():
    run = run_graph(ROWS, Fake(), KNOWLEDGE, thread_id="ok", retries=1)
    assert run.done and not run.paused
    assert run.header == ["full_name", "fee_amount", "total_charge", "agent"]
    assert run.converted[0] == ["Budi Halim", 10, 110, "A1"]
    assert run.converted[1] == ["Siti Wijaya", 20, 220, "A2"]


def test_a_field_near_the_floor_pauses_and_accepting_it_fills_the_field():
    fake = Fake()
    fake.low = ("total_charge", 0, 0.65)
    run = run_graph(ROWS, fake, KNOWLEDGE, thread_id="review", retries=1)
    assert run.paused and not run.done
    pending = run.result["__interrupt__"][0].value
    assert "total_charge" in pending["fields"]
    assert pending["fields"]["total_charge"]["reason"] == "below the fill floor"
    run.resume({"total_charge": "accept"})
    assert run.done
    assert run.converted[0][2] == 110


def test_a_review_can_leave_the_field_empty():
    fake = Fake()
    fake.low = ("total_charge", 0, 0.65)
    run = run_graph(ROWS, fake, KNOWLEDGE, thread_id="review-empty", retries=1)
    run.resume({"total_charge": "empty"})
    assert run.done
    assert run.converted[0][2] is None


def test_a_refusal_is_retried_once_and_then_commits():
    fake = Flaky()
    run = run_graph(ROWS, fake, KNOWLEDGE, thread_id="retry", retries=2)
    assert run.done
    assert fake.bind_batches == 2
    assert run.converted[0] == ["Budi Halim", 10, 110, "A1"]


def test_a_refusal_with_no_retry_fails_closed_and_leaves_the_fields_empty():
    fake = Flaky()
    run = run_graph(ROWS, fake, KNOWLEDGE, thread_id="closed", retries=1)
    assert run.done
    assert run.converted[0] == [None, None, None, None]


class Transient(Fake):
    """Raises once on the role batch, like a transient provider failure."""

    def __init__(self):
        super().__init__()
        self.raised = False

    def decide(self, text, questions):
        if not self.raised and any(q.name.startswith("bind:") for q in questions):
            self.raised = True
            self.bind_batches += 1
            raise RuntimeError("provider hiccup")
        return super().decide(text, questions)


class Broken(Fake):
    """Always fails the role batch, so the graph must degrade instead of raising."""

    def decide(self, text, questions):
        if any(q.name.startswith("bind:") for q in questions):
            raise RuntimeError("model down")
        return super().decide(text, questions)


class BrokenKinds(Fake):
    """Fails the first model stage, so kinds must degrade too."""

    def decide(self, text, questions):
        if any(q.name == "kind" for q in questions):
            raise RuntimeError("kind model down")
        return super().decide(text, questions)


def test_a_transient_exception_is_retried_by_the_node_retry_policy():
    fake = Transient()
    run = run_graph(
        ROWS,
        fake,
        KNOWLEDGE,
        thread_id="transient",
        retries=1,
        retry=retry_policy(attempts=2, interval=0.01, jitter=False),
    )
    assert run.done and fake.raised and fake.bind_batches == 2
    assert run.converted[0] == ["Budi Halim", 10, 110, "A1"]


def test_an_exhausted_retry_degrades_to_exact_matches_and_commits():
    run = run_graph(
        ROWS,
        Broken(),
        KNOWLEDGE,
        thread_id="degraded",
        retries=1,
        retry=retry_policy(attempts=2, interval=0.01, jitter=False),
    )
    assert run.done and run.degraded
    assert any("decide failed after retries" in e for e in run.errors)
    assert run.converted[0] == [
        None,
        None,
        None,
        "A1",
    ]  # only the exact-name copy survives


def test_a_kind_failure_after_retries_degrades_instead_of_raising():
    run = run_graph(
        ROWS,
        BrokenKinds(),
        KNOWLEDGE,
        thread_id="kinds-down",
        retries=1,
        retry=retry_policy(attempts=2, interval=0.01, jitter=False),
    )
    assert run.done and run.degraded
    assert any("kinds failed after retries" in e for e in run.errors)
    assert run.converted[0] == [None, None, None, None]


def test_repair_tries_the_next_candidate_that_evaluates():
    bad = {"op": "div", "args": [{"src": "x"}, {"src": "zero"}]}
    good = {"op": "add", "args": [{"src": "x"}, {"lit": 1}]}
    rec = {
        "best": bad,
        "best_text": "x / zero",
        "cands": ["x / zero", "x + 1"],
        "cand_nodes": [bad, good],
        "kind": "arithmetic",
        "cols": ["x", "zero"],
        "p_present": 0.9,
        "p_none": 0.1,
        "method": "expression",
    }
    out = _repair(
        {
            "rows": [{"x": 10, "zero": 0}],
            "fields": {"f": rec},
            "repair": {"f": "fails"},
            "repair_attempts": 0,
        }
    )
    assert out["fields"]["f"]["best"] == good
    assert out["fields"]["f"]["best_text"] == "x + 1"
    assert out["repair_attempts"] == 1


def test_repair_without_an_alternative_fails_closed():
    bad = {"op": "div", "args": [{"src": "x"}, {"src": "zero"}]}
    rec = {
        "best": bad,
        "best_text": "x / zero",
        "cands": ["x / zero"],
        "cand_nodes": [bad],
        "kind": "arithmetic",
        "cols": ["x", "zero"],
        "p_present": 0.9,
        "p_none": 0.1,
        "method": "expression",
    }
    out = _repair(
        {
            "rows": [{"x": 10, "zero": 0}],
            "fields": {"f": rec},
            "repair": {"f": "fails"},
            "repair_attempts": 0,
        }
    )
    assert out["fields"]["f"]["best"] is None
    assert out["fields"]["f"]["method"] == "repair-empty"
