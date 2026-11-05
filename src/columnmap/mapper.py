# The mapping. Stage 1 scores every input column against every output field with the Decisions model.
# Stage 2 builds candidate expressions from the best columns and lets the model rank them; no ground truth goes in.
# Converting is deterministic. The stage functions keep no state, so a caller can log or resume them; Mapper runs both.

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

from .decider import Decider
from .expressions import candidates, safe_evaluate, size, src_names
from .knowledge import Knowledge, Output
from .prompts import (
    check_question,
    expression_text,
    input_state,
    mapping_question,
    option_text,
    pick_question,
    pool_state,
    samples,
)

FILL_SCORE = 0.9  # score the best expression of a field needs to fill it
TOP_COLUMNS = 4  # columns with the best mapping scores; candidates use every subset of up to three of them
MAX_CANDIDATES = (
    24  # options in one pick question; more made the server drop the connection
)
SHORTLIST = 5  # candidates the model checks one by one
MARGIN = (
    0.2  # a simpler expression wins when its score is within this margin of the best
)


# stage 1: matrix[column][field] = raw score; raw keeps the model's own answer per column for logging
def score_columns(
    decider: Decider, rows: list[dict], knowledge: Knowledge, context: str = "samples"
) -> tuple[dict, dict]:
    question = mapping_question(knowledge.outputs)
    matrix, raw = {}, {}
    for name in rows[0]:
        state = input_state(name, context, samples(rows, name))
        res = decider.decide(state, [question])
        raw[name] = {
            "model": res.model,
            "input": state,
            "answers": [asdict(a) for a in res.answers],
            "usage": res.usage,
        }
        matrix[name] = dict(res.answers[0].raw_scores)
    return matrix, raw


def top_columns(matrix: dict, field: str, k: int = TOP_COLUMNS) -> list[str]:
    return [c for _, c in sorted((-m[field], c) for c, m in matrix.items())[:k]]


# the candidate expressions of one field: every subset of the top columns, simplest first
def field_candidates(names, rows, out_type, tables) -> list[dict]:
    seen, out = set(), []
    for r in (1, 2, 3):
        for sub in combinations(names, r):
            for c in candidates(
                [(n, [row[n] for row in rows]) for n in sub],
                out_type=out_type,
                tables=tables,
            ):
                key = expression_text(c)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
    out.sort(
        key=lambda c: (size(c), expression_text(c))
    )  # by nodes, not text: a lookup is long to print but simple
    return out[:MAX_CANDIDATES]


def _empty() -> dict:
    return {
        "cols": [],
        "top_cols": [],
        "cands": [],
        "shortlist": [],
        "scores": [],
        "best": None,
        "best_text": None,
        "score": None,
        "method": "unmapped",
    }


def _field(decider, out: Output, rows, matrix, tables) -> dict:
    names = top_columns(matrix, out.name)
    rec = {**_empty(), "top_cols": names}
    cands = field_candidates(names, rows, out.type, tables)
    if not cands:
        return rec  # no rule in the catalog fits these columns
    state = pool_state(names, rows)
    if len(cands) > SHORTLIST:
        options = [option_text(c, rows) for c in cands]
        probs = (
            decider.decide(state, [pick_question(out.name, options)])
            .answers[0]
            .probabilities
        )
        shortlist = sorted(range(len(cands)), key=lambda i: -probs[options[i]])[
            :SHORTLIST
        ]
    else:
        shortlist = list(range(len(cands)))
    checks = [
        check_question(out.name, out.description, cands[i], rows, f"c{k}")
        for k, i in enumerate(shortlist)
    ]
    scores = [a.probability for a in decider.decide(state, checks).answers]
    top = max(scores)
    near = [k for k, s in enumerate(scores) if s >= top - MARGIN]
    k = min(near, key=lambda k: (size(cands[shortlist[k]]), -scores[k]))
    best = cands[shortlist[k]]
    rec.update(
        {
            "cols": src_names(best),
            "cands": [expression_text(c) for c in cands],
            "shortlist": shortlist,
            "scores": scores,
            "best": best,
            "best_text": expression_text(best),
            "score": scores[k],
            "method": "copy" if "src" in best else "expression",
        }
    )
    return rec


# stage 2: one record per output field
def choose_expressions(
    decider: Decider, rows: list[dict], matrix: dict, knowledge: Knowledge
) -> dict:
    return {
        o.name: _field(decider, o, rows, matrix, knowledge.tables)
        for o in knowledge.outputs
    }


# a field is filled when it has an expression and its score reaches fill_score; the others stay empty
def is_filled(rec: dict, fill_score: float = FILL_SCORE) -> bool:
    return rec["best"] is not None and rec["score"] >= fill_score


def convert(
    fields: dict, rows: list[dict], order: list[str], fill_score: float = FILL_SCORE
) -> tuple[list[str], list[list]]:
    cols = [
        [safe_evaluate(fields[f]["best"], r) for r in rows]
        if is_filled(fields[f], fill_score)
        else [None] * len(rows)
        for f in order
    ]
    return order, [list(r) for r in zip(*cols)]


@dataclass
class Mapping:
    matrix: dict  # column -> field -> raw score
    fields: dict  # field -> record (expression, columns, score)
    order: list[str]

    def convert(self, rows: list[dict], fill_score: float = FILL_SCORE):
        return convert(self.fields, rows, self.order, fill_score)


class Mapper:
    def __init__(
        self, decider: Decider, knowledge: Knowledge, context: str = "samples"
    ):
        self.decider, self.knowledge, self.context = decider, knowledge, context

    def map(self, rows: list[dict]) -> Mapping:
        matrix, _ = score_columns(self.decider, rows, self.knowledge, self.context)
        fields = choose_expressions(self.decider, rows, matrix, self.knowledge)
        return Mapping(matrix, fields, [o.name for o in self.knowledge.outputs])
