# The mapping, in the order the Decisions model is asked:
#   kind     what kind of value each output field is (copy, join, arithmetic, ...), from its description alone, once
#   stage 1  which input columns feed which output field (one question per input column)
#   stage 2  per field: which SET of columns it needs (or none), then which expression of that kind builds it
# A field is filled when the probability of its column set reaches the fill score. Candidate expressions and converting
# are deterministic. No ground truth goes in. The stage functions keep no state, so a caller can log or resume them.

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from random import Random

from .decider import Decider, Question
from .expressions import candidates, safe_evaluate, size, src_names
from .knowledge import Knowledge, Output
from .prompts import (
    NONE,
    expression_question,
    expression_text,
    field_state,
    input_state,
    kind_question,
    kind_state,
    mapping_question,
    option_text,
    pool_state,
    samples,
    set_label,
    set_question,
)

KIND_SHUFFLES = (
    3  # the kind question is asked with the options in this many orders and averaged
)
TOP_COLUMNS = 5  # columns with the best mapping scores; the column set of a field is chosen among their subsets
MAX_CANDIDATES = (
    24  # options in one expression question; more made the server drop the connection
)
SIZES = {
    "copy": (1,),
    "year": (1,),
    "lookup": (1,),
    "date_difference": (2,),
    "arithmetic": (2, 3),
    "join": (2, 3),
}


# stage 1: matrix[column][field] = raw score; raw keeps the model's own answer per column for logging
def score_columns(
    decider: Decider, rows: list[dict], knowledge: Knowledge
) -> tuple[dict, dict]:
    question = mapping_question(knowledge.outputs)
    matrix, raw = {}, {}
    for name in rows[0]:
        state = input_state(name, samples(rows, name))
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


# which family an expression belongs to, as named in KINDS
def family(ast: dict) -> str:
    if "src" in ast:
        return "copy"
    return {
        "concat": "join",
        "add": "arithmetic",
        "sub": "arithmetic",
        "mul": "arithmetic",
        "div": "arithmetic",
        "days_between": "date_difference",
        "year": "year",
        "lookup": "lookup",
    }[ast["op"]]


# the kind of every output field, read from its description (once per knowledge, not per file); the option order is
# shuffled and the answers averaged, because one fixed order made a field flip between two kinds from run to run
def infer_kinds(
    decider: Decider, knowledge: Knowledge, shuffles: int = KIND_SHUFFLES
) -> dict:
    q = kind_question()
    out = {}
    for o in knowledge.outputs:
        total: dict[str, float] = {}
        for i in range(shuffles):
            choices = list(q.choices)
            Random(f"{o.name}:{i}").shuffle(choices)
            asked = Question(
                q.type, q.instructions, name=q.name, choices=tuple(choices)
            )
            for k, p in (
                decider.decide(kind_state(o), [asked]).answers[0].probabilities.items()
            ):
                total[k] = total.get(k, 0.0) + p / shuffles
        out[o.name] = max(total, key=total.get)
    return out


def _empty() -> dict:
    return {
        "kind": None,
        "cols": [],
        "top_cols": [],
        "set_scores": {},
        "p_none": None,
        "cands": [],
        "p_expr": None,
        "best": None,
        "best_text": None,
        "score": None,
        "method": "unmapped",
    }


# the expressions that build a field from exactly this column set, of the field's kind (a plain copy when there is no other)
def set_candidates(chosen, rows, out: Output, kind: str, tables) -> list[dict]:
    cols = [(n, [r[n] for r in rows]) for n in chosen]
    found = [
        c
        for c in candidates(cols, out_type=out.type, tables=tables)
        if set(src_names(c)) == set(chosen)
    ]
    own = [c for c in found if family(c) == kind]
    if not own and len(chosen) == 1:
        own = [c for c in found if family(c) == "copy"]
    return sorted(own, key=lambda c: (size(c), expression_text(c)))[:MAX_CANDIDATES]


def _field(decider, out: Output, kind, rows, matrix, all_columns, tables) -> dict:
    names = top_columns(matrix, out.name)
    sets = [s for n in SIZES[kind] for s in combinations(names, n)]
    labels = [set_label(s) for s in sets]
    state = field_state(out, all_columns, names, rows)
    probs = decider.decide(state, [set_question(out, labels)]).answers[0].probabilities
    best_i = max(range(len(sets)), key=lambda i: probs[labels[i]])
    chosen = sets[best_i]
    rec = {
        **_empty(),
        "kind": kind,
        "top_cols": names,
        "cols": list(chosen),
        "set_scores": {x: probs[x] for x in labels},
        "p_none": probs[NONE],
        "score": probs[labels[best_i]],
    }
    cands = set_candidates(chosen, rows, out, kind, tables)
    if not cands:
        return rec  # no rule in the catalog builds this field from these columns
    if len(cands) > 1:
        options = [option_text(c, rows) for c in cands]
        pe = (
            decider.decide(
                pool_state(list(chosen), rows), [expression_question(out, options)]
            )
            .answers[0]
            .probabilities
        )
        k = max(range(len(cands)), key=lambda i: pe[options[i]])
        rec["p_expr"] = pe[options[k]]
    else:
        k = 0
    best = cands[k]
    rec.update(
        {
            "cands": [expression_text(c) for c in cands],
            "best": best,
            "best_text": expression_text(best),
            "method": "copy" if "src" in best else "expression",
        }
    )
    return rec


# stage 2: one record per output field; `kinds` comes from infer_kinds (pass it in to reuse it across files)
def choose_expressions(
    decider: Decider,
    rows: list[dict],
    matrix: dict,
    knowledge: Knowledge,
    kinds: dict | None = None,
) -> dict:
    kinds = kinds or infer_kinds(decider, knowledge)
    all_columns = list(matrix)
    fields = {
        o.name: _field(
            decider, o, kinds[o.name], rows, matrix, all_columns, knowledge.tables
        )
        for o in knowledge.outputs
    }
    return drop_double_claims(fields)


# two copied fields rarely take the same column: the field with the lower score loses it (score 0, winner named)
def drop_double_claims(fields: dict) -> dict:
    winner: dict[tuple, str] = {}
    for name, rec in sorted(fields.items(), key=lambda kv: -(kv[1]["score"] or 0)):
        if rec["kind"] != "copy" or rec["best"] is None:
            continue
        key = tuple(rec["cols"])
        if key in winner:
            rec["score"], rec["claimed_by"] = 0.0, winner[key]
        else:
            winner[key] = name
    return fields


# the probability of the column set that fills a field. Derived fields (year, join, arithmetic, ...) need more than a field
# copied from one column, because a copied field is also protected by drop_double_claims. Both were set on the dev split.
@dataclass(frozen=True)
class Fill:
    derived: float = 0.7
    copy: float = 0.4


FILL = Fill()


# a field is filled when it has an expression and its score reaches the fill score of its kind; the others stay empty
def is_filled(rec: dict, fill: Fill = FILL) -> bool:
    if rec["best"] is None:
        return False
    return rec["score"] >= (fill.copy if rec.get("kind") == "copy" else fill.derived)


def convert(
    fields: dict, rows: list[dict], order: list[str], fill: Fill = FILL
) -> tuple[list[str], list[list]]:
    cols = [
        [safe_evaluate(fields[f]["best"], r) for r in rows]
        if is_filled(fields[f], fill)
        else [None] * len(rows)
        for f in order
    ]
    return order, [list(r) for r in zip(*cols)]


@dataclass
class Mapping:
    matrix: dict  # column -> field -> raw score
    fields: dict  # field -> record (expression, columns, score)
    order: list[str]

    def convert(self, rows: list[dict], fill: Fill = FILL):
        return convert(self.fields, rows, self.order, fill)


class Mapper:
    def __init__(self, decider: Decider, knowledge: Knowledge):
        self.decider, self.knowledge = decider, knowledge
        self.kinds: dict | None = (
            None  # read from the descriptions on first use, then reused for every table
        )

    def map(self, rows: list[dict]) -> Mapping:
        if self.kinds is None:
            self.kinds = infer_kinds(self.decider, self.knowledge)
        matrix, _ = score_columns(self.decider, rows, self.knowledge)
        fields = choose_expressions(
            self.decider, rows, matrix, self.knowledge, self.kinds
        )
        return Mapping(matrix, fields, [o.name for o in self.knowledge.outputs])
