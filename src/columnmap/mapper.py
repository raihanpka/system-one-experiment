# The mapping, in the order the Decisions model is asked:
#   kind     what kind of value each output field is (copy, join, arithmetic, ...), from its description alone, once
#   stage 1  which input columns feed which output field (one question per input column)
#   stage 2  per field: the named roles of its description, each bound to one column (or none), then the expression
#            that code builds from those columns. A description the role parser cannot read falls back to the column-set
#            question plus an explicit presence question.
# A field is filled when its source is present (p_present reaches the fill score) and "none" is not the better answer.
# Candidate expressions and converting are deterministic. No ground truth goes in. The stage functions keep no state.

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from random import Random

from .decider import Decider, Question
from .expressions import (
    candidates,
    column_unit,
    compatible,
    covering,
    output_unit,
    result_unit,
    safe_evaluate,
    size,
    src_names,
)
from .knowledge import Knowledge, Output
from .prompts import (
    DEFAULT,
    NONE,
    bind_question,
    expression_question,
    expression_text,
    field_state,
    file_state,
    input_state,
    kind_question,
    kind_state,
    mapping_question,
    option_text,
    pool_state,
    presence_question,
    safe_label,
    set_label,
    set_question,
)
from .roles import shape
from .schema import Schema, extract

KIND_SHUFFLES = (
    3  # the kind question is asked with the options in this many orders and averaged
)
TOP_COLUMNS = 5  # columns with the best mapping scores; the fallback set is chosen among their subsets
MAX_CANDIDATES = (
    24  # options in one expression question; more made the server drop the connection
)
BATCH_FIELDS = 8  # fields whose role questions share one request; the model answers them in parallel
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
    decider: Decider,
    rows: list[dict],
    knowledge: Knowledge,
    schema: Schema | None = None,
    registry=DEFAULT,
) -> tuple[dict, dict]:
    schema = schema or extract(rows)
    question = mapping_question(knowledge.outputs, registry)
    matrix, raw = {}, {}
    for column in schema.columns:
        name = column.name
        state = input_state(
            name,
            column.samples,
            kind=column.type,
            unit=column.unit,
            null_rate=column.null_rate,
            registry=registry,
        )
        res = decider.decide(state, [question])
        raw[name] = {
            "model": res.model,
            "input": state,
            "answers": [asdict(a) for a in res.answers],
            "usage": res.usage,
        }
        answer = res.answers[0] if res.answers else None
        if answer is None or answer.raw_scores is None:
            raw[name]["error"] = (
                "no answer"  # refusal or missing key: this column scores zero
            )
            matrix[name] = {o.name: 0.0 for o in knowledge.outputs}
        else:
            matrix[name] = dict(answer.raw_scores)
    return matrix, raw


def top_columns(matrix: dict, field: str, k: int = TOP_COLUMNS) -> list[str]:
    return [c for _, c in sorted((-m[field], c) for c, m in matrix.items())[:k]]


# which family an expression belongs to, as named in the kind list
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
    decider: Decider,
    knowledge: Knowledge,
    shuffles: int = KIND_SHUFFLES,
    registry=DEFAULT,
) -> dict:
    q = kind_question(registry)
    out = {}
    for o in knowledge.outputs:
        total: dict[str, float] = {}
        for i in range(shuffles):
            choices = list(q.choices)
            Random(f"{o.name}:{i}").shuffle(choices)
            asked = Question(
                q.type, q.instructions, name=q.name, choices=tuple(choices)
            )
            res = decider.decide(kind_state(o, registry), [asked])
            answer = res.answers[0] if res.answers else None
            probs = answer.probabilities if answer is not None else None
            if not probs:
                continue  # a refusal on one order does not decide the field
            for k, p in probs.items():
                total[k] = total.get(k, 0.0) + p / shuffles
        out[o.name] = max(total, key=total.get) if total else "copy"
    return out


def empty_record() -> dict:
    return {
        "kind": None,
        "shape": None,
        "roles": [],
        "cols": [],
        "top_cols": [],
        "set_scores": {},
        "p_none": None,
        "p_present": None,
        "cands": [],
        "cand_nodes": [],
        "p_expr": None,
        "best": None,
        "best_text": None,
        "score": None,
        "method": "unmapped",
    }


def _pick(probs: dict | None, labels: list[str]) -> tuple[str, float, float]:
    probs = probs or {}
    best = max(labels, key=lambda x: probs.get(x, 0.0))
    return best, float(probs.get(best, 0.0)), float(probs.get(NONE, 0.0))


# a role may only see columns of a compatible unit; if the filter would hide every column, keep them all
# and let the check after the answer reject a wrong unit instead of making the model guess blindly.
def _columns_for(role, source, units: dict) -> list[str]:
    names = source.names() if hasattr(source, "names") else list(source[0])
    allowed = [n for n in names if compatible(role.unit, units[n])]
    return allowed or names


# a code column is easier to bind when the model can see the code table it will be translated with
def _lookup_hint(kind: str, columns: list[str], rows: list[dict], tables) -> str | None:
    if kind != "lookup" or not tables:
        return None
    for table in tables:
        if any(covering([r[column] for r in rows], [table]) for column in columns):
            items = list(table.items())[:50]
            return "; ".join(f"{k} -> {v}" for k, v in items)
    return None


# the expressions that build a field from exactly this column set, of the field's kind (a plain copy when there is no other)
def set_candidates(chosen, rows, out: Output, kind: str, tables) -> list[dict]:
    cols = [(n, [r[n] for r in rows]) for n in chosen]
    units = {n: column_unit(n, vals) for n, vals in cols}
    out_unit = output_unit(out)

    def unit_ok(c: dict) -> bool:
        if out.type == "number" and any(units[n] == "id" for n in src_names(c)):
            return False  # a policy number is not an amount, even when it is all digits
        got = result_unit(c, units)
        if out_unit == "amount":
            return got == "amount" or (got == "number" and "src" in c)
        if out_unit == "rate":
            return got == "rate"
        if out_unit == "count":
            return got in ("count", "number")
        if out_unit == "date":
            return got == "date"
        return True

    found = [
        c
        for c in candidates(cols, out_type=out.type, tables=tables)
        if set(src_names(c)) == set(chosen) and unit_ok(c)
    ]
    own = [c for c in found if family(c) == kind]
    if not own and len(chosen) == 1:
        own = [c for c in found if family(c) == "copy"]
    return sorted(own, key=lambda c: (size(c), expression_text(c)))[:MAX_CANDIDATES]


# choice labels are prompt text too: keep the raw column for the mapping, show a collapsed label
def _safe_labels(names: list[str]) -> dict[str, str]:
    out = {}
    for i, name in enumerate(names):
        label = safe_label(name) or f"column_{i + 1}"
        if label == NONE:
            label = f"column: {label}"
        if label in out and out[label] != name:
            suffix = 2
            while f"{label}#{suffix}" in out:
                suffix += 1
            label = f"{label}#{suffix}"
        out[label] = name
    return out


# one request for a group of fields: every role question runs against the whole file in parallel
def _bind_chunk(
    decider: Decider,
    chunk: list[tuple[Output, object]],
    rows: list[dict],
    schema: Schema,
    matrix: dict,
    kinds: dict,
    units: dict,
    tables,
    registry=DEFAULT,
) -> dict:
    questions, meta = [], {}
    for out, shp in chunk:
        for i, role in enumerate(shp.roles):
            name = f"bind:{out.name}:{i}"
            raw = _columns_for(role, schema, units)
            meta[name] = _safe_labels(raw)
            columns = list(meta[name])
            if not columns:
                continue  # no compatible column: the role is absent, no one-choice question
            hint = _lookup_hint(kinds[out.name], raw, rows, tables)
            questions.append(bind_question(out, role, columns, name, registry, hint))
    answers = {
        a.name: a
        for a in decider.decide(file_state(schema, registry), questions).answers
    }
    fields = {}
    for out, shp in chunk:
        rec = {
            **empty_record(),
            "kind": kinds[out.name],
            "shape": shp.op,
            "top_cols": top_columns(matrix, out.name),
        }
        picked, role_p, p_none, failed = [], [], 0.0, False
        for i, role in enumerate(shp.roles):
            a = answers.get(f"bind:{out.name}:{i}")
            probs = a.probabilities if a is not None else None
            labels = list(meta[f"bind:{out.name}:{i}"]) + [NONE]
            label, p, pn = _pick(probs, labels)
            raw = meta[f"bind:{out.name}:{i}"].get(label)
            p_none = max(p_none, pn)
            role_p.append(p)
            if (
                not probs
                or label == NONE
                or raw is None
                or not compatible(role.unit, units[raw])
            ):
                failed = True
            picked.append(raw)
        p_present = 0.0 if failed else min(role_p, default=0.0)
        rec.update(
            {
                "roles": [[role.phrase, picked[i]] for i, role in enumerate(shp.roles)],
                "cols": [c for c in picked if c],
                "p_present": p_present,
                "p_none": p_none,
                "score": p_present,
            }
        )
        if failed:
            rec["method"] = "none"
            fields[out.name] = rec
            continue
        best = shp.build([{"src": c} for c in picked])
        if shp.op == "lookup":
            table = next(iter(covering([r[picked[0]] for r in rows], tables)), None)
            best = (
                {"op": "lookup", "map": table, "args": [{"src": picked[0]}]}
                if table
                else {"src": picked[0]}
            )
        rec.update(
            {
                "cands": [expression_text(best)],
                "cand_nodes": [best],
                "best": best,
                "best_text": expression_text(best),
                "method": "copy" if "src" in best else "expression",
            }
        )
        fields[out.name] = rec
    return fields


# the fallback for a description the role parser cannot read: the old column-set question plus an explicit
# presence question, so "does the source exist" is not inferred from the joint set probability.
def _field(
    decider, out: Output, kind, rows, matrix, all_columns, tables, registry=DEFAULT
) -> dict:
    names = top_columns(matrix, out.name)
    sets = [s for n in SIZES[kind] for s in combinations(names, n)]
    raw_labels = [set_label(s) for s in sets]
    raw_to_set = dict(zip(raw_labels, sets))
    label_to_set = {
        label: raw_to_set[raw] for label, raw in _safe_labels(raw_labels).items()
    }
    state = field_state(out, all_columns, names, rows, registry)
    res = decider.decide(
        state,
        [
            presence_question(out, registry=registry),
            set_question(out, list(label_to_set), registry),
        ],
    )
    answers = {a.name: a for a in res.answers}
    presence = answers.get("presence")
    probs = answers.get("set").probabilities if answers.get("set") else {}
    rec = {
        **empty_record(),
        "kind": kind,
        "top_cols": names,
        "p_present": (
            presence.probability
            if presence is not None and presence.type == "predicate"
            else 0.0  # a refusal or a missing answer fails closed
        ),
        "p_none": probs.get(NONE),
    }
    if not probs:
        rec["method"] = "none"
        return rec
    best_label = max(label_to_set, key=lambda label: probs.get(label, 0.0))
    chosen = label_to_set[best_label]
    rec.update(
        {
            "cols": list(chosen),
            "set_scores": {x: probs[x] for x in label_to_set if x in probs},
            "score": probs.get(best_label),
        }
    )
    cands = set_candidates(chosen, rows, out, kind, tables)
    if not cands:
        rec["method"] = "none"
        return rec  # no rule in the catalog builds this field from these columns
    if len(cands) > 1:
        options = [option_text(c, rows) for c in cands]
        pe = (
            decider.decide(
                pool_state(list(chosen), rows, registry),
                [expression_question(out, options, registry)],
            )
            .answers[0]
            .probabilities
        )
        k = max(range(len(cands)), key=lambda i: pe.get(options[i], 0.0))
        rec["p_expr"] = pe.get(options[k])
    else:
        k = 0
    best = cands[k]
    rec.update(
        {
            "cands": [expression_text(c) for c in cands],
            "cand_nodes": cands,
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
    schema: Schema | None = None,
    registry=DEFAULT,
) -> dict:
    kinds = kinds or infer_kinds(decider, knowledge, registry=registry)
    schema = schema or extract(rows)
    units = schema.units()
    shaped, fallback = [], []
    for out in knowledge.outputs:
        shp = shape(out, kinds[out.name])
        (shaped.append((out, shp)) if shp else fallback.append(out))
    fields = {}
    for i in range(0, len(shaped), BATCH_FIELDS):
        fields.update(
            _bind_chunk(
                decider,
                shaped[i : i + BATCH_FIELDS],
                rows,
                schema,
                matrix,
                kinds,
                units,
                knowledge.tables,
                registry,
            )
        )
    for out in fallback:
        fields[out.name] = _field(
            decider,
            out,
            kinds[out.name],
            rows,
            matrix,
            schema.names(),
            knowledge.tables,
            registry,
        )
    return drop_double_claims(fields)


# two copied fields rarely take the same column: the field with the lower score loses it (score 0, winner named)
def drop_double_claims(fields: dict) -> dict:
    winner: dict[tuple, str] = {}
    for name, rec in sorted(fields.items(), key=lambda kv: -(kv[1]["score"] or 0)):
        if rec["kind"] != "copy" or rec["best"] is None:
            continue
        key = tuple(rec["cols"])
        if key in winner:
            rec["score"], rec["p_present"], rec["claimed_by"] = (
                0.0,
                0.0,
                winner[key],
            )
        else:
            winner[key] = name
    return fields


# the probability of the column set that fills a field. Derived fields (year, join, arithmetic, ...) need more than a field
# copied from one column, because a copied field is also protected by drop_double_claims. Both were set on the dev split.
# presence: the floor for p_present. none_ceiling: a field stays empty when "none" is at least this likely, so a coin
# flip between a column and none never fills a file with a guess.
@dataclass(frozen=True)
class Fill:
    derived: float = 0.75
    copy: float = 0.4
    none_ceiling: float = 0.5


FILL = Fill()


def floor_of(rec: dict, fill: Fill = FILL) -> float:
    return fill.copy if rec.get("kind") in ("copy", "year", "lookup") else fill.derived


# a field is filled when it has an expression and its source is present; the others stay empty.
# Records from before the presence gate (no "p_present" key) keep the old score rule for the report.
def is_filled(rec: dict, fill: Fill = FILL) -> bool:
    if rec["best"] is None:
        return False
    floor = floor_of(rec, fill)
    if "p_present" not in rec:
        return (rec.get("score") or 0.0) >= floor
    present = rec.get("p_present")
    if present is None:
        return False  # a refusal or a missing presence answer must never fill
    return present >= floor and (rec.get("p_none") or 0.0) <= fill.none_ceiling


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
    fields: dict  # field -> record (expression, columns, score, presence)
    order: list[str]
    fill: Fill = FILL  # the policy this mapping was built with

    def convert(self, rows: list[dict], fill: Fill | None = None):
        return convert(self.fields, rows, self.order, fill or self.fill)


class Mapper:
    def __init__(
        self,
        decider: Decider,
        knowledge: Knowledge,
        registry=None,
        fill: Fill | None = None,
    ):
        from .pipeline import (
            Pipeline,
        )  # lazy: pipeline imports Mapping from this module

        self.decider, self.knowledge = decider, knowledge
        self.pipeline = Pipeline(decider, knowledge, registry, fill)
        self.registry = self.pipeline.registry
        self.fill = self.pipeline.fill

    @property
    def kinds(self) -> dict:
        return self.pipeline.kinds()

    def map(self, rows: list[dict]) -> Mapping:
        return self.pipeline.run(rows)
