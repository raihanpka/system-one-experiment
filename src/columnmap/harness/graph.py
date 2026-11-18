# The mapping as a LangGraph agent. Code owns the control flow; the Decisions model only answers atomic
# questions. The graph has four reliability layers:
#   retry     LangGraph RetryPolicy on the model-calling nodes (transient errors, with backoff)
#   degrade   an error handler after retries is exhausted: exact name matches only, everything else empty
#   repair    a bounded loop that tries the next candidate when an expression fails on the sample rows
#   review    an interrupt for fields close to the fill policy, resumed with accept or empty
# The library core never imports this module; it needs the optional graph extra: uv sync --extra graph.

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, default_retry_on, interrupt

from ..decider import Decider
from ..expressions import safe_evaluate
from ..knowledge import Knowledge
from ..mapper import Fill, convert, empty_record, floor_of, is_filled
from ..pipeline import Pipeline
from ..prompts import expression_text
from ..schema import extract
from .retry import Retrying

REVIEW_MARGIN = 0.15  # a field this close to the floor is worth a human look
MAX_REPAIRS = 1


class MappingState(TypedDict):
    rows: list[dict]
    matrix: dict
    kinds: dict
    fields: dict
    review: dict
    repair: dict
    repair_attempts: int
    degraded: bool
    header: list[str]
    converted: list[list]
    status: str
    stages: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


@dataclass
class Context:
    pipeline: Pipeline
    fill: Fill = Fill()


# a transient model/network failure is retried; a programming or data error is not
def _retry_on(error: BaseException) -> bool:
    if isinstance(error, (ValueError, TypeError, KeyError, IndexError)):
        return False
    return isinstance(error, (RuntimeError, OSError, TimeoutError)) or default_retry_on(
        error
    )


def retry_policy(
    attempts: int = 3, interval: float = 0.5, jitter: bool = True
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=attempts,
        initial_interval=interval,
        backoff_factor=2.0,
        jitter=jitter,
        retry_on=_retry_on,
    )


def _kinds(state: MappingState, runtime: Runtime[Context]) -> dict:
    if state.get("kinds"):
        return {}
    kinds = runtime.context.pipeline.kinds()
    return {"kinds": kinds, "stages": [{"stage": "kinds", "fields": len(kinds)}]}


def _score(state: MappingState, runtime: Runtime[Context]) -> dict:
    matrix, raw = runtime.context.pipeline.score(state["rows"])
    errors = [
        f"{name}: {rec['error']}" for name, rec in raw.items() if rec.get("error")
    ]
    return {
        "matrix": matrix,
        "errors": errors,
        "stages": [{"stage": "score", "columns": len(matrix)}],
    }


def _decide(state: MappingState, runtime: Runtime[Context]) -> dict:
    fields = runtime.context.pipeline.decide(
        state["rows"], state["matrix"], state["kinds"]
    )
    return {"fields": fields, "stages": [{"stage": "decide", "fields": len(fields)}]}


def _kinds_failed(state: MappingState, error: NodeError) -> Command:
    return Command(
        update={
            "errors": [f"kinds failed after retries: {error.error}"],
            "degraded": True,
            "status": "degraded",
        },
        goto="degrade",
    )


def _score_failed(state: MappingState, error: NodeError) -> Command:
    return Command(
        update={
            "errors": [f"score failed after retries: {error.error}"],
            "degraded": True,
            "status": "degraded",
        },
        goto="degrade",
    )


def _decide_failed(state: MappingState, error: NodeError) -> Command:
    return Command(
        update={
            "errors": [f"decide failed after retries: {error.error}"],
            "degraded": True,
            "status": "degraded",
        },
        goto="degrade",
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


# the fallback when the model is unavailable: only an exact column-name match for a copied field is
# safe without a judgment. Everything else stays empty. This is deterministic and fail-closed.
def _degrade(state: MappingState, runtime: Runtime[Context]) -> dict:
    schema = extract(state["rows"])
    names = schema.names()
    fields = {}
    for out in runtime.context.pipeline.knowledge.outputs:
        kind = state.get("kinds", {}).get(out.name)
        match = next((c for c in names if _norm(c) == _norm(out.name)), None)
        rec = empty_record()
        rec["kind"] = kind
        if kind == "copy" and match:
            rec.update(
                {
                    "shape": "src",
                    "roles": [[out.description, match]],
                    "cols": [match],
                    "best": {"src": match},
                    "best_text": match,
                    "p_present": 1.0,
                    "p_none": 0.0,
                    "score": 1.0,
                    "method": "degrade-exact",
                }
            )
        else:
            rec["method"] = "degrade-empty"
        fields[out.name] = rec
    return {
        "fields": fields,
        "degraded": True,
        "status": "degraded",
        "stages": [
            {
                "stage": "degrade",
                "filled": sum(1 for r in fields.values() if r["best"]),
            }
        ],
    }


# deterministic checks: an expression that fails on the sample rows is repaired once with the next
# candidate; a field within the review margin of the fill policy goes to a human.
def _verify(state: MappingState, runtime: Runtime[Context]) -> dict:
    fill = runtime.context.fill
    repair, review = {}, {}
    for out in runtime.context.pipeline.knowledge.outputs:
        rec = state["fields"][out.name]
        if rec.get("best") is None:
            continue
        present = rec.get("p_present")
        floor = floor_of(rec, fill)
        detail = {
            "kind": rec.get("kind"),
            "columns": rec.get("cols", []),
            "expression": rec.get("best_text"),
            "p_present": present,
            "p_none": rec.get("p_none"),
        }
        if is_filled(rec, fill):
            if any(safe_evaluate(rec["best"], r) is None for r in state["rows"]):
                if state["repair_attempts"] < MAX_REPAIRS:
                    repair[out.name] = {
                        **detail,
                        "reason": "expression fails on sample rows",
                    }
                else:
                    review[out.name] = {
                        **detail,
                        "reason": "expression fails after repair",
                    }
            elif (
                rec.get("p_none") is not None
                and rec["p_none"] > fill.none_ceiling - 0.1
            ):
                review[out.name] = {
                    **detail,
                    "reason": "p_none is close to the ceiling",
                }
        elif present is not None and present >= floor - REVIEW_MARGIN:
            review[out.name] = {**detail, "reason": "below the fill floor"}
        elif rec.get("p_none") is not None and rec["p_none"] > fill.none_ceiling:
            review[out.name] = {**detail, "reason": "none wins by the ceiling"}
    return {
        "repair": repair,
        "review": review,
        "stages": [{"stage": "verify", "repair": len(repair), "review": len(review)}],
    }


def _repair(state: MappingState) -> dict:
    fields = dict(state["fields"])
    for name in state["repair"]:
        rec = dict(fields[name])
        chosen = None
        for node in rec.get("cand_nodes", []):
            if node == rec.get("best"):
                continue
            if all(safe_evaluate(node, row) is not None for row in state["rows"]):
                chosen = node
                break
        if chosen is None:
            rec.update(
                {
                    "best": None,
                    "best_text": None,
                    "p_present": 0.0,
                    "p_none": 0.0,
                    "method": "repair-empty",
                }
            )
        else:
            rec.update(
                {
                    "best": chosen,
                    "best_text": expression_text(chosen),
                    "method": "repair",
                }
            )
        fields[name] = rec
    return {
        "fields": fields,
        "repair": {},
        "repair_attempts": state["repair_attempts"] + 1,
        "stages": [{"stage": "repair", "fields": list(state["repair"])}],
    }


def _route(state: MappingState) -> str:
    if state["repair"] and state["repair_attempts"] < MAX_REPAIRS:
        return "repair"
    return "review" if state["review"] else "commit"


# one interrupt per review node. The payload is JSON-serializable, and the resume value is a dict of
# field -> "accept" or "empty". A node restarts from the top on resume, so nothing here has side effects.
def _review(state: MappingState) -> dict:
    decision = interrupt(
        {
            "message": "These fields are close to the fill policy. Accept the expression or leave the field empty.",
            "fields": state["review"],
        }
    )
    if not isinstance(decision, dict):
        raise ValueError("resume with a dict of field -> 'accept' or 'empty'")
    fields = dict(state["fields"])
    for name, choice in decision.items():
        if name not in fields:
            continue
        rec = dict(fields[name])
        if choice == "empty":
            rec.update(
                {
                    "best": None,
                    "best_text": None,
                    "method": "review-empty",
                    "p_present": 0.0,
                }
            )
        elif choice == "accept":
            if rec.get("best") is not None:
                rec.update({"p_present": 1.0, "p_none": 0.0, "method": "review-accept"})
        else:
            raise ValueError(f"unknown review choice {choice!r} for {name}")
        fields[name] = rec
    return {
        "fields": fields,
        "review": {},
        "status": "accepted",
        "stages": [{"stage": "review", "fields": decision}],
    }


def _commit(state: MappingState, runtime: Runtime[Context]) -> dict:
    order = [o.name for o in runtime.context.pipeline.knowledge.outputs]
    header, converted = convert(
        state["fields"], state["rows"], order, runtime.context.fill
    )
    return {
        "header": header,
        "converted": converted,
        "status": "done",
        "stages": [{"stage": "commit", "fields": len(header)}],
    }


def build_graph(checkpointer=None, retry: RetryPolicy | None = None):
    retry = retry or retry_policy()
    builder = StateGraph(MappingState, context_schema=Context)
    builder.add_node("kinds", _kinds, retry_policy=retry, error_handler=_kinds_failed)
    builder.add_node("score", _score, retry_policy=retry, error_handler=_score_failed)
    builder.add_node(
        "decide", _decide, retry_policy=retry, error_handler=_decide_failed
    )
    builder.add_node("verify", _verify)
    builder.add_node("repair", _repair)
    builder.add_node("review", _review)
    builder.add_node("degrade", _degrade)
    builder.add_node("commit", _commit)
    builder.add_edge(START, "kinds")
    builder.add_edge("kinds", "score")
    builder.add_edge("score", "decide")
    builder.add_edge("decide", "verify")
    builder.add_conditional_edges(
        "verify",
        _route,
        {"repair": "repair", "review": "review", "commit": "commit"},
    )
    builder.add_edge("repair", "verify")
    builder.add_edge("review", "commit")
    builder.add_edge("degrade", "commit")
    builder.add_edge("commit", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


@dataclass
class Run:
    graph: Any
    context: Context
    thread_id: str
    result: dict

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def paused(self) -> bool:
        return bool(self.result.get("__interrupt__"))

    @property
    def done(self) -> bool:
        return self.result.get("status") == "done"

    @property
    def degraded(self) -> bool:
        return bool(self.result.get("degraded"))

    @property
    def header(self) -> list[str]:
        return self.result.get("header", [])

    @property
    def converted(self) -> list[list]:
        return self.result.get("converted", [])

    @property
    def errors(self) -> list[str]:
        return self.result.get("errors", [])

    def resume(self, decision: dict) -> dict:
        self.result = self.graph.invoke(
            Command(resume=decision), config=self.config, context=self.context
        )
        return self.result


def run_graph(
    rows: list[dict],
    decider: Decider,
    knowledge: Knowledge,
    fill: Fill | None = None,
    thread_id: str = "columnmap",
    checkpointer=None,
    retries: int = 2,
    registry=None,
    retry: RetryPolicy | None = None,
) -> Run:
    graph = build_graph(checkpointer, retry)
    wrapped = Retrying(decider, retries) if retries > 1 else decider
    context = Context(Pipeline(wrapped, knowledge, registry), fill or Fill())
    state = {
        "rows": rows,
        "matrix": {},
        "kinds": {},
        "fields": {},
        "review": {},
        "repair": {},
        "repair_attempts": 0,
        "degraded": False,
        "header": [],
        "converted": [],
        "status": "new",
        "stages": [],
        "errors": [],
    }
    result = graph.invoke(
        state, config={"configurable": {"thread_id": thread_id}}, context=context
    )
    return Run(graph, context, thread_id, result)


def resume_graph(run: Run, decision: dict) -> dict:
    return run.resume(decision)
