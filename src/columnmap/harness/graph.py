# The mapping as a LangGraph: deterministic stages own the control flow, the Decisions model only answers
# atomic questions, and a field close to the fill policy pauses for a human. The library core never imports
# this module; it needs the optional graph extra: uv sync --extra graph.

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ..decider import Decider
from ..expressions import safe_evaluate
from ..knowledge import Knowledge
from ..mapper import (
    Fill,
    choose_expressions,
    convert,
    infer_kinds,
    is_filled,
    score_columns,
)
from .retry import Retrying

REVIEW_MARGIN = 0.15  # a field this close to the floor is worth a human look


class MappingState(TypedDict):
    rows: list[dict]
    matrix: dict
    kinds: dict
    fields: dict
    review: dict
    header: list[str]
    converted: list[list]
    status: str
    stages: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


@dataclass
class Context:
    decider: Decider
    knowledge: Knowledge
    fill: Fill = Fill()


def _floor(rec: dict, fill: Fill) -> float:
    return fill.copy if rec.get("kind") in ("copy", "year", "lookup") else fill.derived


def _kinds(state: MappingState, runtime: Runtime[Context]) -> dict:
    if state.get("kinds"):
        return {}
    kinds = infer_kinds(runtime.context.decider, runtime.context.knowledge)
    return {"kinds": kinds, "stages": [{"stage": "kinds", "fields": len(kinds)}]}


def _score(state: MappingState, runtime: Runtime[Context]) -> dict:
    matrix, raw = score_columns(
        runtime.context.decider, state["rows"], runtime.context.knowledge
    )
    errors = [
        f"{name}: {rec['error']}" for name, rec in raw.items() if rec.get("error")
    ]
    return {
        "matrix": matrix,
        "errors": errors,
        "stages": [{"stage": "score", "columns": len(matrix)}],
    }


def _decide(state: MappingState, runtime: Runtime[Context]) -> dict:
    fields = choose_expressions(
        runtime.context.decider,
        state["rows"],
        state["matrix"],
        runtime.context.knowledge,
        state["kinds"],
    )
    return {"fields": fields, "stages": [{"stage": "decide", "fields": len(fields)}]}


# deterministic checks: an expression that fails on the sample rows, or a field within the review
# margin of the fill policy, goes to a human. Everything else is accepted or confidently empty.
def _verify(state: MappingState, runtime: Runtime[Context]) -> dict:
    fill = runtime.context.fill
    review = {}
    for out in runtime.context.knowledge.outputs:
        rec = state["fields"][out.name]
        if rec.get("best") is None:
            continue
        present = rec.get("p_present")
        floor = _floor(rec, fill)
        detail = {
            "kind": rec.get("kind"),
            "columns": rec.get("cols", []),
            "expression": rec.get("best_text"),
            "p_present": present,
            "p_none": rec.get("p_none"),
        }
        if is_filled(rec, fill):
            if any(safe_evaluate(rec["best"], r) is None for r in state["rows"]):
                review[out.name] = {
                    **detail,
                    "reason": "expression fails on sample rows",
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
    return {"review": review, "stages": [{"stage": "verify", "review": len(review)}]}


def _gate(state: MappingState) -> dict:
    return {"status": "review" if state["review"] else "accepted"}


def _route(state: MappingState) -> str:
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
    order = [o.name for o in runtime.context.knowledge.outputs]
    header, converted = convert(
        state["fields"], state["rows"], order, runtime.context.fill
    )
    return {
        "header": header,
        "converted": converted,
        "status": "done",
        "stages": [{"stage": "commit", "fields": len(header)}],
    }


def build_graph(checkpointer=None):
    builder = StateGraph(MappingState, context_schema=Context)
    builder.add_node("kinds", _kinds)
    builder.add_node("score", _score)
    builder.add_node("decide", _decide)
    builder.add_node("verify", _verify)
    builder.add_node("gate", _gate)
    builder.add_node("review", _review)
    builder.add_node("commit", _commit)
    builder.add_edge(START, "kinds")
    builder.add_edge("kinds", "score")
    builder.add_edge("score", "decide")
    builder.add_edge("decide", "verify")
    builder.add_edge("verify", "gate")
    builder.add_conditional_edges(
        "gate", _route, {"review": "review", "commit": "commit"}
    )
    builder.add_edge("review", "commit")
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
    def header(self) -> list[str]:
        return self.result.get("header", [])

    @property
    def converted(self) -> list[list]:
        return self.result.get("converted", [])

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
) -> Run:
    graph = build_graph(checkpointer)
    wrapped = Retrying(decider, retries) if retries > 1 else decider
    context = Context(wrapped, knowledge, fill or Fill())
    state = {
        "rows": rows,
        "matrix": {},
        "kinds": {},
        "fields": {},
        "review": {},
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
