# The explicit stage machine. Every caller (Mapper, the bench, the graph harness) uses these stages:
#   kinds -> score -> decide
# The stages keep no hidden state except the kind cache, which is a property of the knowledge, not a file.

from __future__ import annotations

from .decider import Decider
from .knowledge import Knowledge
from .mapper import Mapping, choose_expressions, infer_kinds, score_columns
from .schema import Schema, extract


class Pipeline:
    def __init__(self, decider: Decider, knowledge: Knowledge, registry=None):
        self.decider = decider
        self.knowledge = knowledge
        self.registry = knowledge.registry(registry)
        self._kinds: dict | None = None

    def kinds(self, refresh: bool = False) -> dict:
        if self._kinds is None or refresh:
            self._kinds = infer_kinds(
                self.decider, self.knowledge, registry=self.registry
            )
        return self._kinds

    def use_kinds(self, kinds: dict) -> dict:
        self._kinds = dict(kinds)
        return self._kinds

    def score(
        self, rows: list[dict], schema: Schema | None = None
    ) -> tuple[dict, dict]:
        return score_columns(
            self.decider,
            rows,
            self.knowledge,
            schema or extract(rows),
            self.registry,
        )

    def decide(
        self,
        rows: list[dict],
        matrix: dict,
        kinds: dict | None = None,
        schema: Schema | None = None,
    ) -> dict:
        return choose_expressions(
            self.decider,
            rows,
            matrix,
            self.knowledge,
            kinds or self.kinds(),
            schema or extract(rows),
            self.registry,
        )

    def run(self, rows: list[dict]) -> Mapping:
        schema = extract(rows)
        matrix, _ = self.score(rows, schema)
        fields = self.decide(rows, matrix, self.kinds(), schema)
        return Mapping(matrix, fields, [o.name for o in self.knowledge.outputs])
