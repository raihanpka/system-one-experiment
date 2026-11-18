# The input schema we extract ourselves: type, unit, samples, null rate, and distinct count per column.
# Input comes from a partner, so there is no partner schema to read; these facts are all derived from the
# column name and values. The model sees this compact schema instead of raw rows.

from __future__ import annotations

from dataclasses import dataclass

from .expressions import column_type, column_unit


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # string | number | date
    unit: str  # amount | rate | count | id | text | date | number
    samples: tuple[str, ...]
    null_rate: float
    distinct: int


@dataclass(frozen=True)
class Schema:
    columns: tuple[Column, ...]
    rows: int

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def units(self) -> dict:
        return {c.name: c.unit for c in self.columns}

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)


def _samples(values, k: int = 5) -> tuple[str, ...]:
    n = len(values)
    if n == 0:
        return ()
    idx = [i * (n - 1) // (k - 1) for i in range(k)] if n >= k else list(range(n))
    return tuple(str(values[i]) for i in idx)


def extract(rows: list[dict], sample_k: int = 5) -> Schema:
    if not rows:
        return Schema((), 0)
    columns = []
    for name in rows[0]:
        values = [r.get(name) for r in rows]
        present = [v for v in values if v is not None and str(v).strip() != ""]
        columns.append(
            Column(
                name=name,
                type=column_type(values),
                unit=column_unit(name, values),
                samples=_samples(values, sample_k),
                null_rate=round(1 - len(present) / len(values), 4),
                distinct=len({str(v) for v in present}),
            )
        )
    return Schema(tuple(columns), len(rows))
