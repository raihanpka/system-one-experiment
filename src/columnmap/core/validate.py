# deterministic checks before a value is written to a filled field: no ground truth, only the template type
# date: ISO yyyy-mm-dd, number: finite float, string: not blank. An empty cell is never valid.

from __future__ import annotations

import math
from datetime import date


def valid_value(value, typ: str) -> bool:
    if value is None or str(value).strip() == "":
        return False
    if typ == "date":
        try:
            date.fromisoformat(str(value))
            return True
        except ValueError:
            return False
    if typ == "number":
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    return typ == "string"


# a field is valid when every converted cell of it passes the type check
def field_valid(values, typ: str) -> bool:
    return all(valid_value(v, typ) for v in values)
