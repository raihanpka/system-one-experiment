# reference tables: code lists read from data/reference/*.csv (columns code,label); deterministic, no model involved
# a table is only offered for a column whose every non-empty value is one of its codes

from __future__ import annotations

import csv
from pathlib import Path

from .paths import DATA

REFERENCE = DATA / "reference"


def load_tables(path=REFERENCE) -> list[dict]:
    tables = []
    for f in sorted(Path(path).glob("*.csv")):
        with f.open(encoding="utf-8", newline="") as fh:
            tables.append({r["code"]: r["label"] for r in csv.DictReader(fh)})
    return tables


def covering(values, tables) -> list[dict]:
    vals = {str(v) for v in values if str(v).strip() != ""}
    return [t for t in tables if vals and vals <= set(t)]
