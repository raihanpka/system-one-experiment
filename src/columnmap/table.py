# Read a partner table into rows. CSV and TSV use the standard library; Excel needs the optional
# excel extra (uv sync --extra excel). The schema is always extracted afterwards; partners never
# supply one.

from __future__ import annotations

import csv
from pathlib import Path

EXCEL = (".xlsx", ".xlsm")


def read_table(path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
            return [dict(row) for row in reader]
    if suffix in EXCEL:
        try:
            from openpyxl import load_workbook  # optional dependency
        except ImportError as error:
            raise ValueError(
                "Excel input needs the excel extra: run `uv sync --extra excel`"
            ) from error
        book = load_workbook(path, read_only=True, data_only=True)
        sheet = book.active
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            return []
        names = [
            str(v) if v is not None else f"column_{i}" for i, v in enumerate(header)
        ]
        return [dict(zip(names, row)) for row in rows]
    raise ValueError(f"unsupported table format {suffix!r}; use .csv, .tsv, or .xlsx")
