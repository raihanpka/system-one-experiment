from __future__ import annotations

import csv
from pathlib import Path


def write_csv(path, header, rows) -> Path:
    """UTF-8 CSV that Excel opens directly; None is written as an empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([["" if v is None else v for v in r] for r in rows])
    return path
