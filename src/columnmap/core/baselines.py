# the frozen baseline v1: where it lives and how its old case ids map to the current sample names

from __future__ import annotations

from .paths import ROOT
from .store import Store

V1 = ROOT / "baselines" / "v1"
# current case id -> run folder inside baselines/v1
RUNS = {"technical_terms-000": "coded_20261102-103424", "hardest_noisy-000": "mixed_20261102-104030"}
# the baseline files were written with the first sample names
OLD_IDS = {"coded-000": "technical_terms-000", "mixed-000": "hardest_noisy-000"}


# rows of one section and context from the baseline runs, with case ids renamed to the current names
def baseline_rows(section: str, cond: str) -> list[dict]:
    rows = []
    for run in RUNS.values():
        for r in Store(V1 / run / "result.json").rows(section, cond):
            rows.append({**r, "case_id": OLD_IDS.get(r["case_id"], r["case_id"])})
    return rows
