import hashlib
import json

from columnmap.core.paths import ROOT
from columnmap.core.store import Store
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.mapping.metrics import mapping_metrics

V1 = ROOT / "baselines" / "v1"
RUNS = {"coded-000": "coded_20261102-103424", "mixed-000": "mixed_20261102-104030"}
CASES = {c["case_id"]: c for c in generate(load_template(), 1, 1234)}


def test_baseline_files_exist_and_data_is_unchanged():
    meta = json.loads((V1 / "data_hash.json").read_text())
    for cid, run in RUNS.items():
        assert (V1 / run / "result.json").exists() and (V1 / run / "report" / "summary.md").exists()
        case = CASES[cid]
        digest = hashlib.sha256(json.dumps({"columns": case["columns"], "rows": case["rows"]}, sort_keys=True).encode()).hexdigest()
        assert digest == meta["sha256"][cid], f"{cid}: generated data differs from the baseline"


def test_baseline_top1_matches_the_notes():
    expected = {("coded-000", "name"): 61.9, ("coded-000", "samples"): 71.4, ("mixed-000", "name"): 81.2, ("mixed-000", "table"): 93.8}
    for (cid, cond), top1 in expected.items():
        rows = Store(V1 / RUNS[cid] / "result.json").rows("mapping", cond)
        assert round(100 * mapping_metrics(rows, CASES)["top1"], 1) == top1
