import hashlib
import json

from columnmap.core.baselines import OLD_IDS, RUNS, V1, baseline_rows
from columnmap.core.template import load_template
from columnmap.dataset.generate import generate
from columnmap.mapping.metrics import mapping_metrics

CASES = {c["case_id"]: c for c in generate(load_template(), 1, 1234)}


def test_baseline_files_exist_and_data_is_unchanged():
    meta = json.loads((V1 / "data_hash.json").read_text())["sha256"]
    for cid, run in RUNS.items():
        assert (V1 / run / "result.json").exists() and (V1 / run / "report" / "summary.md").exists()
        old = next(o for o, n in OLD_IDS.items() if n == cid)
        case = CASES[cid]
        digest = hashlib.sha256(json.dumps({"columns": case["columns"], "rows": case["rows"]}, sort_keys=True).encode()).hexdigest()
        assert digest == meta[old], f"{cid}: generated data differs from the baseline"


def test_baseline_top1_matches_the_notes():
    expected = {("hard_codes-000", "name"): 61.9, ("hard_codes-000", "samples"): 71.4, ("hardest_noisy-000", "name"): 81.2,
                ("hardest_noisy-000", "table"): 93.8}
    for (cid, cond), top1 in expected.items():
        rows = [r for r in baseline_rows("mapping", cond) if r["case_id"] == cid]
        assert round(100 * mapping_metrics(rows, CASES)["top1"], 1) == top1
