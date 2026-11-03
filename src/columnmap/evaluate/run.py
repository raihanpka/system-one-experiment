# run the DeepEval metrics on a finished run and write report/deepeval.json; DeepEval is imported here only

from __future__ import annotations

import json
import os
from pathlib import Path


def evaluate_run(run_dir) -> dict:
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = (
        "YES"  # forced: no usage data leaves the machine
    )
    from deepeval.test_case import LLMTestCase

    from ..core.settings import load_settings
    from .cases import cases_from_run
    from .metrics import FillPrecision, SourceAccuracy

    target = load_settings()["run"]["target_precision"]
    metrics = [SourceAccuracy(threshold=0.5), FillPrecision(threshold=target)]
    rows = cases_from_run(run_dir)
    summary: dict = {}
    for m in metrics:
        per_ctx: dict[str, list[float]] = {}
        for row in rows:
            tc = LLMTestCase(
                input=row["input"],
                actual_output=row["actual_output"],
                expected_output=row["expected_output"],
            )
            per_ctx.setdefault(row["context"], []).append(m.measure(tc))
        summary[m.__name__] = {
            ctx: round(sum(v) / len(v), 4) for ctx, v in per_ctx.items()
        }
        shown = ", ".join(
            f"{ctx} {s:.3f} (n={len(per_ctx[ctx])})"
            for ctx, s in summary[m.__name__].items()
        )
        print(f"{m.__name__}: {shown}")
    out = Path(run_dir) / "report" / "deepeval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"metrics": summary, "cases": len(rows), "fill_threshold": target},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
