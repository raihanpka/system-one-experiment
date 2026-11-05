# DeepEval metrics on a finished run, against the ground truth; no judge model and no model call.
# Imported only by `--deepeval`, so the rest of the package does not need DeepEval.

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = (
    "YES"  # forced: no usage data leaves the machine
)

from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from ..config import load_settings  # noqa: E402
from ..mapper import convert, is_filled  # noqa: E402
from .metrics import field_correct  # noqa: E402
from .store import Store  # noqa: E402
from .summary import top_column  # noqa: E402
from .generate import generate  # noqa: E402
from .template import load_template  # noqa: E402


def load_run(run_dir) -> tuple[Store, dict, dict]:
    store = Store(Path(run_dir) / "result.json")
    meta = store.data["meta"]
    cases = {c["case_id"]: c for c in generate(load_template(), 1, meta["seed"])}
    return store, meta, cases


def cases_from_run(run_dir, fill_score: float) -> list[dict]:
    store, _, cases = load_run(run_dir)
    honest = {r["case_id"]: r for r in store.rows("honest")}
    out = []
    for row in store.rows("mapping"):
        rec = honest.get(row["case_id"])
        if rec is None:
            continue
        case = cases[row["case_id"]]
        names = [o["name"] for o in case["outputs"]]
        _, conv = convert(rec["fields"], case["rows"], names, fill_score)
        right = field_correct(case, conv)
        actual, expected = {}, {}
        for o in case["outputs"]:
            f = o["name"]
            actual[f] = {
                "column": top_column(row["matrix"], f),
                "filled": is_filled(rec["fields"][f], fill_score),
                "right": right[f],
            }
            expected[f] = sorted(o["sources"]) if o["available"] else []
        out.append(
            {
                "name": row["case_id"],
                "input": f"{row['case_id']} columns={len(case['columns'])}",
                "actual_output": json.dumps(actual, sort_keys=True),
                "expected_output": json.dumps(expected, sort_keys=True),
            }
        )
    return out


def _read(test_case) -> tuple[dict, dict]:
    return json.loads(test_case.actual_output), json.loads(test_case.expected_output)


class SourceAccuracy(BaseMetric):
    # share of the available output fields whose top column is a true source (the column choice only)
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs) -> float:
        actual, expected = _read(test_case)
        avail = [f for f, srcs in expected.items() if srcs]
        hit = [f for f in avail if actual[f]["column"] in expected[f]]
        self.score = len(hit) / len(avail) if avail else 1.0
        self.reason = (
            f"{len(hit)} / {len(avail)} available fields have the right top column"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Source accuracy"


class FieldsRight(BaseMetric):
    # share of all output fields whose converted column equals the expected column (a field with no source is right when empty)
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs) -> float:
        actual, _ = _read(test_case)
        right = [f for f in actual if actual[f]["right"]]
        self.score = len(right) / len(actual) if actual else 1.0
        self.reason = f"{len(right)} / {len(actual)} fields equal the expected file"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Fields right"


def evaluate_run(run_dir, fill_score: float | None = None) -> dict:
    fill_score = (
        fill_score if fill_score is not None else load_settings()["run"]["fill_score"]
    )
    metrics = [SourceAccuracy(), FieldsRight()]
    rows = cases_from_run(run_dir, fill_score)
    summary: dict = {}
    for m in metrics:
        scores = []
        for row in rows:
            tc = LLMTestCase(
                input=row["input"],
                actual_output=row["actual_output"],
                expected_output=row["expected_output"],
            )
            scores.append(m.measure(tc))
        summary[m.__name__] = round(sum(scores) / len(scores), 4) if scores else None
        print(f"{m.__name__}: {summary[m.__name__]} (mean over {len(scores)} samples)")
    out = Path(run_dir) / "report" / "deepeval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"metrics": summary, "cases": len(rows), "fill_score": fill_score},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
