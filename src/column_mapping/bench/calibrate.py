# Choose the fill floors from a labeled run. The thresholds are a policy, not a per-batch knob:
#   budget      the largest false-fill rate the business accepts (no-source fields that get filled)
#   point       the observed rate must meet the budget
#   upper       a Wilson 95% upper bound; on a small calibration set this can be above the budget,
#               which is the honest signal to collect more labels, not to lower the threshold
# The chosen policy is written next to outputs.json and prompts.json.

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..mapper import Fill, convert, is_filled
from ..policy import Policy
from .cases import records_from_run
from .metrics import field_correct


def wilson_upper(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return min(1.0, centre + margin)


def metrics_at(records: list[dict], fill: Fill) -> dict:
    right = filled = tp = fp = fn = tn = wrong = fields_total = 0
    for item in records:
        case, fields = item["case"], item["fields"]
        names = [o["name"] for o in case["outputs"]]
        fields_total += len(names)
        _, conv = convert(fields, case["rows"], names, fill)
        ok = field_correct(case, conv)
        for o in case["outputs"]:
            f = o["name"]
            is_filled_here = is_filled(fields[f], fill)
            if is_filled_here:
                filled += 1
            if o["available"]:
                if is_filled_here and ok[f]:
                    tp += 1
                elif is_filled_here:
                    wrong += 1
                else:
                    fn += 1
            elif is_filled_here:
                fp += 1
            else:
                tn += 1
            if ok[f]:
                right += 1
    no_source = tn + fp
    source = tp + fn + wrong
    return {
        "right": right,
        "fields": fields_total,
        "filled": filled,
        "filled_right": tp,
        "wrong": wrong,
        "false_fill": fp,
        "false_empty": fn,
        "no_source": no_source,
        "source": source,
        "false_fill_rate": fp / no_source if no_source else 0.0,
        "upper_false_fill": wilson_upper(fp, no_source),
        "precision": tp / filled if filled else 1.0,
        "recall": tp / source if source else 1.0,
    }


def _grid(start: float = 0.3, stop: float = 0.95, step: float = 0.05) -> list[float]:
    values = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 2))
        value += step
    return values


def calibrate_run(
    run_dir, knowledge_dir, budget: float = 0.02, as_json: bool = False
) -> dict:
    records, meta = records_from_run(run_dir)
    if not records:
        raise ValueError(f"no mapping records in {run_dir}")
    scored = [
        (derived, copy, metrics_at(records, Fill(derived, copy, 0.5)))
        for derived in _grid()
        for copy in _grid()
    ]
    within = [item for item in scored if item[2]["false_fill_rate"] <= budget]
    default = Policy()
    if within:
        # Keep the best result; when a whole plateau ties, stay closest to the current policy
        # instead of drifting to the most permissive edge of the plateau.
        top_right = max(item[2]["right"] for item in within)
        top = [item for item in within if item[2]["right"] == top_right]
        top_precision = max(item[2]["precision"] for item in top)
        top = [item for item in top if item[2]["precision"] == top_precision]
        top_false = min(item[2]["false_fill"] for item in top)
        top = [item for item in top if item[2]["false_fill"] == top_false]
        chosen = min(
            top,
            key=lambda item: (
                abs(item[0] - default.derived) + abs(item[1] - default.copy)
            ),
        )
    else:  # the budget is not reachable even at the highest floors; take the cleanest point
        chosen = min(
            scored, key=lambda item: (item[2]["false_fill_rate"], -item[2]["right"])
        )
    derived, copy, metrics = chosen
    policy = Policy(
        derived=derived,
        copy=copy,
        none_ceiling=0.5,
        model=meta.get("model"),
        calibration={
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": Path(run_dir).name,
            "seed": meta.get("seed"),
            "budget": budget,
            "false_fill": metrics["false_fill"],
            "no_source": metrics["no_source"],
            "false_fill_rate": round(metrics["false_fill_rate"], 4),
            "upper_false_fill": round(metrics["upper_false_fill"], 4),
            "budget_met_point": metrics["false_fill_rate"] <= budget,
            "budget_met_upper": metrics["upper_false_fill"] <= budget,
            "right": metrics["right"],
            "fields": metrics["fields"],
            "precision": round(metrics["precision"], 4),
        },
    )
    Path(knowledge_dir).mkdir(parents=True, exist_ok=True)
    path = policy.save(knowledge_dir)
    summary = {
        "policy": policy.as_dict(),
        "path": str(path),
        "metrics": {k: v for k, v in metrics.items()},
    }
    if as_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(
            f"policy: derived >= {derived}, copy >= {copy}, none <= 0.5 -> {path}\n"
            f"right {metrics['right']} fields, false fill {metrics['false_fill']}"
            f" / {metrics['no_source']} ({metrics['false_fill_rate']:.1%}),"
            f" 95% upper {metrics['upper_false_fill']:.1%},"
            f" precision {metrics['precision']:.1%}"
        )
        if not policy.calibration["budget_met_upper"]:
            print(
                "warning: the 95% upper bound is above the budget; "
                "the point estimate meets it, but more labeled fields are needed to certify it"
            )
    return summary
