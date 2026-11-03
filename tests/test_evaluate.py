# DeepEval metrics on hand-made choices with known scores, and an end-to-end run folder through the entry point
import json

from deepeval.test_case import LLMTestCase

from columnmap.evaluate.cases import cases_from_run
from columnmap.evaluate.metrics import FillPrecision, SourceAccuracy
from columnmap.evaluate.run import evaluate_run
from columnmap.main import main
from tests.fakes import Generic


def tc(actual, expected):
    return LLMTestCase(
        input="x",
        actual_output=json.dumps(actual),
        expected_output=json.dumps(expected),
    )


# f1 right, f2 wrong, f3 has no source and is filled with p 0.7 (a wrong fill), f4 has no source and is not filled
ACTUAL = {
    "f1": {"column": "a", "p": 0.95},
    "f2": {"column": "b", "p": 0.80},
    "f3": {"column": "c", "p": 0.70},
    "f4": {"column": "d", "p": 0.20},
}
EXPECTED = {"f1": ["a"], "f2": ["x"], "f3": [], "f4": []}


def test_source_accuracy_counts_only_the_available_fields():
    m = SourceAccuracy(threshold=0.5)
    assert (
        m.measure(tc(ACTUAL, EXPECTED)) == 0.5
    )  # f1 right, f2 wrong: 1 of 2 available
    assert m.is_successful() is True


def test_fill_precision_counts_a_filled_field_with_no_source_as_wrong():
    m = FillPrecision(threshold=0.9)
    assert round(m.measure(tc(ACTUAL, EXPECTED)), 4) == round(
        1 / 3, 4
    )  # fills: f1 right, f2 wrong, f3 wrong
    assert m.is_successful() is False


def test_end_to_end_run_folder_gets_metric_scores_and_a_report_file(tmp_path, capsys):
    root, data = tmp_path / "r", tmp_path / "d"
    main(
        [
            "--samples",
            "ac",
            "--conds",
            "name",
            "--root",
            str(root),
            "--data",
            str(data),
            "--split",
            "dev",
        ],
        decider_factory=Generic,
    )
    run = next(p for p in root.iterdir() if p.is_dir())
    rows = cases_from_run(run)
    assert rows and {r["context"] for r in rows} == {"name"}
    summary = evaluate_run(run)
    assert set(summary) == {"Source accuracy", "Fill precision"}
    saved = json.loads((run / "report" / "deepeval.json").read_text())
    assert saved["metrics"] == summary and saved["cases"] == len(rows)
    assert "Source accuracy: name" in capsys.readouterr().out
