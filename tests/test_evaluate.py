# DeepEval metrics on hand-made choices with known scores, the test cases against the run record, and the --deepeval entry point
import json

from deepeval.test_case import LLMTestCase

from columnmap.calibrate import SECTIONS, load_run
from columnmap.core.scoring import T0, pair_p
from columnmap.evaluate.cases import cases_from_run, top_column
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


def test_a_tie_in_the_top_score_goes_to_the_name_that_sorts_first():
    assert top_column({"b": 0.7, "a": 0.7, "c": 0.2}) == ("a", 0.7)


def test_test_cases_match_the_run_record_without_rounding(tmp_path):
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
    store, _, cases = load_run(run)
    rows = cases_from_run(run)
    records = store.rows(SECTIONS["options"], "name")
    assert len(rows) == len(records) > 0
    for rec, row in zip(records, rows):
        case = cases[rec["case_id"]]
        actual, expected = (
            json.loads(row["actual_output"]),
            json.loads(row["expected_output"]),
        )
        for o in case["outputs"]:
            col = actual[o["name"]]["column"]
            assert col in [k["name"] for k in case["columns"]]
            assert actual[o["name"]]["p"] == pair_p(rec["matrix"][col][o["name"]], T0)
            assert expected[o["name"]] == (
                sorted(o["sources"]) if o["available"] else []
            )


def test_deepeval_flag_scores_a_run_folder_and_writes_the_report(tmp_path, capsys):
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
    assert main(["--deepeval", str(run)]) == 0
    saved = json.loads((run / "report" / "deepeval.json").read_text())
    assert set(saved["metrics"]) == {"Source accuracy", "Fill precision"}
    assert "Source accuracy: name" in capsys.readouterr().out
    assert evaluate_run(run) == saved["metrics"]
