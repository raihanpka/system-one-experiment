# DeepEval metrics on hand-made choices with known scores, the test cases against the run record, and the --deepeval entry point
import json

from deepeval.test_case import LLMTestCase

from columnmap.bench.cli import main
from columnmap.bench.deepeval import (
    FieldsRight,
    SourceAccuracy,
    cases_from_run,
    evaluate_run,
    load_run,
)
from columnmap.bench.metrics import field_correct
from columnmap.bench.summary import top_column
from columnmap.mapper import convert
from tests.bench.fakes import Generic


def tc(actual, expected):
    return LLMTestCase(
        input="x",
        actual_output=json.dumps(actual),
        expected_output=json.dumps(expected),
    )


# f1 right column and right file, f2 wrong column, f3 has no source and was filled (wrong), f4 has no source and stays empty (right)
ACTUAL = {
    "f1": {"column": "a", "filled": True, "right": True},
    "f2": {"column": "b", "filled": True, "right": False},
    "f3": {"column": "c", "filled": True, "right": False},
    "f4": {"column": "d", "filled": False, "right": True},
}
EXPECTED = {"f1": ["a"], "f2": ["x"], "f3": [], "f4": []}


def test_source_accuracy_counts_only_the_available_fields():
    m = SourceAccuracy(threshold=0.5)
    assert (
        m.measure(tc(ACTUAL, EXPECTED)) == 0.5
    )  # f1 right, f2 wrong: 1 of 2 available
    assert m.is_successful() is True


def test_fields_right_counts_every_field_and_succeeds_only_when_all_are_right():
    m = FieldsRight()
    assert m.measure(tc(ACTUAL, EXPECTED)) == 0.5  # f1 and f4 of 4
    assert m.is_successful() is False
    assert (
        FieldsRight().measure(
            tc({"f": {"column": "a", "filled": True, "right": True}}, {"f": ["a"]})
        )
        == 1.0
    )


def test_a_tie_in_the_top_score_goes_to_the_name_that_sorts_first():
    assert top_column({"b": {"f": 0.7}, "a": {"f": 0.7}, "c": {"f": 0.2}}, "f") == "a"


def _run(tmp_path):
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
    return next(p for p in root.iterdir() if p.is_dir())


def test_test_cases_match_the_run_record(tmp_path):
    run = _run(tmp_path)
    store, _, cases = load_run(run)
    rows = cases_from_run(run, 0.9)
    records = store.rows("mapping", "name")
    honest = {r["case_id"]: r for r in store.rows("honest", "name")}
    assert len(rows) == len(records) > 0
    for rec, row in zip(records, rows):
        case = cases[rec["case_id"]]
        outs = [o["name"] for o in case["outputs"]]
        _, conv = convert(honest[rec["case_id"]]["fields"], case["rows"], outs, 0.9)
        right = field_correct(case, conv)
        actual, expected = (
            json.loads(row["actual_output"]),
            json.loads(row["expected_output"]),
        )
        for o in case["outputs"]:
            assert actual[o["name"]]["column"] == top_column(rec["matrix"], o["name"])
            assert actual[o["name"]]["right"] == right[o["name"]]
            assert expected[o["name"]] == (
                sorted(o["sources"]) if o["available"] else []
            )


def test_deepeval_flag_scores_a_run_folder_and_writes_the_report(tmp_path, capsys):
    run = _run(tmp_path)
    assert main(["--deepeval", str(run)]) == 0
    saved = json.loads((run / "report" / "deepeval.json").read_text())
    assert set(saved["metrics"]) == {"Source accuracy", "Fields right"}
    assert "Source accuracy: name" in capsys.readouterr().out
    assert evaluate_run(run) == saved["metrics"]
