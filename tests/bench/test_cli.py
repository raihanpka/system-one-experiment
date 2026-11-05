import csv
import json
import re

import pytest

from columnmap.bench.cli import main
from tests.bench.fakes import Approves, Generic


class Counting(Generic):
    calls = 0

    def decide(self, text, questions):
        type(self).calls += 1
        return super().decide(text, questions)


class Forbidden(Generic):
    def decide(self, text, questions):
        raise AssertionError("the model must not be called")


def paths(tmp_path):
    return ["--root", str(tmp_path / "results"), "--data", str(tmp_path / "data")]


def runs(tmp_path):
    return sorted(p for p in (tmp_path / "results").iterdir() if p.is_dir())


def test_run_folder_layout_files_and_summary(tmp_path):
    assert main(["--samples", "abcd", *paths(tmp_path)], decider_factory=Generic) == 0
    (run,) = runs(tmp_path)
    assert re.fullmatch(r"all_\d{8}-\d{6}", run.name)  # <samples>_<timestamp>
    for f in (
        "input_technical_terms.csv",
        "expected_technical_terms.csv",
        "ground_truth_technical_terms.csv",
    ):
        assert (tmp_path / "data" / "samples" / "c_technical_terms" / f).exists(), f
    assert (
        tmp_path / "data" / "knowledge" / "outputs.json"
    ).exists()  # the format another project fills in
    raw = json.loads((run / "result.json").read_text())
    assert set(raw) == {"meta", "mapping", "honest"}
    assert [r["case_id"] for r in raw["mapping"]][0] == "easy_identical-000"
    for letter, word in (
        ("a", "easy_identical"),
        ("b", "medium_synonym"),
        ("c", "technical_terms"),
        ("d", "hardest_noisy"),
    ):
        for f in (
            f"converted_{word}.csv",
            f"decisions_{word}.csv",
            f"scores_{word}.csv",
        ):
            assert (run / f"{letter}_{word}" / f).exists(), f
    with (run / "c_technical_terms" / "decisions_technical_terms.csv").open() as fh:
        assert len(list(csv.reader(fh))) == 32  # header + 31 fields
    md = (run / "report" / "summary.md").read_text()
    assert (
        re.search(r"\| a \| \d+ / \d+ \| \d+ / \d+ \|", md)
        and "**total**" in md
        and "Fields still wrong" in md
    )


def test_a_finished_run_resumes_without_calling_the_model_again(tmp_path):
    main(
        ["--samples", "ac", *paths(tmp_path)],
        decider_factory=Counting,
    )
    (run,) = runs(tmp_path)
    Counting.calls = 0
    main(
        ["--samples", "ac", "--out", str(run), *paths(tmp_path)],
        decider_factory=Counting,
    )
    assert Counting.calls == 0


def test_report_only_rebuilds_the_newest_run_with_another_fill_score_and_no_model_call(
    tmp_path,
):
    main(
        ["--samples", "a", *paths(tmp_path)],
        decider_factory=Approves,
    )
    (run,) = runs(tmp_path)
    dec = run / "a_easy_identical" / "decisions_easy_identical.csv"

    def filled():
        with dec.open() as fh:
            return sum(r["filled"] == "True" for r in csv.DictReader(fh))

    assert (
        filled() > 0
    )  # the first column set of every field gets 0.95, above the default fill score
    main(
        ["--report-only", "--fill-score", "0.99", *paths(tmp_path)],
        decider_factory=Forbidden,
    )
    assert filled() == 0


def test_each_run_gets_its_own_folder(tmp_path):
    import time

    main(["--samples", "a", *paths(tmp_path)], decider_factory=Generic)
    time.sleep(1.1)
    main(["--samples", "a", *paths(tmp_path)], decider_factory=Generic)
    assert len(runs(tmp_path)) == 2


def test_seed_option_builds_other_data_and_is_recorded(tmp_path):
    main(
        ["--samples", "a", "--seed", "5", *paths(tmp_path)],
        decider_factory=Generic,
    )
    (run,) = runs(tmp_path)
    assert run.name.startswith("easy_identical-seed5_")
    assert json.loads((run / "result.json").read_text())["meta"]["seed"] == 5
    assert not (
        tmp_path / "data" / "samples"
    ).exists()  # data/ holds the baseline seed only


def test_splits_run_one_folder_per_seed_and_the_test_split_needs_final(tmp_path):
    main(
        ["--samples", "a", "--split", "holdout", *paths(tmp_path)],
        decider_factory=Generic,
    )
    assert sorted(r.name.rsplit("_", 1)[0] for r in runs(tmp_path)) == [
        "easy_identical-seed12",
        "easy_identical-seed13",
    ]
    with pytest.raises(SystemExit):
        main(
            ["--samples", "a", "--split", "test", *paths(tmp_path)],
            decider_factory=Generic,
        )
