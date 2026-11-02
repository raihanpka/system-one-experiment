import json
import re
import time

import pytest

from columnmap.decisions.types import Answer, Response
from columnmap.main import main


# No model: first option for choices, p=0.3 for yes/no
class Generic:

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def decide(self, text, questions):
        out = []
        for q in questions:
            if q.type == "predicate":
                out.append(Answer("predicate", q.name, probability=0.3, raw_scores={"false": 0.0, "true": -0.8}))
            else:
                names = [c.value for c in q.choices]
                raw = {n: (1.0 if i == 0 else -1.0) for i, n in enumerate(names)}
                out.append(Answer("choice", q.name, choice=names[0], confidence=0.5,
                                  probabilities={n: 1 / len(names) for n in names}, raw_scores=raw))
        return Response("fake", out, {})


def test_run_folder_layout_files_and_resume(tmp_path, capsys):
    root, data = tmp_path / "results", tmp_path / "data"
    args = ["--root", str(root), "--data", str(data)]
    assert main(["--samples", "abcd", *args], decider_factory=Generic) == 0
    runs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]
    assert len(runs) == 1 and re.fullmatch(r"identical-synonym-coded-mixed_\d{8}-\d{6}", runs[0].name)   # <samples>_<timestamp>
    run = runs[0]
    assert not (root / "latest").exists()                                                          # no extra link next to the runs

    for f in ("input_coded.csv", "expected_coded.csv", "ground_truth_coded.csv"):              # single source of truth
        assert (data / "samples" / "c_coded" / f).exists(), f
    assert (data / "template_header.csv").exists()
    assert not list(run.rglob("input_*.csv")) and not list(run.rglob("expected_*.csv"))              # not copied into the run

    data_json = json.loads((run / "result.json").read_text())                                    # this run's raw output
    assert set(data_json) == {"meta", "mapping", "mapping_pairs", "transform", "permute"} and data_json["meta"]["mode"] == "options"
    assert data_json["meta"]["samples"] == "abcd"
    assert [r["case_id"] for r in data_json["mapping"]["name"]] == ["identical-000", "synonym-000", "coded-000", "mixed-000"]
    assert set(data_json["mapping"]) == {"name", "samples"}

    for letter, word in (("a", "identical"), ("b", "synonym"), ("c", "coded"), ("d", "mixed")):
        d = run / f"{letter}_{word}"
        for f in (f"converted_{word}_name.csv", f"converted_{word}_samples.csv", f"decisions_{word}_name.csv",
                  f"probabilities_{word}_samples.csv"):
            assert (d / f).exists(), f"{d.name}/{f}"
    converted = (run / "a_identical" / "converted_identical_name.csv").read_text().splitlines()
    assert converted[0].startswith("policy_number,insured_full_name") and len(converted) > 10
    assert len((run / "c_coded" / "decisions_coded_name.csv").read_text().splitlines()) == 32   # header + 31 fields

    rep = run / "report"
    for f in ("summary.md", "figures/accuracy.svg", "figures/outcomes.svg", "figures/drift.svg", "figures/calibration.svg",
              "figures/kind.svg", "figures/transform.svg", "figures/heatmap_a_identical_name.svg", "figures/heatmap_d_mixed_samples.svg"):
        assert (rep / f).exists(), f
    svg = (rep / "figures/accuracy.svg").read_text()
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg and "Times New Roman" in svg
    md = (rep / "summary.md").read_text()
    assert "![accuracy](figures/accuracy.svg)" in md and "data/samples/c_coded/" in md
    assert "## Mapping" in md and "## Converted File Against the Expected File" in md
    assert "## At a Glance" in md and "## Rejecting Fields Without a Source" in md

    capsys.readouterr()
    main(["--out", str(run), "--samples", "abcd", *args], decider_factory=Generic)              # same folder again: nothing recomputed
    assert "0 new, 4 resumed" in capsys.readouterr().out
    main(["--root", str(root), "--data", str(data), "--report-only"])                            # newest run, no model, samples from meta
    assert "d_mixed" in (rep / "summary.md").read_text()


def test_each_run_gets_its_own_folder_and_report_only_takes_the_newest(tmp_path):
    root, data = tmp_path / "r", tmp_path / "d"
    main(["--samples", "a", "--root", str(root), "--data", str(data), "--conds", "name"], decider_factory=Generic)
    time.sleep(1.1)                                                                              # names carry seconds
    main(["--samples", "b", "--root", str(root), "--data", str(data), "--conds", "name"], decider_factory=Generic)
    runs = sorted((p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda n: n.rsplit("_", 1)[-1])
    assert len(runs) == 2 and runs[0].startswith("identical_") and runs[1].startswith("synonym_")
    assert all((root / r / "result.json").exists() for r in runs)                                # one result.json per run
    main(["--root", str(root), "--data", str(data), "--report-only"])                            # rebuilds the newest run only
    assert (root / runs[1] / "report" / "summary.md").stat().st_mtime > (root / runs[0] / "report" / "summary.md").stat().st_mtime


def test_bad_sample_letter_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["--samples", "az", "--root", str(tmp_path), "--data", str(tmp_path)], decider_factory=Generic)


def test_report_only_without_any_run_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path / "empty"), "--data", str(tmp_path), "--report-only"])


def test_two_modes_whole_table_and_cutoffs_fitted_on_another_run(tmp_path):
    root, data = tmp_path / "r", tmp_path / "d"
    args = ["--root", str(root), "--data", str(data), "--conds", "name,table", "--mode", "both"]
    main(["--samples", "c", *args], decider_factory=Generic)
    time.sleep(1.1)
    first = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()][0]
    main(["--samples", "d", "--calibrate-from", str(first), *args], decider_factory=Generic)
    second = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name.rsplit("_", 1)[-1])[-1]
    assert second != first and second.name.startswith("mixed_")

    store = json.loads((second / "result.json").read_text())
    assert set(store["mapping"]) == {"name", "table"} and set(store["mapping_pairs"]) == {"name", "table"}
    assert set(store["transform"]) == {"name"}                                           # the whole-table context reuses the others
    assert store["meta"]["mode"] == "options,pairs"

    md = (second / "report" / "summary.md").read_text()
    assert "## Options Mode Versus Pairs Mode" in md and "fitted on other run" in md
    assert "name + whole table, options mode" in md                                      # readable context labels in the headline
    assert (second / "d_mixed" / "converted_mixed_name_calibrated.csv").exists()
    main(["--report-only", "--out", str(second), "--data", str(data)])                          # contexts are read from the run itself
    assert "name + whole table, options mode" in (second / "report" / "summary.md").read_text()
    first_md = (first / "report" / "summary.md").read_text()
    assert "fitted on this run, optimistic" in first_md and "fitted on other run" not in first_md
