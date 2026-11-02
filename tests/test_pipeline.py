import json
import re
import time

import pytest

from src.decisions.types import Answer, Response
from src.pipeline import main


class Generic:
    """No model: first option for choices, p=0.3 for yes/no."""

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
    root = tmp_path / "results"
    assert main(["--samples", "abcd", "--root", str(root)], decider_factory=Generic) == 0
    runs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]
    assert len(runs) == 1 and re.fullmatch(r"\d{8}-\d{6}_abcd", runs[0].name)         # sorts by time, names the samples
    run = runs[0]
    assert (root / "latest").is_symlink() and (root / "latest").resolve() == run.resolve()

    data = json.loads((run / "result.json").read_text())                                 # one plain JSON file
    assert set(data) == {"meta", "mapping", "transform", "permute"} and data["meta"]["mode"] == "options"
    assert data["meta"]["samples"] == "abcd"
    assert [r["case_id"] for r in data["mapping"]["name"]] == ["identical-000", "synonym-000", "legacy-000", "mixed-000"]
    assert set(data["mapping"]) == {"name", "samples"}

    assert (run / "template_header.csv").exists()
    for letter, word in (("a", "identical"), ("b", "synonym"), ("c", "legacy"), ("d", "mixed")):
        d = run / f"{letter}_{word}"
        for f in (f"input_{word}.csv", f"expected_{word}.csv", f"ground_truth_{word}.csv",
                  f"converted_{word}_name.csv", f"converted_{word}_samples.csv", f"decisions_{word}_name.csv",
                  f"probabilities_{word}_samples.csv"):
            assert (d / f).exists(), f"{d.name}/{f}"
    converted = (run / "a_identical" / "converted_identical_name.csv").read_text().splitlines()
    assert converted[0].startswith("policy_number,insured_full_name") and len(converted) > 10
    assert len((run / "c_legacy" / "decisions_legacy_name.csv").read_text().splitlines()) == 32   # header + 31 fields

    rep = run / "report"
    for f in ("summary.md", "figures/accuracy.svg", "figures/outcomes.svg", "figures/drift.svg", "figures/calibration.svg",
              "figures/kind.svg", "figures/transform.svg", "figures/heatmap_a_identical_name.svg", "figures/heatmap_d_mixed_samples.svg"):
        assert (rep / f).exists(), f
    svg = (rep / "figures/accuracy.svg").read_text()
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg and "Times New Roman" in svg
    md = (rep / "summary.md").read_text()
    assert "![accuracy](figures/accuracy.svg)" in md and "| sample | folder | input naming |" in md
    assert "## Mapping" in md and "## Converted file against the expected file" in md and "c_legacy/" in md

    capsys.readouterr()
    main(["--out", str(run), "--samples", "abcd"], decider_factory=Generic)             # same folder again: nothing recomputed
    assert "0 new, 4 resumed" in capsys.readouterr().out
    main(["--root", str(root), "--report-only"])                                         # newest run, no model, samples from meta
    assert "d_mixed" in (rep / "summary.md").read_text()


def test_new_run_does_not_overwrite_and_latest_follows(tmp_path):
    root = tmp_path / "r"
    main(["--samples", "a", "--root", str(root), "--conds", "name"], decider_factory=Generic)
    time.sleep(1.1)                                                                      # folder names carry seconds
    main(["--samples", "b", "--root", str(root), "--conds", "name"], decider_factory=Generic)
    runs = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    assert len(runs) == 2 and runs[0].endswith("_a") and runs[1].endswith("_b")
    assert (root / "latest").resolve().name == runs[1]


def test_bad_sample_letter_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["--samples", "az", "--root", str(tmp_path)], decider_factory=Generic)


def test_report_only_without_any_run_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path / "empty"), "--report-only"])
