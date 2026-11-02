import json

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


def test_one_command_writes_result_json_figures_tables_and_resumes(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["--samples", "abcd", "--out", str(out)], decider_factory=Generic) == 0
    data = json.loads((out / "result.json").read_text())                      # one plain JSON file
    assert set(data) == {"meta", "mapping", "transform", "permute"} and data["meta"]["mode"] == "options"
    assert [r["case_id"] for r in data["mapping"]["name"]] == ["identical-000", "synonym-000", "legacy-000", "mixed-000"]
    assert set(data["mapping"]) == {"name", "samples"}

    rep = out / "report"
    for f in ("summary.md", "figures/accuracy.svg", "figures/outcomes.svg", "figures/drift.svg", "figures/calibration.svg",
              "figures/transform.svg", "figures/heatmap_a_name.svg", "figures/heatmap_d_samples.svg",
              "predictions_c_name.csv"):
        assert (rep / f).exists(), f
    svg = (rep / "figures/accuracy.svg").read_text()
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg and "Times New Roman" in svg
    md = (rep / "summary.md").read_text()
    assert "![accuracy](figures/accuracy.svg)" in md and "| sample | case | input naming |" in md and "## Mapping" in md
    assert (out / "samples" / "legacy-000" / "input.csv").exists() and (out / "samples" / "template_header.csv").exists()
    assert len((rep / "predictions_a_name.csv").read_text().splitlines()) == 32      # header + 31 output fields

    capsys.readouterr()
    main(["--samples", "abcd", "--out", str(out)], decider_factory=Generic)           # second process: nothing recomputed
    assert "0 new, 4 resumed" in capsys.readouterr().out
    main(["--samples", "ab", "--out", str(out), "--report-only"])                     # no model needed
    assert "heatmap_c" not in (rep / "summary.md").read_text()


def test_bad_sample_letter_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["--samples", "az", "--out", str(tmp_path)], decider_factory=Generic)
