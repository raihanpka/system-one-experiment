import json
import time

import pytest

from columnmap.main import main
from tests.fakes import Generic


def _runs(root):
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.rsplit("_", 1)[-1])


def _setup(tmp_path):
    root, data = tmp_path / "r", tmp_path / "d"
    base = ["--samples", "ac", "--conds", "name", "--root", str(root), "--data", str(data)]
    main([*base, "--split", "calibration"], decider_factory=Generic)
    return root, data, base, _runs(root)


def test_fit_writes_pair_and_field_entries_keyed_by_the_setup(tmp_path, capsys):
    root, data, base, runs = _setup(tmp_path)
    path = tmp_path / "calibration.json"
    main(["--fit-calibration", *map(str, runs), "--calibration", str(path), "--root", str(root), "--data", str(data)])
    keys = json.loads(path.read_text())["entries"]
    assert any(k.endswith("|options|name|v1|pair") for k in keys)
    out = capsys.readouterr().out
    assert "|field" in out                                           # the field level is fitted, or reported as skipped when one class only
    entry = next(v for k, v in keys.items() if k.endswith("|pair"))
    assert entry["seeds"] == [12, 13] and entry["n"] > 100 and entry["positives"] > 0


def test_a_run_of_the_test_split_can_never_be_used_for_fitting(tmp_path, capsys):
    root, data, base, _ = _setup(tmp_path)
    time.sleep(1.1)
    main([*base, "--split", "test", "--final"], decider_factory=Generic)
    test_runs = [p for p in _runs(root) if "-seed21_" in p.name]
    assert test_runs
    with pytest.raises(SystemExit):
        main(["--fit-calibration", str(test_runs[0]), "--calibration", str(tmp_path / "c.json"), "--root", str(root), "--data", str(data)])
    assert "the test split is never used for fitting" in capsys.readouterr().err
    assert not (tmp_path / "c.json").exists()


def test_report_shows_calibration_on_held_out_data_and_says_so(tmp_path):
    root, data, base, runs = _setup(tmp_path)
    path = tmp_path / "calibration.json"
    main(["--fit-calibration", *map(str, runs), "--calibration", str(path), "--root", str(root), "--data", str(data)])
    time.sleep(1.1)
    main([*base, "--split", "dev", "--calibration", str(path)], decider_factory=Generic)
    dev = [p for p in _runs(root) if "-seed11_" in p.name][0]
    md = (dev / "report" / "summary.md").read_text()
    assert "## Calibration" in md and "held-out" in md and "![reliability](figures/reliability.svg)" in md
    svg = (dev / "report" / "figures" / "reliability.svg").read_text()
    assert "perfect calibration" in svg and "Times New Roman" in svg
    main(["--report-only", "--out", str(runs[0]), "--calibration", str(path), "--root", str(root), "--data", str(data)])
    assert "in-sample" in (runs[0] / "report" / "summary.md").read_text()


def test_a_setup_without_calibration_is_reported_as_uncalibrated_not_an_error(tmp_path):
    root, data, base, _ = _setup(tmp_path)
    main([*base, "--split", "dev", "--calibration", str(tmp_path / "none.json")], decider_factory=Generic)
    dev = [p for p in _runs(root) if "-seed11_" in p.name][0]
    assert "No calibration fitted" in (dev / "report" / "summary.md").read_text()
