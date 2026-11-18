# thresholds from a labeled run: point estimate meets the budget, the upper bound is reported honestly

import json

from column_mapping.bench.calibrate import calibrate_run, wilson_upper
from column_mapping.bench.cli import main
from column_mapping.policy import Policy
from tests.bench.fakes import Approves


def test_wilson_upper_is_honest_about_a_small_sample():
    assert wilson_upper(0, 0) == 0.0
    assert wilson_upper(0, 50) < 0.08
    assert (
        wilson_upper(1, 50) > 0.02
    )  # the point estimate meets 2%, the upper bound does not


def test_calibrate_writes_a_policy_next_to_the_dataset(tmp_path, capsys):
    root, data = tmp_path / "results", tmp_path / "data"
    main(
        ["--samples", "a", "--root", str(root), "--data", str(data)],
        decider_factory=Approves,
    )
    capsys.readouterr()
    run = next(p for p in root.iterdir() if p.is_dir())
    knowledge = tmp_path / "knowledge"
    assert (
        main(
            [
                "--calibrate",
                str(run),
                "--knowledge",
                str(knowledge),
                "--budget",
                "0.02",
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["policy"]["calibration"]["budget_met_point"] in (True, False)
    policy = Policy.load(knowledge)
    assert policy is not None
    assert 0 <= policy.derived <= 1 and 0 <= policy.copy <= 1
    assert policy.model
    assert policy.calibration["seed"] is not None


def test_calibrate_run_returns_the_same_policy_it_writes(tmp_path, capsys):
    root, data = tmp_path / "results", tmp_path / "data"
    main(
        ["--samples", "a", "--root", str(root), "--data", str(data)],
        decider_factory=Approves,
    )
    run = next(p for p in root.iterdir() if p.is_dir())
    knowledge = tmp_path / "knowledge"
    summary = calibrate_run(run, knowledge, 0.02, as_json=True)
    capsys.readouterr()
    assert json.loads((knowledge / "policy.json").read_text()) == summary["policy"]
