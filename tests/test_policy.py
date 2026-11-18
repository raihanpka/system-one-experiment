# one dataset, one policy: the floors, the model they were calibrated for, and the fill they produce

import json

import pytest

from column_mapping.knowledge import Knowledge
from column_mapping.mapper import Fill
from column_mapping.policy import Policy


def test_policy_defaults_and_fill():
    policy = Policy()
    assert (policy.derived, policy.copy, policy.none_ceiling) == (0.75, 0.4, 0.5)
    assert policy.fill() == Fill(0.75, 0.4, 0.5)
    assert policy.model is None and policy.calibration == {}


def test_policy_round_trips_next_to_the_dataset(tmp_path):
    policy = Policy(
        derived=0.8,
        copy=0.45,
        none_ceiling=0.4,
        model="microsoft/microsoft-decision-1.1",
        calibration={"seed": 11, "budget": 0.02},
    )
    path = policy.save(tmp_path)
    assert path.name == "policy.json"
    assert Policy.load(tmp_path) == policy
    raw = json.loads(path.read_text())
    assert raw["version"] == 1
    assert raw["fill"] == {"derived": 0.8, "copy": 0.45, "none_ceiling": 0.4}
    assert Policy.load(tmp_path / "missing") is None


def test_a_policy_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        Policy(derived=1.2)
    with pytest.raises(ValueError):
        Policy(copy=-0.1)


def test_knowledge_loads_and_saves_the_policy(tmp_path):
    (tmp_path / "outputs.json").write_text(
        json.dumps([{"name": "x", "type": "string", "description": "X"}]),
        encoding="utf-8",
    )
    Policy(copy=0.55, calibration={"seed": 11}).save(tmp_path)
    knowledge = Knowledge.load(tmp_path)
    assert knowledge.policy.copy == 0.55
    out = tmp_path / "saved"
    knowledge.save(out)
    assert (out / "policy.json").exists()
    assert Knowledge.load(out) == knowledge
