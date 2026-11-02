import pytest

from columnmap.core.device import resolve_device


@pytest.mark.parametrize("system, machine, expected", [
    ("Darwin", "arm64", "metal"),
    ("Darwin", "x86_64", "cpu"),
    ("Linux", "x86_64", "cpu"),
    ("Linux", "aarch64", "cpu"),
])
def test_auto_picks_metal_on_apple_silicon_and_cpu_elsewhere(system, machine, expected):
    assert resolve_device("auto", system=system, machine=machine) == expected


def test_explicit_mode_wins_over_detection():
    assert resolve_device("cpu", system="Darwin", machine="arm64") == "cpu"
    assert resolve_device("metal", system="Linux", machine="x86_64") == "metal"


def test_unknown_mode_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="device mode must be auto, metal or cpu"):
        resolve_device("cuda", system="Linux", machine="x86_64")
