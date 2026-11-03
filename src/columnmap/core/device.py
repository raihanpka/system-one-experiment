# device for the local scorer: auto picks Metal on Apple Silicon and CPU elsewhere (CUDA is not in scope on this project)

from __future__ import annotations

import platform

MODES = ("auto", "metal", "cpu")


def resolve_device(
    mode: str, system: str | None = None, machine: str | None = None
) -> str:
    if mode not in MODES:
        raise ValueError(f"device mode must be auto, metal or cpu, got {mode!r}")
    if mode != "auto":
        return mode
    system = system or platform.system()
    machine = machine or platform.machine()
    return "metal" if system == "Darwin" and machine == "arm64" else "cpu"
