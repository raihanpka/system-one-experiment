# One dataset, one policy: the fill floors, the model they were calibrated for, and the calibration
# facts that justify them. Stored next to outputs.json and prompts.json, so production never edits a
# global setting per dataset.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS = {"derived": 0.75, "copy": 0.4, "none_ceiling": 0.5}


@dataclass(frozen=True)
class Policy:
    derived: float = DEFAULTS["derived"]
    copy: float = DEFAULTS["copy"]
    none_ceiling: float = DEFAULTS["none_ceiling"]
    model: str | None = None
    calibration: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in DEFAULTS:
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"policy {name} must be in [0, 1]")

    def fill(self):
        from .mapper import Fill

        return Fill(self.derived, self.copy, self.none_ceiling)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "fill": {
                "derived": self.derived,
                "copy": self.copy,
                "none_ceiling": self.none_ceiling,
            },
            "calibration": self.calibration,
        }

    @classmethod
    def load(cls, folder, name: str = "policy.json"):
        path = Path(folder) / name
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        fill = raw.get("fill", {})
        return cls(
            derived=float(fill.get("derived", DEFAULTS["derived"])),
            copy=float(fill.get("copy", DEFAULTS["copy"])),
            none_ceiling=float(fill.get("none_ceiling", DEFAULTS["none_ceiling"])),
            model=raw.get("model"),
            calibration=raw.get("calibration", {}),
        )

    def save(self, folder, name: str = "policy.json") -> Path:
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
