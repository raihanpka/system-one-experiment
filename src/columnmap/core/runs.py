"""Run folders: results/<samples>_<YYYYmmdd-HHMMSS>/ e.g. identical-synonym_20261102-153012; results/latest points at the newest."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def new_run_dir(root: Path, names: str, now: datetime | None = None) -> Path:
    d = Path(root) / f"{names}_{(now or datetime.now()):%Y%m%d-%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def link_latest(root: Path, run_dir: Path) -> None:
    link = Path(root) / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(Path(run_dir).name, link)                   # relative, so the results folder can be moved


def latest_run(root: Path) -> Path | None:
    link = Path(root) / "latest"
    if link.exists():
        return link.resolve()
    runs = [p for p in Path(root).glob("*_*") if p.is_dir() and not p.is_symlink()] if Path(root).exists() else []
    return max(runs, key=lambda p: p.name.rsplit("_", 1)[-1], default=None)
