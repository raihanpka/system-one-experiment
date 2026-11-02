"""Run folders: results/<YYYYmmdd-HHMMSS>_<samples>/ sort by time; results/latest points at the newest."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def new_run_dir(root: Path, letters: str, now: datetime | None = None) -> Path:
    d = Path(root) / f"{(now or datetime.now()):%Y%m%d-%H%M%S}_{letters}"
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
    runs = sorted(p for p in Path(root).glob("*_*") if p.is_dir()) if Path(root).exists() else []
    return runs[-1] if runs else None
