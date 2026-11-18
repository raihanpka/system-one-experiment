# run folders: results/<samples>_<YYYYmmdd-HHMMSS>/, for example identical-synonym_20261102-153012

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def new_run_dir(root: Path, names: str, now: datetime | None = None) -> Path:
    d = Path(root) / f"{names}_{(now or datetime.now()):%Y%m%d-%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# the newest run folder, found from the timestamp at the end of the folder name
def newest_run(root: Path) -> Path | None:
    runs = (
        [p for p in Path(root).glob("*_*") if p.is_dir()] if Path(root).exists() else []
    )
    return max(runs, key=lambda p: p.name.rsplit("_", 1)[-1], default=None)
