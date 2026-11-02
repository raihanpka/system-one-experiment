# One readable JSON file per run: results/<run>/result.json (resumable, rewritten after every case)

from __future__ import annotations

import json
import os
from pathlib import Path

SECTIONS = ("mapping", "mapping_pairs", "transform", "permute")
COMPACT_FROM = 5                                   # nesting depth at which objects are written on a single line


def _round(o, nd: int = 4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def _fmt(o, depth: int = 0) -> str:
    if depth >= COMPACT_FROM or not isinstance(o, (dict, list)) or not o:
        return json.dumps(o, ensure_ascii=False, separators=(", ", ": "))
    pad, end = "  " * (depth + 1), "  " * depth
    if isinstance(o, dict):
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k, ensure_ascii=False)}: {_fmt(v, depth + 1)}" for k, v in o.items()) + f"\n{end}}}"
    return "[\n" + ",\n".join(pad + _fmt(v, depth + 1) for v in o) + f"\n{end}]"


class Store:
    def __init__(self, path, meta: dict | None = None):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"meta": {}, **{s: {} for s in SECTIONS}}
        if meta:
            self.data["meta"].update(meta)

    def rows(self, section: str, cond: str, only=None) -> list[dict]:
        return [r for r in self.data.get(section, {}).get(cond, []) if only is None or r["case_id"] in only]

    def has(self, section: str, cond: str, case_id: str) -> bool:
        return any(r["case_id"] == case_id for r in self.rows(section, cond))

    def add(self, section: str, cond: str, rec: dict) -> None:
        self.data.setdefault(section, {}).setdefault(cond, []).append(_round(rec))
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(_fmt(self.data) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)                 # atomic: a killed run never leaves half a file
