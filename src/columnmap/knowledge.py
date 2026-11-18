# What the mapping needs to know about the target: the output fields and the reference tables (code lists).
# Switch the data by pointing to another folder:
#   <folder>/outputs.json     [{"name": ..., "type": "string" | "number" | "date", "description": ...}, ...]
#   <folder>/reference/*.csv  columns code,label (optional; one file per code list)

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Output:
    name: str
    type: str  # string | number | date
    description: str


@dataclass(frozen=True)
class Knowledge:
    outputs: tuple[Output, ...]
    tables: tuple[dict, ...] = ()  # code -> label
    prompts: tuple[tuple[str, str], ...] = ()  # prompt name -> template override

    @classmethod
    def load(cls, folder) -> Knowledge:
        folder = Path(folder)
        outputs = tuple(
            Output(o["name"], o["type"], o["description"])
            for o in json.loads((folder / "outputs.json").read_text(encoding="utf-8"))
        )
        tables = []
        for f in sorted((folder / "reference").glob("*.csv")):
            with f.open(encoding="utf-8", newline="") as fh:
                tables.append({r["code"]: r["label"] for r in csv.DictReader(fh)})
        prompts = ()
        prompt_file = folder / "prompts.json"
        if prompt_file.exists():
            raw = json.loads(prompt_file.read_text(encoding="utf-8"))
            prompts = tuple(
                sorted(
                    (k, v["template"] if isinstance(v, dict) else v)
                    for k, v in raw.items()
                )
            )
        return cls(outputs, tuple(tables), prompts)

    def save(self, folder) -> None:
        folder = Path(folder)
        (folder / "reference").mkdir(parents=True, exist_ok=True)
        rows = [
            {"name": o.name, "type": o.type, "description": o.description}
            for o in self.outputs
        ]
        (folder / "outputs.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for n, table in enumerate(self.tables, 1):
            with (folder / "reference" / f"table_{n}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as fh:
                w = csv.writer(fh)
                w.writerow(["code", "label"])
                w.writerows(table.items())
        if self.prompts:
            (folder / "prompts.json").write_text(
                json.dumps(dict(self.prompts), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    # the folder's overrides first, then the chosen backend (built-in file registry or MLflow)
    def registry(self, base=None):
        from .prompts import DEFAULT
        from .registry import ChainRegistry, FileRegistry

        base = base or DEFAULT
        if not self.prompts:
            return base
        return ChainRegistry(FileRegistry(dict(self.prompts)), base)
