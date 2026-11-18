# Prompt templates live in a registry, not in an f-string. The default is the built-in book; a knowledge
# folder can override any prompt in prompts.json (source-controlled with the dataset). MLflow is an optional
# backend for versioned prompts; see registry_mlflow.py.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    template: str
    version: str | None = None

    def __post_init__(self):
        if not isinstance(self.template, str):
            raise ValueError(f"prompt {self.name!r} is not a text template")

    def format(self, **values) -> str:
        def replace(match):
            key = match.group(1)
            if key not in values:
                raise KeyError(f"prompt {self.name!r} needs {key!r}")
            return str(values[key])

        return _VAR.sub(replace, self.template)


class Registry(Protocol):
    def get(self, name: str, version: str | None = None) -> Prompt: ...


class ChainRegistry:
    """Ask each registry in order; the first one that has the name wins."""

    def __init__(self, *registries: Registry):
        self._registries = registries

    def get(self, name: str, version: str | None = None) -> Prompt:
        for registry in self._registries:
            try:
                return registry.get(name, version)
            except KeyError:
                continue
        raise KeyError(f"no prompt named {name!r} in any registry")


class FileRegistry:
    def __init__(self, templates: Mapping[str, str], version: str | None = None):
        self._templates = dict(templates)
        self._version = version

    def get(self, name: str, version: str | None = None) -> Prompt:
        if name not in self._templates:
            raise KeyError(f"no prompt named {name!r} in the registry")
        return Prompt(name, self._templates[name], version or self._version)

    def overlay(self, templates: Mapping[str, str]) -> FileRegistry:
        return FileRegistry({**self._templates, **dict(templates)}, self._version)

    @property
    def names(self) -> list[str]:
        return sorted(self._templates)

    @classmethod
    def load(cls, folder, name: str = "prompts.json") -> FileRegistry:
        path = Path(folder) / name
        if not path.exists():
            return cls({})
        raw = json.loads(path.read_text(encoding="utf-8"))
        templates = {}
        for key, value in raw.items():
            templates[key] = value["template"] if isinstance(value, dict) else value
        return cls(templates)
