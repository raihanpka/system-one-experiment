# Settings in the .NET style: appsettings.json (committed defaults, PascalCase), appsettings.local.json (optional, gitignored),
# then environment variables COLUMNMAP__Section__Key, which win over both. Code reads snake_case keys.
# Secrets are never in these files: the API key is read from the environment variable that OpenRouter:ApiKeyEnv names.

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .paths import ENV_FILE, SETTINGS, SETTINGS_LOCAL

PREFIX = "COLUMNMAP__"


# .env in the project root, loaded once at start (stdlib, no dependency). Values already in the shell win; the file never overrides them.
def load_dotenv(path=ENV_FILE, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    loaded = {}
    p = Path(path)
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[
                0
            ].rstrip()  # inline comment after a bare value
        if key and key not in environ:
            environ[key] = value
            loaded[key] = value
    return loaded


RUN_MODES = ("options", "pairs", "both")
REQUIRED_OPENROUTER = {
    "model": "Model",
    "base_url": "BaseUrl",
    "api_key_env": "ApiKeyEnv",
}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def normalize(obj):
    if isinstance(obj, dict):
        return {snake(k): normalize(v) for k, v in obj.items()}
    return obj


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def env_overrides(env) -> dict:
    out: dict = {}
    for name, value in env.items():
        if not name.startswith(PREFIX):
            continue
        parts = [snake(p) for p in name[len(PREFIX) :].split("__")]
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(value)  # numbers, lists, true/false, null
        except ValueError:
            node[parts[-1]] = value
    return out


def validate(cfg: dict) -> dict:
    section = cfg.get("open_router", {})
    for key, pascal in REQUIRED_OPENROUTER.items():
        if not section.get(key):
            raise ValueError(f"OpenRouter:{pascal} is required in appsettings.json")
    run = cfg.get("run", {})
    if run.get("mode") not in RUN_MODES:
        raise ValueError(f"Run:Mode must be one of {', '.join(RUN_MODES)}")
    if not 0 < float(run.get("target_precision", 0)) <= 1:
        raise ValueError("Run:TargetPrecision must be in (0, 1]")
    if "seed" not in run:
        raise ValueError("Run:Seed is required")
    for split in ("dev", "calibration", "test"):
        if not isinstance(cfg.get("splits", {}).get(split), list):
            raise ValueError(f"Splits:{split.capitalize()} must be a list of seeds")
    return cfg


def load_settings(defaults=SETTINGS, local=SETTINGS_LOCAL, env=None) -> dict:
    if env is None:
        load_dotenv()
    env = os.environ if env is None else env
    cfg = normalize(json.loads(Path(defaults).read_text(encoding="utf-8")))
    if Path(local).exists():
        cfg = merge(cfg, normalize(json.loads(Path(local).read_text(encoding="utf-8"))))
    return validate(merge(cfg, env_overrides(env)))


# OpenRouter settings: model and base URL from the file, the key from the environment only
def openrouter_settings(cfg: dict, env=None) -> dict:
    env = os.environ if env is None else env
    section = cfg["open_router"]
    name = section["api_key_env"]
    key = env.get(name, "").strip()
    if not key:
        raise ValueError(
            f"{name} is not set: export it or put it in .env (copy .env.example)"
        )
    return {"model": section["model"], "base_url": section["base_url"], "api_key": key}
