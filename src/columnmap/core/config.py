# config.toml: load, check the required keys and value ranges, and turn [device] into model options.
# Secrets are never in config.toml: the API key is read from the environment variable the config names.

from __future__ import annotations

import os
import tomllib

from .device import resolve_device

BACKEND_TYPES = ("local_gguf", "openrouter")
QUANTS = ("F16", "Q8_0", "Q4_K_M")
RUN_MODES = ("options", "pairs", "both")


def _need(cfg: dict, section: str, key: str):
    if key not in cfg.get(section, {}):
        raise ValueError(f"{section}.{key} is required in config.toml")
    return cfg[section][key]


def load_config(text: str) -> dict:
    cfg = tomllib.loads(text)
    cfg.setdefault("backend", {"type": "local_gguf"})
    cfg.setdefault("device", {"mode": "auto"})
    if cfg["backend"].get("type") not in BACKEND_TYPES:
        raise ValueError(f"backend type must be one of {', '.join(BACKEND_TYPES)}, got {cfg['backend'].get('type')!r}")
    if cfg["backend"]["type"] == "openrouter":
        for key in ("model", "base_url", "api_key_env"):
            if key not in cfg["backend"].get("openrouter", {}):
                raise ValueError(f"backend.openrouter.{key} is required in config.toml")
    if _need(cfg, "model", "quant") not in QUANTS:
        raise ValueError(f"model.quant must be one of {', '.join(QUANTS)}")
    _need(cfg, "model", "ngl")
    if _need(cfg, "run", "mode") not in RUN_MODES:
        raise ValueError(f"run.mode must be one of {', '.join(RUN_MODES)}")
    if not 0 < _need(cfg, "run", "target_precision") <= 1:
        raise ValueError("run.target_precision must be in (0, 1]")
    resolve_device(cfg["device"].get("mode", "auto"))              # raises on an unknown mode
    for split in ("dev", "calibration", "test"):
        if not isinstance(_need(cfg, "splits", split), list):
            raise ValueError(f"splits.{split} must be a list of seeds")
    return cfg


# model options for the scorer: the device decides GPU layers (cpu = 0, metal keeps the configured ngl)
def model_options(cfg: dict) -> dict:
    opts = dict(cfg["model"])
    if resolve_device(cfg.get("device", {}).get("mode", "auto")) == "cpu":
        opts["ngl"] = 0
    return opts


# OpenRouter settings: model and base URL from config.toml, the key from the environment only (see .env.example)
def openrouter_settings(cfg: dict, env=None) -> dict:
    env = os.environ if env is None else env
    section = cfg["backend"]["openrouter"]
    name = section["api_key_env"]
    key = env.get(name, "").strip()
    if not key:
        raise ValueError(f"{name} is not set: export it or put it in .env (copy .env.example)")
    return {"model": section["model"], "base_url": section["base_url"], "api_key": key}
