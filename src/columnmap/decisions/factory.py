# picks the decider from Backend:Type; adapters are imported only when chosen

from __future__ import annotations

from typing import Callable


def make_decider(cfg: dict) -> Callable:
    from ..core.settings import model_options, openrouter_settings

    if cfg["backend"]["type"] == "openrouter":
        from ..backends.openrouter import OpenRouterDecider

        opts = openrouter_settings(cfg)
        return lambda: OpenRouterDecider(**opts)
    from ..backends.jev_gguf import JevDecider

    opts = model_options(cfg)
    return lambda: JevDecider(**opts)


def backend_meta(cfg: dict) -> dict:
    # what a run records about its backend; calibration keys read model and quant from here
    if cfg["backend"]["type"] == "openrouter":
        return {
            "model": cfg["open_router"]["model"],
            "backend": "openrouter",
            "quant": "api",
        }
    return {"model": "Jev-Style-2B-Decision-v3", "backend": "llama.cpp", **cfg["local"]}
