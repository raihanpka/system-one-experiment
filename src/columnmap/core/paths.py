from pathlib import Path

# repository root: src/columnmap/core/paths.py -> three levels up
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
CONFIG = ROOT / "config.toml"
VENDOR = ROOT / "vendor" / "jev-v3-2b-gguf"
