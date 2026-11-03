from pathlib import Path

# repository root: src/columnmap/core/paths.py -> three levels up
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SETTINGS = ROOT / "appsettings.json"
SETTINGS_LOCAL = ROOT / "appsettings.local.json"
ENV_FILE = ROOT / ".env"
