"""Shared figure style: seaborn theme with Times New Roman; SVG output keeps text as text."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT = "Times New Roman"


def setup_style() -> bool:
    """Seaborn theme with Times New Roman. Returns False if the font is not installed (falls back to a serif)."""
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
        ok = True
    except ValueError:
        ok = False
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind",
                  rc={"font.family": "serif", "font.serif": [FONT, "Times", "DejaVu Serif"], "svg.fonttype": "none",
                      "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 100})
    return ok


def save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path
