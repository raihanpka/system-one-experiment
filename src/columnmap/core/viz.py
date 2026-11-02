# Shared figure style: seaborn theme with Times New Roman; SVG output keeps text as text

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT = "Times New Roman"


# Seaborn theme with Times New Roman. Returns False if the font is not installed (falls back to a serif)
def setup_style() -> bool:
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
        ok = True
    except ValueError:
        ok = False
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind",
                  rc={"font.family": "serif", "font.serif": [FONT, "Times", "DejaVu Serif"], "svg.fonttype": "none",
                      "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 100})
    return ok


ISSUES: list[str] = []


def legend_outside(ax, **kw):
    # legend beside the plot, never on top of the data
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, **kw)
    leg.set_gid("legend-outside")
    return leg


def save(fig, path) -> Path:
    from .figcheck import check_figure
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.tick_params(labelsize=9, pad=4)
    ISSUES.extend(f"{path.name}: {m}" for m in check_figure(fig))
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
