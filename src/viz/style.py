"""Shared plotting style for the stage notebooks.

This module sets how exploratory and QC plots look; it contains no figure
functions. The eight reported figures live in ``src/viz/figures.py``.

Rules applied here: one fixed categorical order that is never cycled; one hue
for magnitude; thin marks; a recessive grid; text in ink colours rather than
series colours; one y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path

#: Categorical slots, assigned in this order and never cycled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948")
#: Sequential blue, light to dark, for ordinal stages such as a funnel.
SEQUENTIAL = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def apply() -> None:
    """Install the style for every subsequent matplotlib figure."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.prop_cycle": cycler(color=list(SERIES)),
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 2.0, "patch.linewidth": 0,
        "legend.frameon": False, "legend.fontsize": 8,
        "font.size": 9, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, figures_dir: Path, stage: str, name: str) -> Path:
    """Save a figure under ``figures/<stage>/<name>.png`` and return the path."""
    out = Path(figures_dir) / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    return path


__all__ = ["SERIES", "SEQUENTIAL", "SURFACE", "INK", "INK_SECONDARY", "GRID", "apply", "save"]
