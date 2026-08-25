from __future__ import annotations

import os
from pathlib import Path


_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg", force=True)

DPI = 240
FONT_FAMILY = "serif"
FONT_SERIF = [
    "Latin Modern Roman",
    "Computer Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
]
REPORT_BASE_FONT = 9.8
REPORT_SMALL_FONT = 8.4
REPORT_AXIS_FONT = 9.6
REPORT_LEGEND_FONT = 8.0
REPORT_PANEL_FONT = 10.6

SCRATCH_COLOR = "#0072B2"
RESNET_COLOR = "#D55E00"
REFERENCE_COLOR = "#009E73"
BASELINE_COLOR = "#6B7280"
POSITIVE_RESIDUAL_COLOR = "#B2182B"
NEGATIVE_RESIDUAL_COLOR = "#2166AC"
GRID_COLOR = "#D7DCE2"
MAP_CONTEXT_FACE = "#F1F3F5"
MAP_INTERNAL_EDGE = "#FFFFFF"
MAP_OUTER_EDGE = "#9AA3AD"
BOUNDARY_COLOR = MAP_OUTER_EDGE
INNER_BOUNDARY_COLOR = MAP_INTERNAL_EDGE
NODATA_COLOR = MAP_CONTEXT_FACE
CONTEXT_COLOR = MAP_CONTEXT_FACE
TEXT_COLOR = "#1F2933"

PM_CMAP = "viridis"
RESIDUAL_CMAP = "RdBu_r"
SOCIOECONOMIC_CMAP = "cividis"

FIGSIZE_SINGLE = (6.8, 4.0)
FIGSIZE_TWO_PANEL = (7.2, 3.4)
FIGSIZE_MAP_TWO_PANEL = (7.2, 3.6)
FIGSIZE_WIDE = (7.2, 2.6)

PALETTE = {
    "blue": SCRATCH_COLOR,
    "orange": RESNET_COLOR,
    "green": REFERENCE_COLOR,
    "red": POSITIVE_RESIDUAL_COLOR,
    "purple": "#8063AC",
    "grey": BASELINE_COLOR,
}

MODEL_COLORS = {
    "cnn_deep_wide": SCRATCH_COLOR,
    "scratch": SCRATCH_COLOR,
    "Scratch CNN": SCRATCH_COLOR,
    "resnet_frozen": RESNET_COLOR,
    "ResNet frozen": RESNET_COLOR,
    "baseline": BASELINE_COLOR,
}


def apply(context: str = "paper") -> None:
    import matplotlib.pyplot as plt

    rc = {
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "font.family": FONT_FAMILY,
        "font.serif": FONT_SERIF,
        "font.size": REPORT_BASE_FONT,
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "text.color": TEXT_COLOR,
        "axes.titlesize": REPORT_BASE_FONT,
        "axes.titleweight": "semibold",
        "axes.labelsize": REPORT_AXIS_FONT,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": BOUNDARY_COLOR,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "xtick.labelsize": REPORT_SMALL_FONT,
        "ytick.labelsize": REPORT_SMALL_FONT,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.fontsize": REPORT_LEGEND_FONT,
        "legend.title_fontsize": REPORT_LEGEND_FONT,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "patch.edgecolor": BOUNDARY_COLOR,
    }
    try:
        import seaborn as sns

        sns.set_theme(context=context, style="whitegrid", font=FONT_FAMILY, rc=rc)
    except ImportError:
        plt.rcParams.update(rc)


def clean_axis(ax, *, grid: bool = True) -> None:
    ax.grid(grid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_map_axis(ax, extent: tuple[float, float, float, float] | None = None) -> None:
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")


def plot_country_context(ax, countries, **kwargs) -> None:
    opts = {
        "facecolor": MAP_CONTEXT_FACE,
        "edgecolor": MAP_INTERNAL_EDGE,
        "linewidth": 0.28,
        "zorder": 1,
    }
    opts.update(kwargs)
    countries.plot(ax=ax, **opts)


def plot_germany_state_context(ax, states, **kwargs) -> None:
    opts = {
        "facecolor": MAP_CONTEXT_FACE,
        "edgecolor": MAP_INTERNAL_EDGE,
        "linewidth": 0.28,
        "zorder": 1,
    }
    opts.update(kwargs)
    states.plot(ax=ax, **opts)


def plot_data_polygons(ax, polygons, **kwargs) -> None:
    if getattr(polygons, "empty", False):
        return
    opts = {
        "edgecolor": MAP_INTERNAL_EDGE,
        "linewidth": 0.32,
        "zorder": 2,
    }
    opts.update(kwargs)
    polygons.plot(ax=ax, **opts)


def plot_outer_outline(ax, polygons, **kwargs) -> None:
    if getattr(polygons, "empty", False):
        return
    opts = {
        "color": MAP_OUTER_EDGE,
        "linewidth": 0.55,
        "zorder": 8,
    }
    opts.update(kwargs)
    polygons.dissolve().boundary.plot(ax=ax, **opts)


def panel_label(ax, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=REPORT_PANEL_FONT,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def model_label(model: str) -> str:
    labels = {
        "cnn_deep_wide": "Scratch CNN",
        "resnet_frozen": "Frozen ResNet",
    }
    return labels.get(model, model)


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, BASELINE_COLOR)


def savefig(fig, path: str | Path, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opts = {"dpi": DPI, "bbox_inches": "tight", "facecolor": "white"}
    opts.update(kwargs)
    fig.savefig(path, **opts)
