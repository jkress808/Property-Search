"""
Shared chart styling for the King County market-analysis pipeline.

Import and call apply() at the top of each plotting script to give every
output a single, consistent, professional look: clean sans-serif type, soft
gridlines, no boxed-in spines, a coherent brand palette, and high-DPI export.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import seaborn as sns

# --- Brand palette ---------------------------------------------------------
INK = "#1b2733"          # primary text / dark elements
MUTED = "#5c6b7a"        # secondary text, axis labels
FAINT = "#9aa7b2"        # tertiary / footnotes
GRID = "#e2e7ec"         # gridlines
SPINE = "#c7d0d8"        # axis spines
PRIMARY = "#1d557a"      # primary series color (deep blue)
ACCENT = "#e8722c"       # highlight / "your focus" accent (warm orange)
POSITIVE = "#2a9d8f"     # teal for favorable values
WATER = "#cfe0ec"        # map water bodies / background
LAND_NODATA = "#eae6de"  # map land polygons with no data

# Sequential colormap for value maps and ranked bars (light -> deep teal/blue)
VALUE_CMAP = sns.color_palette("crest", as_cmap=True)

# Halo for text drawn over busy/colored backgrounds (e.g., map labels)
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
HALO_DARK = [pe.withStroke(linewidth=2.6, foreground="#0d1620")]


def apply() -> None:
    """Set global matplotlib rcParams for the house style."""
    sns.set_theme(style="white", context="talk")
    mpl.rcParams.update({
        # canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        # type
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 11,
        "text.color": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 12,
        "axes.labelsize": 11.5,
        "axes.labelcolor": MUTED,
        "axes.labelweight": "normal",
        # spines & ticks
        "axes.edgecolor": SPINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        # grid
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        # legend
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10.5,
        # figure-level title
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    })


def title_block(fig, title: str, subtitle: str | None = None,
                x: float = 0.012, y: float = 0.985) -> None:
    """Top-left bold title with an optional muted subtitle line beneath it."""
    fig.text(x, y, title, ha="left", va="top",
             fontsize=18, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, y - 0.052, subtitle, ha="left", va="top",
                 fontsize=11, color=MUTED)


def footer(fig, text: str, x: float = 0.012, y: float = 0.012) -> None:
    """Consistent muted source/attribution line at the bottom-left."""
    fig.text(x, y, text, ha="left", va="bottom", fontsize=8.5, color=FAINT)


def grid_x_only(ax) -> None:
    """Show only vertical gridlines (for horizontal bar charts)."""
    ax.grid(True, axis="x", color=GRID, linewidth=1.0)
    ax.grid(False, axis="y")


def grid_y_only(ax) -> None:
    """Show only horizontal gridlines (for vertical bar / line charts)."""
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.grid(False, axis="x")


def style_table(tbl, n_rows: int, n_cols: int, header_bg: str = PRIMARY,
                fontsize: float = 9.0) -> None:
    """Give a matplotlib table a clean banded look: colored header row,
    alternating row shading, no heavy cell borders."""
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.0)
        if r == 0:  # header
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f4f6f8" if r % 2 else "white")
            cell.set_text_props(color=INK)


def usd(v: float) -> str:
    """Compact USD label: $1.2M / $850K / $1,200."""
    if v is None or v != v:  # NaN
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:.2f}M".replace(".00M", "M")
    if abs(v) >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"
