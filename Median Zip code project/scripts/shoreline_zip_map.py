"""
Choropleth map of ZIP codes near Shoreline, colored by median $/lot-sqft.

Uses the cached joined sales+parcel dataframe produced by
dev_score_and_shoreline.py. Pulls WA state ZIP boundary GeoJSON from the
OpenDataDE public mirror (cached in data/) and filters to ZIPs within a
bounding box around Shoreline. The user's seven working ZIPs are drawn
with a thicker outline so they stand out from neighbors.

Output: outputs_current/09_zip_choropleth_map.png
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import chart_style as cs

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_current"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

JOINED_CACHE = DATA_DIR / "kc_market_joined.csv"
GEOJSON_CACHE = DATA_DIR / "wa_zip_codes.geojson"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/"
    "master/wa_washington_zip_codes_geo.min.json"
)

# User's seven working ZIPs — these get thicker borders
WORKING_ZIPS = {"98155", "98133", "98177", "98125", "98103", "98115", "98117"}

# Map view: centered roughly on Shoreline, wide enough to capture the
# residential ring from Edmonds south through Capitol Hill and east to
# Kenmore. Coordinates: (lon_min, lon_max, lat_min, lat_max)
MAP_BOUNDS = (-122.45, -122.20, 47.55, 47.85)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_market_df() -> pd.DataFrame:
    if not JOINED_CACHE.exists():
        raise FileNotFoundError(
            f"Missing cache {JOINED_CACHE}. Run dev_score_and_shoreline.py first."
        )
    df = pd.read_csv(JOINED_CACHE, dtype={"PIN": str, "zip5": str})
    print(f"[data] Loaded {len(df):,} joined sales from {JOINED_CACHE.name}")
    return df


def load_geojson() -> dict:
    """Fetch (and cache) WA state ZIP boundary GeoJSON."""
    if GEOJSON_CACHE.exists() and GEOJSON_CACHE.stat().st_size > 1_000_000:
        print(f"[data] Loading GeoJSON cache {GEOJSON_CACHE.name}")
        with GEOJSON_CACHE.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[data] Downloading WA ZIP boundaries from {GEOJSON_URL}")
    r = requests.get(GEOJSON_URL, timeout=120)
    r.raise_for_status()
    GEOJSON_CACHE.write_bytes(r.content)
    print(f"       Saved {GEOJSON_CACHE.stat().st_size / 1024 / 1024:.1f} MB cache")
    return json.loads(r.content.decode("utf-8"))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def feature_polygons(feature: dict) -> list[list[tuple[float, float]]]:
    """Return list of (lon, lat) ring sequences for a GeoJSON feature.

    GeoJSON polygons can be either Polygon (single outer ring + holes) or
    MultiPolygon (list of polygons). We flatten to a list of outer rings,
    one per polygon — holes are ignored (none expected in ZIP boundaries).
    """
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    rings: list[list[tuple[float, float]]] = []
    if gtype == "Polygon":
        if coords:
            rings.append([(pt[0], pt[1]) for pt in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(pt[0], pt[1]) for pt in poly[0]])
    return rings


def ring_centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    """Quick polygon centroid (signed-area weighted)."""
    if len(ring) < 3:
        return ring[0]
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        # Degenerate: average vertices
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (6 * a), cy / (6 * a)


def ring_in_bounds(ring: list[tuple[float, float]], bounds: tuple) -> bool:
    """True if the ring intersects the map bounding box."""
    lon_min, lon_max, lat_min, lat_max = bounds
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return not (
        max(lons) < lon_min
        or min(lons) > lon_max
        or max(lats) < lat_min
        or min(lats) > lat_max
    )


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def _largest_rings(candidates: list[tuple[str, list]]) -> dict[str, list]:
    """One representative (largest-area) ring per ZIP, for label placement."""
    by_area: dict[str, float] = {}
    largest: dict[str, list] = {}
    for zip5, ring in candidates:
        a = 0.0
        for i in range(len(ring) - 1):
            x0, y0 = ring[i]
            x1, y1 = ring[i + 1]
            a += x0 * y1 - x1 * y0
        a = abs(a) * 0.5
        if a > by_area.get(zip5, 0):
            by_area[zip5] = a
            largest[zip5] = ring
    return largest


def render_map(market_df: pd.DataFrame, geojson: dict) -> None:
    cs.apply()

    # Aggregate market data
    by_zip = (
        market_df.dropna(subset=["price_per_lot_sqft"])
        .groupby("zip5")
        .agg(
            median_ppsf=("price_per_lot_sqft", "median"),
            median_price=("SalePrice", "median"),
            sales_count=("SalePrice", "size"),
        )
    )
    print(f"[agg] {len(by_zip)} ZIPs with $/lot-sqft data")

    # Collect features that fall inside the map view
    candidates: list[tuple[str, list[tuple[float, float]]]] = []
    for feat in geojson["features"]:
        zip5 = str(feat["properties"].get("ZCTA5CE10")
                   or feat["properties"].get("ZIP")
                   or "").zfill(5)
        if not zip5.isdigit() or len(zip5) != 5:
            continue
        for ring in feature_polygons(feat):
            if ring_in_bounds(ring, MAP_BOUNDS):
                candidates.append((zip5, ring))

    print(f"[geo] {len(candidates)} polygons intersect map bounds "
          f"({len({c[0] for c in candidates})} unique ZIPs)")

    # Color scaling: clip the very-low rural outliers so the center of
    # mass of the colormap sits where the residential ZIPs live.
    vals = by_zip["median_ppsf"].dropna()
    vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    cmap = cs.VALUE_CMAP
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    date_min = pd.to_datetime(market_df["SaleDate"]).min().date()
    date_max = pd.to_datetime(market_df["SaleDate"]).max().date()

    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.30], wspace=0.02,
                          left=0.015, right=0.985, top=0.86, bottom=0.04)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])
    panel.axis("off")

    ax.set_xlim(MAP_BOUNDS[0], MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2], MAP_BOUNDS[3])
    ax.set_aspect(1.0 / np.cos(np.radians(47.7)))
    # Water shows through the gaps between land ZIP polygons (Puget Sound,
    # Lake Washington, Lake Union) — a soft blue background reads as water.
    ax.set_facecolor(cs.WATER)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(cs.SPINE)
        s.set_linewidth(1.2)
    ax.grid(False)

    # Draw no-data ZIPs first (land base), then data ZIPs, then focus outlines
    ordered = sorted(
        candidates,
        key=lambda c: (
            c[0] in WORKING_ZIPS,                       # focus drawn last
            pd.notna(by_zip["median_ppsf"].get(c[0])),  # data over no-data
        ),
    )
    for zip5, ring in ordered:
        is_working = zip5 in WORKING_ZIPS
        ppsf = by_zip["median_ppsf"].get(zip5)
        has_data = pd.notna(ppsf)
        face = cmap(norm(ppsf)) if has_data else cs.LAND_NODATA
        ax.add_patch(mpatches.Polygon(
            ring, closed=True, facecolor=face,
            edgecolor=(cs.INK if is_working else "white"),
            linewidth=(2.6 if is_working else 0.7),
            zorder=(5 if is_working else (3 if has_data else 2)),
        ))

    # Labels — one per ZIP at its largest polygon's centroid
    largest = _largest_rings(candidates)
    mid = (vmin + vmax) / 2
    for zip5, ring in largest.items():
        cx, cy = ring_centroid(ring)
        is_working = zip5 in WORKING_ZIPS
        ppsf = by_zip["median_ppsf"].get(zip5)
        has_data = pd.notna(ppsf)
        if not has_data and not is_working:
            # de-emphasize non-focus, no-data neighbors
            ax.text(cx, cy, zip5, ha="center", va="center",
                    fontsize=6.5, color=cs.FAINT, zorder=6)
            continue
        on_dark = has_data and ppsf > mid
        txt_color = "white" if on_dark else cs.INK
        halo = cs.HALO_DARK if on_dark else cs.HALO
        zlabel = ("★ " + zip5) if is_working else zip5
        ax.text(cx, cy + 0.006, zlabel, ha="center", va="center",
                fontsize=(11 if is_working else 8.5),
                fontweight="bold", color=txt_color,
                path_effects=halo, zorder=7)
        if has_data:
            ax.text(cx, cy - 0.010, f"${ppsf:.0f}/sf", ha="center", va="center",
                    fontsize=(9.5 if is_working else 7.5),
                    color=txt_color, path_effects=halo, zorder=7)

    # North arrow (top-right corner of the map)
    ax.annotate("N", xy=(0.965, 0.945), xytext=(0.965, 0.875),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=cs.INK,
                arrowprops=dict(arrowstyle="-|>", color=cs.INK, linewidth=2))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Title block + footer
    cs.title_block(
        fig,
        "Land Value by ZIP — Shoreline & North Seattle",
        f"Median sale price per lot square foot   ·   "
        f"12-month window {date_min:%b %Y} – {date_max:%b %Y}",
    )
    cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels   ·   "
                   "★ = your 7 focus ZIPs (dark outline)")

    # Side panel — ranked focus ZIPs
    work = by_zip.loc[by_zip.index.intersection(WORKING_ZIPS)].copy()
    work = work.sort_values("median_price", ascending=False)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.text(0.0, 0.98, "Your focus ZIPs", fontsize=14, fontweight="bold",
               color=cs.INK, va="top")
    panel.text(0.0, 0.935, "ranked by median sale price", fontsize=9.5,
               color=cs.MUTED, va="top")

    y = 0.86
    row_h = 0.105
    for zip5, r in work.iterrows():
        # accent chip + ZIP
        panel.add_patch(mpatches.Rectangle((0.0, y - 0.052), 0.022, 0.058,
                                           facecolor=cs.ACCENT, edgecolor="none",
                                           transform=panel.transAxes, clip_on=False))
        panel.text(0.05, y, zip5, fontsize=13, fontweight="bold",
                   color=cs.INK, va="center")
        panel.text(0.99, y, cs.usd(r.median_price), fontsize=12.5,
                   color=cs.PRIMARY, fontweight="bold", va="center", ha="right")
        panel.text(0.05, y - row_h * 0.5,
                   f"${r.median_ppsf:.0f}/lot-sqft   ·   {int(r.sales_count)} sales",
                   fontsize=9.5, color=cs.MUTED, va="center")
        panel.plot([0.0, 1.0], [y - row_h * 0.82, y - row_h * 0.82],
                   color=cs.GRID, linewidth=1.0)
        y -= row_h

    # Colorbar legend beneath the focus list (inside the side panel)
    cax = panel.inset_axes([0.0, max(y - 0.02, 0.04), 0.92, 0.022])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5, length=3, color=cs.MUTED)
    cbar.set_label("Median $ / lot sqft   (5th–95th pct scale)",
                   fontsize=9.5, color=cs.INK, labelpad=5)
    cbar.ax.xaxis.set_label_position("top")

    out = OUTPUTS_DIR / "09_zip_choropleth_map.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[out] Saved {out}")

    # Also print a quick summary
    print("\nUser's 7 working ZIPs:")
    print(work[["median_ppsf", "median_price", "sales_count"]].to_string())


def main() -> int:
    try:
        market_df = load_market_df()
        geojson = load_geojson()
        render_map(market_df, geojson)
    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
