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
from matplotlib.collections import PatchCollection

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

def render_map(market_df: pd.DataFrame, geojson: dict) -> None:
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
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(MAP_BOUNDS[0], MAP_BOUNDS[1])
    ax.set_ylim(MAP_BOUNDS[2], MAP_BOUNDS[3])
    # Approximate equal-aspect for this latitude
    ax.set_aspect(1.0 / np.cos(np.radians(47.7)))
    ax.set_facecolor("#e8edf0")

    drawn_zips: set[str] = set()
    for zip5, ring in candidates:
        is_working = zip5 in WORKING_ZIPS
        ppsf = by_zip["median_ppsf"].get(zip5)

        if pd.notna(ppsf):
            color = cmap(norm(ppsf))
        else:
            color = "#cccccc"

        edge = "black" if is_working else "#555555"
        lw = 2.2 if is_working else 0.6

        poly = mpatches.Polygon(
            ring,
            closed=True,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.92 if pd.notna(ppsf) else 0.55,
        )
        ax.add_patch(poly)
        drawn_zips.add(zip5)

    # Label each ZIP at its centroid (one label per ZIP, biggest polygon)
    zip_to_largest: dict[str, list[tuple[float, float]]] = {}
    zip_to_area: dict[str, float] = {}
    for zip5, ring in candidates:
        # Approx area via shoelace
        a = 0.0
        for i in range(len(ring) - 1):
            x0, y0 = ring[i]
            x1, y1 = ring[i + 1]
            a += x0 * y1 - x1 * y0
        a = abs(a) * 0.5
        if a > zip_to_area.get(zip5, 0):
            zip_to_area[zip5] = a
            zip_to_largest[zip5] = ring

    for zip5, ring in zip_to_largest.items():
        cx, cy = ring_centroid(ring)
        is_working = zip5 in WORKING_ZIPS
        ppsf = by_zip["median_ppsf"].get(zip5)
        label = f"{zip5}\n${ppsf:.0f}/sf" if pd.notna(ppsf) else zip5
        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=8.5 if is_working else 7,
            fontweight="bold" if is_working else "normal",
            color="white" if pd.notna(ppsf) and ppsf > (vmin + vmax) / 2 else "#111",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=("#000000" if is_working else "none"),
                alpha=0.35 if is_working else 0.0,
                edgecolor="none",
            ) if is_working else None,
        )

    # Title & colorbar
    ax.set_title(
        "Median $/lot-sqft by ZIP — Shoreline area\n"
        "(thick black border = user's 7 working ZIPs · 12-mo window: "
        f"{pd.to_datetime(market_df['SaleDate']).min().date()} → "
        f"{pd.to_datetime(market_df['SaleDate']).max().date()})",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Median $ / lot sqft (5th–95th pct color scale)")

    # Side panel: ranked list of the 7 working ZIPs by $/sqft
    work_summary = by_zip.loc[by_zip.index.intersection(WORKING_ZIPS)].copy()
    work_summary = work_summary.sort_values("median_ppsf", ascending=False)
    info = "User's 7 working ZIPs (ranked by $/lot-sqft):\n"
    for zip5, row in work_summary.iterrows():
        info += (
            f"  {zip5}  ${row.median_ppsf:>5.0f}/sf  "
            f"med ${row.median_price/1000:>5.0f}k  "
            f"n={int(row.sales_count)}\n"
        )
    ax.text(
        0.01, 0.01, info.strip(),
        transform=ax.transAxes,
        fontsize=8.5, family="monospace",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#888", alpha=0.92),
    )

    fig.tight_layout()
    out = OUTPUTS_DIR / "09_zip_choropleth_map.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[out] Saved {out}")

    # Also print a quick summary
    print("\nUser's 7 working ZIPs:")
    print(work_summary[["median_ppsf", "median_price", "sales_count"]]
          .to_string())


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
