"""
Two analyses on top of the live King County GIS market data:

    A. Developer Opportunity Score by ZIP — composite z-score combining
       end-value (median sale price), land cost (median $/lot-sqft, inverted),
       lot size (median LOTSQFT), and teardown supply (% single-family
       residences from PREUSE_DESC). Higher = better dev play.

    B. Shoreline / north-Seattle deep dive — filtered to the ZIPs the user
       transacts in (98155, 98133, 98177, 98125, 98103, 98115, 98117) with
       per-ZIP comparison, monthly trends, and a CSV of undervalued sales
       (sold >25% below their ZIP's median $/lot-sqft) to feed an outreach
       pipeline.

Inputs:
    Re-uses the same GIS endpoints as current_market_analysis.py. Caches the
    joined sales+parcel dataframe to data/kc_market_joined.csv so subsequent
    runs skip the ~30s network pull.

Outputs (PNG/CSV):
    outputs_current/dev_score_by_zip.csv
    outputs_current/shoreline_targets.csv
    outputs_current/05_dev_score_top20.png
    outputs_current/06_shoreline_comparison.png
    outputs_current/07_shoreline_monthly_trends.png
    outputs_current/08_shoreline_snapshot.png
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

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

GIS_BASE = (
    "https://gismaps.kingcounty.gov/arcgis/rest/services/Property/"
    "KingCo_PropertyInfo/MapServer"
)
SALES_LAYER = f"{GIS_BASE}/3/query"
PARCEL_LAYER = f"{GIS_BASE}/2/query"

MIN_PRICE = 10_000
LOOKBACK_MONTHS = 12
SALES_PAGE_SIZE = 1000
PARCEL_PIN_BATCH = 400
REQUEST_TIMEOUT = 60

# User's working ZIPs (Shoreline + north Seattle)
SHORELINE_ZIPS = ["98155", "98133", "98177", "98125", "98103", "98115", "98117"]

# Statistical cutoff: ignore ZIPs with too few sales for the dev score
MIN_SALES_FOR_SCORE = 25

# Acquisition target threshold: sales priced >25% below ZIP median $/lot-sqft
UNDERVALUE_PCT = 0.25

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Data pull (same shape as current_market_analysis.py, but cached)
# ---------------------------------------------------------------------------

def fetch_recent_sales() -> pd.DataFrame:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=14)
    where = (
        f"Principal_Use = 'RESIDENTIAL' "
        f"AND SalePrice >= {MIN_PRICE} "
        f"AND SaleDate > DATE '{cutoff.strftime('%Y-%m-%d %H:%M:%S')}'"
    )
    print(f"  Sales fetch: {where}")
    feats: list[dict] = []
    offset = 0
    page = 0
    while True:
        params = {
            "where": where,
            "outFields": "PIN,SaleDate,SalePrice,Principal_Use,Property_Type,address",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": SALES_PAGE_SIZE,
            "orderByFields": "OBJECTID ASC",
            "f": "json",
        }
        r = requests.post(SALES_LAYER, data=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"Sales API error: {body['error']}")
        page_feats = body.get("features", [])
        feats.extend(page_feats)
        page += 1
        print(f"    page {page:>3}  +{len(page_feats):>4}  total {len(feats):,}")
        if not body.get("exceededTransferLimit") and len(page_feats) < SALES_PAGE_SIZE:
            break
        offset += SALES_PAGE_SIZE
        time.sleep(0.1)

    df = pd.DataFrame([f["attributes"] for f in feats])
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], unit="ms", errors="coerce")
    df = df.dropna(subset=["SaleDate", "PIN", "SalePrice"])
    max_d = df["SaleDate"].max()
    df = df[df["SaleDate"] > (max_d - pd.DateOffset(months=LOOKBACK_MONTHS))]
    return df


def fetch_parcel_attrs(pins: list[str]) -> pd.DataFrame:
    print(f"  Parcel fetch: {len(pins):,} PINs in {math.ceil(len(pins)/PARCEL_PIN_BATCH)} batches")
    rows: list[dict] = []
    batches = math.ceil(len(pins) / PARCEL_PIN_BATCH)
    for i in range(batches):
        chunk = pins[i * PARCEL_PIN_BATCH:(i + 1) * PARCEL_PIN_BATCH]
        in_list = ",".join(f"'{p}'" for p in chunk)
        params = {
            "where": f"PIN IN ({in_list})",
            "outFields": "PIN,ZIP5,LOTSQFT,PREUSE_DESC,PROPTYPE,CTYNAME,POSTALCTYNAME,ADDR_FULL",
            "returnGeometry": "false",
            "f": "json",
        }
        r = requests.post(PARCEL_LAYER, data=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"Parcel API error batch {i+1}: {body['error']}")
        rows.extend(f["attributes"] for f in body.get("features", []))
        if (i + 1) % 10 == 0 or i + 1 == batches:
            print(f"    batch {i+1}/{batches}  total {len(rows):,}")
        time.sleep(0.05)
    return pd.DataFrame(rows)


def load_joined() -> pd.DataFrame:
    """Load cached joined dataframe, or pull fresh from GIS."""
    if JOINED_CACHE.exists():
        age_h = (time.time() - JOINED_CACHE.stat().st_mtime) / 3600
        if age_h < 168:  # 7 days
            print(f"[data] Loading cache {JOINED_CACHE.name} (age {age_h:.1f}h)")
            df = pd.read_csv(JOINED_CACHE, dtype={"PIN": str, "zip5": str})
            df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
            print(f"       {len(df):,} rows loaded from cache")
            return df

    print("[data] No fresh cache — pulling from GIS")
    sales = fetch_recent_sales()
    pins = sales["PIN"].astype(str).unique().tolist()
    parcels = fetch_parcel_attrs(pins)
    merged = sales.merge(parcels, on="PIN", how="inner")
    merged["zip5"] = merged["ZIP5"].astype(str).str.strip().str.zfill(5)
    merged = merged[merged["zip5"].str.match(r"^\d{5}$")]
    merged["LOTSQFT"] = pd.to_numeric(merged["LOTSQFT"], errors="coerce")
    merged["price_per_lot_sqft"] = merged.apply(
        lambda r: (r["SalePrice"] / r["LOTSQFT"]) if r["LOTSQFT"] and r["LOTSQFT"] > 0 else None,
        axis=1,
    )
    merged.to_csv(JOINED_CACHE, index=False)
    print(f"[data] Cached {len(merged):,} joined rows -> {JOINED_CACHE}")
    return merged


# ---------------------------------------------------------------------------
# A. Developer Opportunity Score by ZIP
# ---------------------------------------------------------------------------

def is_single_family(pre: str | float) -> bool:
    """PREUSE_DESC strings that indicate a teardown-candidate SFR.

    King County uses descriptions like 'Single Family(Res Use/Zone)',
    'Single Family(C/I Zone)', and 'Vacant(Single-family)' for SFR parcels.
    Multifamily and condo descriptions are excluded.
    """
    if not isinstance(pre, str):
        return False
    p = pre.lower()
    if "single family" in p:
        return True
    return False


def compute_dev_score(df: pd.DataFrame) -> pd.DataFrame:
    """Composite z-score per ZIP for developer/teardown plays.

    Components (each z-scored, higher = better for the dev thesis):
      + median sale price       (end-value upside)
      + median lot sqft         (developable land)
      - median $/lot-sqft       (land cost; negate so cheaper land helps)
      + pct single-family       (teardown supply)

    Equal-weight sum gives an interpretable 4-component composite.
    """
    df = df.copy()
    df["is_sfr"] = df["PREUSE_DESC"].apply(is_single_family)

    grouped = df.groupby("zip5").agg(
        median_price=("SalePrice", "median"),
        median_lot_sqft=("LOTSQFT", "median"),
        median_ppsf=("price_per_lot_sqft", "median"),
        pct_sfr=("is_sfr", "mean"),
        sales_count=("SalePrice", "size"),
    ).reset_index()

    # Require enough sales to make medians meaningful
    grouped = grouped[grouped["sales_count"] >= MIN_SALES_FOR_SCORE].copy()
    grouped = grouped.dropna(subset=["median_ppsf"])

    def z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0)

    grouped["z_price"] = z(grouped["median_price"])
    grouped["z_lot"] = z(grouped["median_lot_sqft"])
    grouped["z_ppsf_inv"] = -z(grouped["median_ppsf"])  # cheaper land = better
    grouped["z_sfr"] = z(grouped["pct_sfr"])

    grouped["dev_score"] = (
        grouped["z_price"] + grouped["z_lot"] + grouped["z_ppsf_inv"] + grouped["z_sfr"]
    )
    grouped = grouped.sort_values("dev_score", ascending=False).reset_index(drop=True)
    return grouped


def render_dev_score(score_df: pd.DataFrame) -> None:
    top20 = score_df.head(20).iloc[::-1]  # reverse so highest bar is at top
    zips = top20["zip5"].astype(str).tolist()
    scores = top20["dev_score"].tolist()

    fig, ax = plt.subplots(figsize=(12, 9))
    norm = plt.Normalize(min(scores), max(scores))
    colors = [
        cs.ACCENT if z in SHORELINE_ZIPS else cs.VALUE_CMAP(norm(s))
        for z, s in zip(zips, scores)
    ]
    bars = ax.barh(zips, scores, color=colors, edgecolor="white", linewidth=0.6)

    # value labels at bar ends
    span = max(scores) - min(0, min(scores))
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + span * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{s:.2f}", va="center", ha="left", fontsize=9.5, color=cs.MUTED)

    ax.set_xlabel("Composite developer score  (z-sum across 4 factors)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(scores) * 1.12)
    cs.grid_x_only(ax)
    ax.tick_params(axis="y", labelsize=11)
    # bold the focus-ZIP tick labels
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in SHORELINE_ZIPS:
            lbl.set_fontweight("bold")
            lbl.set_color(cs.INK)

    cs.title_block(
        fig,
        "Developer Opportunity Score — Top 20 ZIPs",
        "Higher = more upside: high end-value + big lots + cheap land + more "
        "single-family teardown supply",
    )
    cs.footer(fig, "Score = z(median price) + z(median lot) − z($/lot-sqft) + "
                   "z(%SFR)   ·   orange = your focus ZIPs")
    fig.subplots_adjust(top=0.86, left=0.08, right=0.96, bottom=0.12)
    out = OUTPUTS_DIR / "05_dev_score_top20.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# B. Shoreline / north-Seattle deep dive
# ---------------------------------------------------------------------------

def shoreline_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["zip5"].isin(SHORELINE_ZIPS)]
    out = sub.groupby("zip5").agg(
        median_price=("SalePrice", "median"),
        median_ppsf=("price_per_lot_sqft", "median"),
        median_lot_sqft=("LOTSQFT", "median"),
        sales_count=("SalePrice", "size"),
        pct_sfr=("PREUSE_DESC", lambda s: s.apply(is_single_family).mean()),
    ).reset_index()
    out["zip5"] = pd.Categorical(out["zip5"], categories=SHORELINE_ZIPS, ordered=True)
    return out.sort_values("zip5").reset_index(drop=True)


def render_shoreline_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    sp = summary.copy()
    sp["zip5"] = sp["zip5"].astype(str)
    zips = sp["zip5"].tolist()

    # Median price by ZIP (sorted high -> low for readability)
    ax1 = axes[0]
    s1 = sp.sort_values("median_price", ascending=False)
    bars1 = ax1.bar(s1["zip5"], s1["median_price"], color=cs.PRIMARY,
                    edgecolor="white", linewidth=0.6)
    ax1.set_title("Median sale price", loc="left")
    ax1.set_ylabel("USD")
    cs.grid_y_only(ax1)
    ax1.set_ylim(0, s1["median_price"].max() * 1.13)
    for b, v in zip(bars1, s1["median_price"]):
        ax1.text(b.get_x() + b.get_width() / 2, v, cs.usd(v),
                 ha="center", va="bottom", fontsize=9.5, color=cs.INK,
                 fontweight="bold")

    # Median $/lot-sqft by ZIP (sorted high -> low)
    ax2 = axes[1]
    s2 = sp.sort_values("median_ppsf", ascending=False)
    bars2 = ax2.bar(s2["zip5"], s2["median_ppsf"], color=cs.POSITIVE,
                    edgecolor="white", linewidth=0.6)
    ax2.set_title("Median land cost  ($ / lot sqft)", loc="left")
    ax2.set_ylabel("$ / sqft")
    cs.grid_y_only(ax2)
    ax2.set_ylim(0, s2["median_ppsf"].max() * 1.13)
    for b, v in zip(bars2, s2["median_ppsf"]):
        if pd.notna(v):
            ax2.text(b.get_x() + b.get_width() / 2, v, f"${v:.0f}",
                     ha="center", va="bottom", fontsize=9.5, color=cs.INK,
                     fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=10.5)

    cs.title_block(
        fig,
        "Shoreline & North Seattle — ZIP Comparison",
        "Higher sale price with lower land cost signals teardown / "
        "rebuild margin",
    )
    cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels")
    fig.subplots_adjust(top=0.82, bottom=0.09, left=0.07, right=0.97, wspace=0.18)
    out = OUTPUTS_DIR / "06_shoreline_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


def render_shoreline_trends(df: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    sub = df[df["zip5"].isin(SHORELINE_ZIPS)].copy()
    fig, ax = plt.subplots(figsize=(13, 6.8))
    palette = sns.color_palette("husl", len(SHORELINE_ZIPS))
    for color, zip_ in zip(palette, SHORELINE_ZIPS):
        zsub = sub[sub["zip5"] == zip_]
        if zsub.empty:
            continue
        monthly = zsub.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        if monthly.empty:
            continue
        ax.plot(monthly.index, monthly.values, marker="o", label=zip_,
                color=color, linewidth=2.2, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8)
    cs.grid_y_only(ax)
    ax.set_ylabel("Median sale price")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: cs.usd(v)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    leg = ax.legend(title="ZIP", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    fontsize=10, ncol=1)
    leg.get_title().set_fontweight("bold")
    cs.title_block(
        fig,
        "Monthly Median Sale Price by ZIP",
        "Shoreline & North Seattle  ·  12-month trend",
    )
    cs.footer(fig, "Thin monthly medians can swing on low volume — read the "
                   "trend, not single points.")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.085, right=0.88)
    out = OUTPUTS_DIR / "07_shoreline_monthly_trends.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.name}")


def find_undervalued_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sales in Shoreline ZIPs that closed >UNDERVALUE_PCT below ZIP median $/sqft."""
    sub = df[
        df["zip5"].isin(SHORELINE_ZIPS)
        & df["price_per_lot_sqft"].notna()
        & (df["LOTSQFT"] >= 4_000)  # rule out condos / sliver lots
    ].copy()

    zip_med = sub.groupby("zip5")["price_per_lot_sqft"].median()
    sub["zip_median_ppsf"] = sub["zip5"].map(zip_med)
    sub["pct_below_median"] = 1 - (sub["price_per_lot_sqft"] / sub["zip_median_ppsf"])
    targets = sub[sub["pct_below_median"] >= UNDERVALUE_PCT].copy()
    targets = targets.sort_values(["zip5", "pct_below_median"], ascending=[True, False])

    cols = [
        "PIN", "ADDR_FULL", "zip5", "SaleDate", "SalePrice",
        "LOTSQFT", "price_per_lot_sqft", "zip_median_ppsf", "pct_below_median",
        "PREUSE_DESC",
    ]
    targets = targets[cols].rename(columns={
        "ADDR_FULL": "address",
        "SaleDate": "sale_date",
        "SalePrice": "sale_price",
        "LOTSQFT": "lot_sqft",
        "PREUSE_DESC": "pre_use",
    })
    return targets


def render_shoreline_snapshot(summary: pd.DataFrame, targets: pd.DataFrame,
                              df: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.35],
                          left=0.05, right=0.97, top=0.80, bottom=0.07,
                          hspace=0.32, wspace=0.16)

    sub = df[df["zip5"].isin(SHORELINE_ZIPS)]
    total = len(sub)

    # Header: title block + KPI strip
    cs.title_block(
        fig,
        "Shoreline & North Seattle — Market & Target Snapshot",
        f"ZIPs {', '.join(SHORELINE_ZIPS)}   ·   window "
        f"{sub['SaleDate'].min():%b %Y} – {sub['SaleDate'].max():%b %Y}",
    )
    kpis = [
        ("Sales analyzed", f"{total:,}"),
        ("Median price", cs.usd(sub["SalePrice"].median())),
        ("Median $/lot-sqft", f"${sub['price_per_lot_sqft'].median():,.0f}"),
        ("Undervalued targets", f"{len(targets):,}"),
    ]
    for i, (label, val) in enumerate(kpis):
        x = 0.05 + i * 0.235
        fig.text(x, 0.875, val, fontsize=20, fontweight="bold", color=cs.PRIMARY)
        fig.text(x, 0.845, label.upper(), fontsize=9, color=cs.MUTED,
                 fontweight="bold")

    # Summary table
    tbl_ax = fig.add_subplot(gs[0, 0])
    tbl_ax.axis("off")
    tbl_ax.set_title("Per-ZIP summary", fontsize=12.5, loc="left", color=cs.INK)
    cell_text = [
        [
            str(row.zip5),
            cs.usd(row.median_price),
            f"${row.median_ppsf:,.0f}" if pd.notna(row.median_ppsf) else "—",
            f"{int(row.median_lot_sqft):,}" if pd.notna(row.median_lot_sqft) else "—",
            f"{int(row.sales_count)}",
            f"{row.pct_sfr*100:.0f}%",
        ]
        for row in summary.itertuples()
    ]
    tbl = tbl_ax.table(
        cellText=cell_text,
        colLabels=["ZIP", "Med price", "$/lot-sf", "Lot sqft", "Sales", "%SFR"],
        loc="upper center", cellLoc="center",
    )
    cs.style_table(tbl, len(cell_text), 6, header_bg=cs.PRIMARY)

    # Top 10 undervalued targets
    tbl2_ax = fig.add_subplot(gs[0, 1])
    tbl2_ax.axis("off")
    tbl2_ax.set_title("Top 10 below-ZIP-median sales", fontsize=12.5, loc="left",
                      color=cs.INK)
    top_targets = targets.sort_values("pct_below_median", ascending=False).head(10)
    cell_text2 = [
        [
            str(row.zip5),
            cs.usd(row.sale_price),
            f"{int(row.lot_sqft):,}",
            f"{row.pct_below_median*100:.0f}%",
        ]
        for row in top_targets.itertuples()
    ]
    tbl2 = tbl2_ax.table(
        cellText=cell_text2,
        colLabels=["ZIP", "Sale price", "Lot sqft", "% below"],
        loc="upper center", cellLoc="center",
    )
    cs.style_table(tbl2, len(cell_text2), 4, header_bg=cs.ACCENT)

    # Monthly trend (full width bottom)
    trend_ax = fig.add_subplot(gs[1, :])
    palette = sns.color_palette("husl", len(SHORELINE_ZIPS))
    for color, zip_ in zip(palette, SHORELINE_ZIPS):
        zsub = sub[sub["zip5"] == zip_]
        if zsub.empty:
            continue
        monthly = zsub.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        if monthly.empty:
            continue
        trend_ax.plot(monthly.index, monthly.values, marker="o", label=zip_,
                      color=color, linewidth=2, markersize=4,
                      markeredgecolor="white", markeredgewidth=0.7)
    trend_ax.set_title("Monthly median sale price by ZIP", fontsize=12.5,
                       loc="left", color=cs.INK)
    trend_ax.set_ylabel("Median price")
    cs.grid_y_only(trend_ax)
    trend_ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: cs.usd(v)))
    trend_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    leg = trend_ax.legend(title="ZIP", loc="upper left", bbox_to_anchor=(1.005, 1),
                          fontsize=9)
    leg.get_title().set_fontweight("bold")

    cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels")
    out_png = OUTPUTS_DIR / "08_shoreline_snapshot.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved {out_png.name}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("KC market analysis — Developer score + Shoreline deep dive")
    print("=" * 72)
    cs.apply()

    try:
        df = load_joined()
    except Exception as exc:
        print(f"[data] FAILED: {exc}")
        traceback.print_exc()
        return 1

    print(f"\n[A] Developer Opportunity Score (min {MIN_SALES_FOR_SCORE} sales/ZIP)")
    try:
        score = compute_dev_score(df)
        score_path = OUTPUTS_DIR / "dev_score_by_zip.csv"
        score.to_csv(score_path, index=False)
        print(f"  Wrote {score_path.name} ({len(score)} ZIPs ranked)")
        print("\n  Top 15 ZIPs by dev score:")
        print(score.head(15)[
            ["zip5", "dev_score", "median_price", "median_ppsf",
             "median_lot_sqft", "pct_sfr", "sales_count"]
        ].to_string(index=False))
        render_dev_score(score)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        traceback.print_exc()

    print(f"\n[B] Shoreline / north-Seattle deep dive: {SHORELINE_ZIPS}")
    try:
        summary = shoreline_summary(df)
        print("\n  Per-ZIP summary:")
        print(summary.to_string(index=False))
        render_shoreline_comparison(summary)
        render_shoreline_trends(df)

        targets = find_undervalued_sales(df)
        targets_path = OUTPUTS_DIR / "shoreline_targets.csv"
        targets.to_csv(targets_path, index=False)
        print(f"\n  {len(targets):,} undervalued sales identified — wrote {targets_path.name}")
        print("\n  Top 15 acquisition targets (largest discount to ZIP median):")
        cols_show = ["zip5", "address", "sale_price", "lot_sqft",
                     "price_per_lot_sqft", "pct_below_median"]
        print(targets.sort_values("pct_below_median", ascending=False)
                     .head(15)[cols_show].to_string(index=False))

        render_shoreline_snapshot(summary, targets, df)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        traceback.print_exc()

    print("\nDone. See outputs_current/ for charts and CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
