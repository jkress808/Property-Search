"""
Current King County residential market analysis pipeline.

Sibling to market_analysis.py — same shape of output, but the data is pulled
live from the King County GIS REST API (KingCo_PropertyInfo MapServer) rather
than the static 2014-2015 Kaggle file. The Sales layer carries the most recent
3 years of sales; this script filters to the most recent 12 months relative to
the max SaleDate in the live data.

Two data sources are joined on PIN:
    - Layer 3 (Sales)   : PIN, SaleDate, SalePrice, Principal_Use, address
    - Layer 2 (Parcels) : PIN, ZIP5, LOTSQFT, PREUSE_DESC, PROPTYPE, CTYNAME

The Parcel layer does NOT expose living-square-footage, only lot square
footage, so the price-per-sqft metric here is $ / lot sqft (clearly labeled
in chart titles and column names). The Kaggle pipeline used $ / sqft_living.
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
import matplotlib.ticker as mticker
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

GIS_BASE = (
    "https://gismaps.kingcounty.gov/arcgis/rest/services/Property/"
    "KingCo_PropertyInfo/MapServer"
)
SALES_LAYER = f"{GIS_BASE}/3/query"
PARCEL_LAYER = f"{GIS_BASE}/2/query"

# Filters
MIN_PRICE = 10_000          # drop non-arms-length sales
LOOKBACK_MONTHS = 12        # 12-month window relative to max SaleDate in data
SALES_PAGE_SIZE = 1000      # ArcGIS default cap
PARCEL_PIN_BATCH = 400      # safe size for `PIN IN (...)` queries
REQUEST_TIMEOUT = 60

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Stage 1 — fetch recent sales from GIS Sales layer
# ---------------------------------------------------------------------------

def fetch_recent_sales() -> pd.DataFrame:
    """Paginate the Sales layer for residential, arms-length, last-12-mo sales.

    The Sales layer holds up to 3 years; we over-fetch all residential sales
    with price >= MIN_PRICE in the last 14 months (a small buffer over
    LOOKBACK_MONTHS) and trim in pandas after we know the true max date.
    """
    # Use a generous 14-month server-side filter so we never miss recent sales;
    # the exact 12-month trim happens in pandas based on max(SaleDate).
    # ArcGIS Date fields require literal `DATE 'YYYY-MM-DD HH:MM:SS'`.
    cutoff_dt = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=14)
    cutoff_lit = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

    where = (
        f"Principal_Use = 'RESIDENTIAL' "
        f"AND SalePrice >= {MIN_PRICE} "
        f"AND SaleDate > DATE '{cutoff_lit}'"
    )

    print(f"[1/5] Fetching residential sales from King County GIS ...")
    print(f"      WHERE {where}")

    all_features: list[dict] = []
    offset = 0
    page = 0
    while True:
        params = {
            "where": where,
            "outFields": "PIN,SaleDate,SalePrice,Principal_Use,Property_Type,address",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": SALES_PAGE_SIZE,
            "orderByFields": "OBJECTID ASC",  # stable order for offset pagination
            "f": "json",
        }
        # POST avoids URL-length and encoding issues with the DATE literal.
        r = requests.post(SALES_LAYER, data=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"Sales API error: {body['error']}")

        feats = body.get("features", [])
        all_features.extend(feats)
        page += 1
        print(f"      page {page:>3}  +{len(feats):>4} rows  (running total {len(all_features):,})")

        if not body.get("exceededTransferLimit") and len(feats) < SALES_PAGE_SIZE:
            break
        offset += SALES_PAGE_SIZE
        time.sleep(0.1)  # polite throttle

    if not all_features:
        raise RuntimeError("Sales API returned zero rows.")

    df = pd.DataFrame([f["attributes"] for f in all_features])
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], unit="ms", errors="coerce")
    df = df.dropna(subset=["SaleDate", "PIN", "SalePrice"])

    # Trim to true 12-month window relative to max date in this snapshot
    max_date = df["SaleDate"].max()
    cutoff = max_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    before = len(df)
    df = df[df["SaleDate"] > cutoff]
    print(
        f"      Trimmed to last {LOOKBACK_MONTHS} months "
        f"({cutoff.date()} -> {max_date.date()}): {before:,} -> {len(df):,}"
    )

    # Snapshot the raw pull to data/ for traceability
    snapshot = DATA_DIR / "kc_gis_sales_recent.csv"
    df.to_csv(snapshot, index=False)
    print(f"      Snapshot saved: {snapshot}")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — fetch parcel attributes (ZIP, LOTSQFT, PreUse) and join
# ---------------------------------------------------------------------------

def fetch_parcel_attrs(pins: list[str]) -> pd.DataFrame:
    """Batch-query Parcel layer for the PINs we care about."""
    print(f"[2/5] Fetching parcel attributes for {len(pins):,} unique PINs ...")
    rows: list[dict] = []
    batches = math.ceil(len(pins) / PARCEL_PIN_BATCH)
    for i in range(batches):
        chunk = pins[i * PARCEL_PIN_BATCH:(i + 1) * PARCEL_PIN_BATCH]
        # ArcGIS expects single-quoted strings in IN clause
        in_list = ",".join(f"'{p}'" for p in chunk)
        params = {
            "where": f"PIN IN ({in_list})",
            "outFields": "PIN,ZIP5,LOTSQFT,PREUSE_DESC,PROPTYPE,CTYNAME,POSTALCTYNAME,ADDR_FULL",
            "returnGeometry": "false",
            "f": "json",
        }
        # PIN IN (...) request body can be long — use POST to avoid URL limits
        r = requests.post(PARCEL_LAYER, data=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"Parcel API error on batch {i+1}: {body['error']}")
        feats = body.get("features", [])
        rows.extend(f["attributes"] for f in feats)
        print(f"      batch {i+1:>3}/{batches}  +{len(feats):>4} parcels  (running total {len(rows):,})")
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("Parcel API returned zero rows.")
    return pd.DataFrame(rows)


def clean_and_join(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Join sales to parcels, add price_per_lot_sqft, filter to single-family-ish."""
    unique_pins = sales_df["PIN"].dropna().astype(str).unique().tolist()
    parcels_df = fetch_parcel_attrs(unique_pins)

    print(f"      Joining {len(sales_df):,} sales to {len(parcels_df):,} parcels on PIN ...")
    merged = sales_df.merge(parcels_df, on="PIN", how="inner")
    print(f"      Joined rows: {len(merged):,}")

    # Use postal ZIP if ZIP5 missing
    merged["zip5"] = merged["ZIP5"].astype(str).str.strip().str.zfill(5)
    merged = merged[merged["zip5"].str.match(r"^\d{5}$")]

    # Compute $ / lot sqft where possible
    merged["LOTSQFT"] = pd.to_numeric(merged["LOTSQFT"], errors="coerce")
    merged["price_per_lot_sqft"] = merged.apply(
        lambda r: (r["SalePrice"] / r["LOTSQFT"]) if r["LOTSQFT"] and r["LOTSQFT"] > 0 else None,
        axis=1,
    )
    print(f"      Rows with usable LOTSQFT: "
          f"{merged['price_per_lot_sqft'].notna().sum():,} / {len(merged):,}")
    return merged


# ---------------------------------------------------------------------------
# Stage 3 — zip summary
# ---------------------------------------------------------------------------

def summarize_by_zip(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Aggregating by zip code ...")
    summary = (
        df.groupby("zip5")
          .agg(
              median_price=("SalePrice", "median"),
              median_price_per_lot_sqft=("price_per_lot_sqft", "median"),
              sales_count=("SalePrice", "size"),
              median_lot_sqft=("LOTSQFT", "median"),
          )
          .reset_index()
          .rename(columns={"zip5": "zipcode"})
          .sort_values("median_price", ascending=False)
    )
    print("\n      Top 15 zip codes by median sale price:")
    print(summary.head(15).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Stage 4 — visualizations
# ---------------------------------------------------------------------------

def make_visualizations(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("[4/5] Building visualizations ...")

    # 4a. Top 15 zips by median $/lot-sqft
    try:
        top15 = (
            summary.head(15).copy()
            .dropna(subset=["median_price_per_lot_sqft"])
            .sort_values("median_price_per_lot_sqft")
        )
        zips = top15["zipcode"].astype(str).tolist()
        vals = top15["median_price_per_lot_sqft"].tolist()
        fig, ax = plt.subplots(figsize=(12, 8))
        norm = plt.Normalize(min(vals), max(vals))
        bars = ax.barh(zips, vals, color=[cs.VALUE_CMAP(norm(v)) for v in vals],
                       edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_width() + max(vals) * 0.012,
                    b.get_y() + b.get_height() / 2, f"${v:.0f}",
                    va="center", ha="left", fontsize=9.5, color=cs.MUTED)
        ax.set_xlabel("Median $ / lot sqft")
        ax.set_ylabel("")
        ax.set_xlim(0, max(vals) * 1.12)
        cs.grid_x_only(ax)
        cs.title_block(
            fig,
            "Land Cost by ZIP — Top 15 (highest-priced ZIPs)",
            "Median sale price per lot square foot",
        )
        cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels")
        fig.subplots_adjust(top=0.86, left=0.08, right=0.96, bottom=0.12)
        fig.savefig(OUTPUTS_DIR / "01_median_price_per_lot_sqft_top15.png")
        plt.close(fig)
        print("      Saved 01_median_price_per_lot_sqft_top15.png")
    except Exception as exc:
        print(f"      [warn] bar chart failed: {exc}")

    # 4b. Log-x histogram of sale price
    try:
        import numpy as np
        prices = df["SalePrice"][df["SalePrice"] > 0]
        bins = np.logspace(np.log10(prices.min()), np.log10(prices.max()), 45)
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.hist(prices, bins=bins, color=cs.PRIMARY, edgecolor="white")
        ax.set_xscale("log")
        med = df["SalePrice"].median()
        ax.axvline(med, color=cs.ACCENT, linewidth=2, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {cs.usd(med)}",
                color=cs.ACCENT, fontweight="bold", va="top", fontsize=10.5)
        ax.set_xlabel("Sale price (log scale)")
        ax.set_ylabel("Number of sales")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: cs.usd(v)))
        cs.grid_y_only(ax)
        cs.title_block(
            fig,
            "Sale Price Distribution — King County",
            "Residential arms-length sales, last 12 months",
        )
        cs.footer(fig, "Source: King County Assessor sales extract")
        fig.subplots_adjust(top=0.85, left=0.08, right=0.96, bottom=0.11)
        fig.savefig(OUTPUTS_DIR / "02_price_distribution_log.png")
        plt.close(fig)
        print("      Saved 02_price_distribution_log.png")
    except Exception as exc:
        print(f"      [warn] histogram failed: {exc}")

    # 4c. Scatter — lot sqft vs price, top 10 zips
    try:
        top10_zips = summary.head(10)["zipcode"].tolist()
        scatter_df = df[df["zip5"].isin(top10_zips) & df["LOTSQFT"].between(1, 100_000)].copy()
        fig, ax = plt.subplots(figsize=(12, 7.5))
        sns.scatterplot(
            data=scatter_df,
            x="LOTSQFT", y="SalePrice",
            hue="zip5", palette="tab10",
            alpha=0.65, s=38, ax=ax, edgecolor="white", linewidth=0.3,
        )
        ax.set_xlabel("Lot size (sqft)")
        ax.set_ylabel("Sale price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: cs.usd(v)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
        cs.grid_y_only(ax)
        leg = ax.legend(title="ZIP", bbox_to_anchor=(1.01, 1), loc="upper left")
        leg.get_title().set_fontweight("bold")
        cs.title_block(
            fig,
            "Lot Size vs Sale Price",
            "Top 10 ZIPs by median price  ·  last 12 months",
        )
        cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels")
        fig.subplots_adjust(top=0.85, left=0.09, right=0.86, bottom=0.10)
        fig.savefig(OUTPUTS_DIR / "03_lotsqft_vs_price_top10_zips.png")
        plt.close(fig)
        print("      Saved 03_lotsqft_vs_price_top10_zips.png")
    except Exception as exc:
        print(f"      [warn] scatter failed: {exc}")

    # 4d. Monthly trend line
    try:
        import matplotlib.dates as mdates
        monthly = (
            df.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        )
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.plot(monthly.index, monthly.values, marker="o", color=cs.PRIMARY,
                linewidth=2.4, markersize=6, markeredgecolor="white",
                markeredgewidth=0.9)
        ax.fill_between(monthly.index, monthly.values, alpha=0.08, color=cs.PRIMARY)
        ax.set_xlabel("")
        ax.set_ylabel("Median sale price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: cs.usd(v)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        cs.grid_y_only(ax)
        cs.title_block(
            fig,
            "Median Sale Price by Month — King County",
            "Residential arms-length sales  ·  last 12 months",
        )
        cs.footer(fig, "Source: King County Assessor sales extract")
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.subplots_adjust(top=0.85, left=0.09, right=0.96, bottom=0.13)
        fig.savefig(OUTPUTS_DIR / "04_monthly_median_trend.png")
        plt.close(fig)
        print("      Saved 04_monthly_median_trend.png")
    except Exception as exc:
        print(f"      [warn] monthly trend failed: {exc}")


# ---------------------------------------------------------------------------
# Stage 5 — exports
# ---------------------------------------------------------------------------

def export_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("[5/5] Writing exports ...")

    try:
        path = OUTPUTS_DIR / "zipcode_summary.csv"
        summary.to_csv(path, index=False)
        print(f"      Saved {path.name}")
    except Exception as exc:
        print(f"      [warn] zip summary CSV failed: {exc}")

    try:
        import matplotlib.dates as mdates

        total = len(df)
        med_price = df["SalePrice"].median()
        med_ppsf = df["price_per_lot_sqft"].median()
        date_min = df["SaleDate"].min()
        date_max = df["SaleDate"].max()

        fig = plt.figure(figsize=(13, 9.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.3],
                              left=0.05, right=0.97, top=0.80, bottom=0.08,
                              hspace=0.32, wspace=0.16)

        # Header: title block + KPI strip
        cs.title_block(
            fig,
            "King County Residential Market Snapshot",
            f"Arms-length residential sales  ·  window "
            f"{date_min:%b %Y} – {date_max:%b %Y}",
        )
        kpis = [
            ("Sales analyzed", f"{total:,}"),
            ("Median price", cs.usd(med_price)),
            ("Median $/lot-sqft", f"${med_ppsf:,.0f}"),
            ("ZIPs covered", f"{summary['zipcode'].nunique()}"),
        ]
        for i, (label, val) in enumerate(kpis):
            x = 0.05 + i * 0.235
            fig.text(x, 0.875, val, fontsize=20, fontweight="bold", color=cs.PRIMARY)
            fig.text(x, 0.845, label.upper(), fontsize=9, color=cs.MUTED,
                     fontweight="bold")

        # Top 10 zips table
        tbl_ax = fig.add_subplot(gs[0, 0])
        tbl_ax.axis("off")
        tbl_ax.set_title("Top 10 ZIPs by median price", fontsize=12.5, loc="left",
                         color=cs.INK)
        top10 = summary.head(10).copy()
        cell_text = [
            [
                str(row.zipcode),
                cs.usd(row.median_price),
                (f"${row.median_price_per_lot_sqft:,.0f}"
                 if pd.notna(row.median_price_per_lot_sqft) else "—"),
                f"{int(row.sales_count)}",
            ]
            for row in top10.itertuples()
        ]
        tbl = tbl_ax.table(
            cellText=cell_text,
            colLabels=["ZIP", "Med price", "$/lot-sqft", "Sales"],
            loc="upper center", cellLoc="center",
        )
        cs.style_table(tbl, len(cell_text), 4, header_bg=cs.PRIMARY)

        # Top 15 zips chart
        chart_ax = fig.add_subplot(gs[0, 1])
        top15 = (
            summary.head(15).copy()
            .dropna(subset=["median_price_per_lot_sqft"])
            .sort_values("median_price_per_lot_sqft")
        )
        vals = top15["median_price_per_lot_sqft"].tolist()
        norm = plt.Normalize(min(vals), max(vals))
        chart_ax.barh(top15["zipcode"].astype(str), vals,
                      color=[cs.VALUE_CMAP(norm(v)) for v in vals],
                      edgecolor="white", linewidth=0.5)
        chart_ax.set_title("Top 15 ZIPs — median $/lot-sqft", fontsize=12.5,
                           loc="left", color=cs.INK)
        chart_ax.set_xlabel("$ / lot sqft")
        cs.grid_x_only(chart_ax)
        chart_ax.tick_params(axis="y", labelsize=9)

        # Monthly trend
        trend_ax = fig.add_subplot(gs[1, :])
        monthly = (
            df.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        )
        trend_ax.plot(monthly.index, monthly.values, marker="o", color=cs.PRIMARY,
                      linewidth=2.4, markersize=5, markeredgecolor="white",
                      markeredgewidth=0.8)
        trend_ax.fill_between(monthly.index, monthly.values, alpha=0.08,
                              color=cs.PRIMARY)
        trend_ax.set_title("Monthly median sale price — King County",
                           fontsize=12.5, loc="left", color=cs.INK)
        trend_ax.set_ylabel("Median price")
        cs.grid_y_only(trend_ax)
        trend_ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: cs.usd(v)))
        trend_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        cs.footer(fig, "Source: King County Assessor sales extract + GIS parcels")
        fig.savefig(OUTPUTS_DIR / "market_snapshot.png")
        fig.savefig(OUTPUTS_DIR / "market_snapshot.pdf")
        plt.close(fig)
        print("      Saved market_snapshot.png and market_snapshot.pdf")
    except Exception as exc:
        print(f"      [warn] market snapshot failed: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

JOINED_CACHE = DATA_DIR / "kc_market_joined.csv"


def load_joined_cache() -> pd.DataFrame | None:
    """Use the joined cache (built from bulk extracts) if it's fresh (<7 days).

    The bulk-extract path carries current 2026 sales; the live GIS Sales layer
    lags by months. When the cache exists we prefer it so all scripts run on the
    same fresh data. Falls back to the live GIS pull when no fresh cache exists.
    """
    if not JOINED_CACHE.exists():
        return None
    age_h = (time.time() - JOINED_CACHE.stat().st_mtime) / 3600
    if age_h >= 168:
        return None
    print(f"[data] Using joined cache {JOINED_CACHE.name} (age {age_h:.1f}h)")
    df = pd.read_csv(JOINED_CACHE, dtype={"PIN": str, "zip5": str})
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    df["LOTSQFT"] = pd.to_numeric(df["LOTSQFT"], errors="coerce")
    df["price_per_lot_sqft"] = pd.to_numeric(df["price_per_lot_sqft"], errors="coerce")
    print(f"       {len(df):,} rows loaded from cache")
    return df


def main() -> int:
    print("=" * 70)
    print("King County current-market analysis")
    print("=" * 70)
    cs.apply()

    df = load_joined_cache()
    if df is None:
        print("[data] No fresh cache — pulling live from King County GIS")
        try:
            sales = fetch_recent_sales()
        except Exception as exc:
            print(f"[1/5] FAILED: {exc}")
            traceback.print_exc()
            return 1

        try:
            df = clean_and_join(sales)
        except Exception as exc:
            print(f"[2/5] FAILED: {exc}")
            traceback.print_exc()
            return 2

    try:
        summary = summarize_by_zip(df)
    except Exception as exc:
        print(f"[3/5] FAILED: {exc}")
        traceback.print_exc()
        return 3

    try:
        make_visualizations(df, summary)
    except Exception as exc:
        print(f"[4/5] FAILED: {exc}")
        traceback.print_exc()

    try:
        export_outputs(df, summary)
    except Exception as exc:
        print(f"[5/5] FAILED: {exc}")
        traceback.print_exc()

    print("\nDone. See outputs_current/ for charts, zipcode_summary.csv, and the snapshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
