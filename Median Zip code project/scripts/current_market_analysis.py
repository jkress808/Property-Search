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
import pandas as pd
import requests
import seaborn as sns

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
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(
            top15["zipcode"].astype(str),
            top15["median_price_per_lot_sqft"],
            color=sns.color_palette("viridis", len(top15)),
        )
        ax.set_xlabel("Median $ / lot sqft")
        ax.set_ylabel("ZIP code")
        ax.set_title("Median price per lot sqft — top 15 ZIPs (by median sale price)")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "01_median_price_per_lot_sqft_top15.png", dpi=140)
        plt.close(fig)
        print("      Saved 01_median_price_per_lot_sqft_top15.png")
    except Exception as exc:
        print(f"      [warn] bar chart failed: {exc}")

    # 4b. Log-x histogram of sale price
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(df["SalePrice"], bins=60, color="#3a7bd5", edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Sale price (log scale, USD)")
        ax.set_ylabel("Number of sales")
        ax.set_title("Sale price distribution — King County (last 12 months, live data)")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "02_price_distribution_log.png", dpi=140)
        plt.close(fig)
        print("      Saved 02_price_distribution_log.png")
    except Exception as exc:
        print(f"      [warn] histogram failed: {exc}")

    # 4c. Scatter — lot sqft vs price, top 10 zips
    try:
        top10_zips = summary.head(10)["zipcode"].tolist()
        scatter_df = df[df["zip5"].isin(top10_zips) & df["LOTSQFT"].between(1, 100_000)].copy()
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.scatterplot(
            data=scatter_df,
            x="LOTSQFT", y="SalePrice",
            hue="zip5", palette="tab10",
            alpha=0.7, s=35, ax=ax,
        )
        ax.set_xlabel("Lot size (sqft)")
        ax.set_ylabel("Sale price (USD)")
        ax.set_title("Lot size vs sale price — top 10 ZIPs by median price")
        ax.legend(title="ZIP", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "03_lotsqft_vs_price_top10_zips.png", dpi=140)
        plt.close(fig)
        print("      Saved 03_lotsqft_vs_price_top10_zips.png")
    except Exception as exc:
        print(f"      [warn] scatter failed: {exc}")

    # 4d. Monthly trend line
    try:
        monthly = (
            df.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(monthly.index, monthly.values, marker="o", color="#d6336c", linewidth=2)
        ax.set_xlabel("Month")
        ax.set_ylabel("Median sale price (USD)")
        ax.set_title("Median sale price by month — King County (live data)")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "04_monthly_median_trend.png", dpi=140)
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
        total = len(df)
        med_price = df["SalePrice"].median()
        med_ppsf = df["price_per_lot_sqft"].median()
        date_min = df["SaleDate"].min().date()
        date_max = df["SaleDate"].max().date()

        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.6, 1.6])

        header_ax = fig.add_subplot(gs[0, :])
        header_ax.axis("off")
        header_ax.text(0.0, 0.85,
            "King County Residential Market Snapshot — LIVE",
            fontsize=20, fontweight="bold")
        header_ax.text(0.0, 0.45,
            f"Window: {date_min} to {date_max}   |   "
            f"{total:,} sales analyzed   |   Source: KingCo GIS",
            fontsize=12, color="#555")
        header_ax.text(0.0, 0.10,
            f"Median price: ${med_price:,.0f}   |   "
            f"Median $/lot-sqft: ${med_ppsf:,.0f}   |   "
            f"ZIPs covered: {summary['zipcode'].nunique()}",
            fontsize=12, color="#222")

        # Top 10 zips table
        tbl_ax = fig.add_subplot(gs[1, 0])
        tbl_ax.axis("off")
        tbl_ax.set_title("Top 10 ZIPs by median price", fontsize=12, loc="left")
        top10 = summary.head(10).copy()
        cell_text = [
            [
                str(row.zipcode),
                f"${row.median_price:,.0f}",
                (f"${row.median_price_per_lot_sqft:,.0f}"
                 if pd.notna(row.median_price_per_lot_sqft) else "—"),
                f"{int(row.sales_count)}",
            ]
            for row in top10.itertuples()
        ]
        tbl = tbl_ax.table(
            cellText=cell_text,
            colLabels=["ZIP", "Med. price", "Med. $/lot-sqft", "Sales"],
            loc="upper center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

        # Top 15 zips chart
        chart_ax = fig.add_subplot(gs[1, 1])
        top15 = (
            summary.head(15).copy()
            .dropna(subset=["median_price_per_lot_sqft"])
            .sort_values("median_price_per_lot_sqft")
        )
        chart_ax.barh(
            top15["zipcode"].astype(str),
            top15["median_price_per_lot_sqft"],
            color=sns.color_palette("viridis", len(top15)),
        )
        chart_ax.set_title("Top 15 ZIPs — median $/lot-sqft", fontsize=12, loc="left")
        chart_ax.set_xlabel("$ / lot sqft")
        chart_ax.tick_params(axis="y", labelsize=8)

        # Monthly trend
        trend_ax = fig.add_subplot(gs[2, :])
        monthly = (
            df.set_index("SaleDate")["SalePrice"].resample("MS").median().dropna()
        )
        trend_ax.plot(monthly.index, monthly.values, marker="o", color="#d6336c", linewidth=2)
        trend_ax.set_title("Monthly median sale price — King County", fontsize=12, loc="left")
        trend_ax.set_ylabel("Median price (USD)")
        trend_ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "market_snapshot.png", dpi=150)
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
