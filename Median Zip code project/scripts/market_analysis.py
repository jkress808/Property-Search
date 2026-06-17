"""
King County residential real estate market analysis pipeline.

Pipeline stages:
    1. Download the kc_house_data CSV (fallback URL if needed) into data/
    2. Clean & filter: residential only, drop non-arms-length, last 12 months,
       add price_per_sqft
    3. Aggregate by zip code (median price, median $/sqft, sale count)
    4. Build four charts in outputs/
    5. Write the zip summary CSV and a one-page PNG market snapshot

Each stage is wrapped so a single failure does not abort the entire run.
Progress is printed to stdout so the user can follow along.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

# Use non-interactive backend so figures save cleanly without a display.
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
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "kc_house_data.csv"

# Kaggle requires auth, so a public GitHub mirror is the primary URL.
# The dsrscientist mirror referenced in the original spec is 404 as of 2026-06.
# These four were probed live and returned 200 OK.
PRIMARY_URL = (
    "https://raw.githubusercontent.com/remijul/dataset/master/kc_house_data.csv"
)
FALLBACK_URLS = [
    "https://raw.githubusercontent.com/franciscadias/data/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/NikhilKumarMutyala/Linear-Regression-from-scartch-on-KC-House-Dataset/master/kc_house_data.csv",
]

# Filter thresholds
MIN_PRICE = 10_000  # drop sub-$10k sales as non-arms-length
LOOKBACK_MONTHS = 12  # most-recent-12-months window relative to max sale date

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Stage 1 — download
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download kc_house_data.csv if it isn't already in data/."""
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        print(f"[1/5] Dataset already present at {CSV_PATH} — skipping download.")
        return CSV_PATH

    urls = [PRIMARY_URL, *FALLBACK_URLS]
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"[1/5] Downloading dataset from {url} ...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            CSV_PATH.write_bytes(r.content)
            size_kb = CSV_PATH.stat().st_size / 1024
            print(f"[1/5] Saved {size_kb:,.0f} KB to {CSV_PATH}")
            return CSV_PATH
        except Exception as exc:  # try the next mirror
            print(f"      Failed: {exc}")
            last_err = exc

    raise RuntimeError(f"All download URLs failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Stage 2 — load, clean, filter
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load the CSV, filter to residential arms-length sales in the last 12 months."""
    print("[2/5] Loading CSV into pandas ...")
    df = pd.read_csv(csv_path)
    print(f"      Rows loaded: {len(df):,}")

    # The kc_house_data dataset is entirely residential single/multi-family
    # records — there is no commercial mixed in. If a 'property_type' column
    # ever appears, filter on it; otherwise this is a no-op.
    if "property_type" in df.columns:
        before = len(df)
        df = df[df["property_type"].str.contains("resid", case=False, na=False)]
        print(f"      Residential filter: {before:,} -> {len(df):,}")

    # Drop non-arms-length sales (price = 0 or under $10k)
    before = len(df)
    df = df[df["price"].fillna(0) >= MIN_PRICE]
    print(f"      Price >= ${MIN_PRICE:,}: {before:,} -> {len(df):,}")

    # Parse the date column. kc_house_data uses 'date' like '20141013T000000'.
    df["sale_date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["sale_date"])
    print(f"      Parsed sale dates: dropped {before - len(df):,} unparseable rows")

    # Filter to most-recent-12-months window relative to the max date in data.
    max_date = df["sale_date"].max()
    cutoff = max_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    before = len(df)
    df = df[df["sale_date"] > cutoff]
    print(
        f"      Last {LOOKBACK_MONTHS} months "
        f"({cutoff.date()} to {max_date.date()}): {before:,} -> {len(df):,}"
    )

    # Add price_per_sqft. Guard against zero/missing sqft_living.
    df = df[df["sqft_living"].fillna(0) > 0].copy()
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    print(f"      Added price_per_sqft column. Final row count: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Stage 3 — zip code aggregation
# ---------------------------------------------------------------------------

def summarize_by_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Group by zipcode and compute median price, $/sqft, sale count."""
    print("[3/5] Aggregating by zip code ...")

    agg_spec: dict[str, tuple[str, str]] = {
        "median_price": ("price", "median"),
        "median_price_per_sqft": ("price_per_sqft", "median"),
        "sales_count": ("price", "size"),
    }

    # kc_house_data does not include days-on-market, but if a future dataset
    # adds it, surface the median automatically.
    for candidate in ("days_on_market", "dom", "DaysOnMarket"):
        if candidate in df.columns:
            agg_spec["median_days_on_market"] = (candidate, "median")
            break

    summary = (
        df.groupby("zipcode")
        .agg(**agg_spec)
        .reset_index()
        .sort_values("median_price", ascending=False)
    )

    print("\n      Top 15 zip codes by median sale price:")
    print(summary.head(15).to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Stage 4 — visualizations
# ---------------------------------------------------------------------------

def make_visualizations(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write four charts to outputs/."""
    print("[4/5] Building visualizations ...")

    # 4a. Bar chart — median $/sqft for top 15 zips (by median sale price)
    try:
        top15 = summary.head(15).copy().sort_values("median_price_per_sqft")
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(
            top15["zipcode"].astype(str),
            top15["median_price_per_sqft"],
            color=sns.color_palette("viridis", len(top15)),
        )
        ax.set_xlabel("Median $ / sqft")
        ax.set_ylabel("ZIP code")
        ax.set_title("Median price per sqft — top 15 ZIP codes by median sale price")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "01_median_price_per_sqft_top15.png", dpi=140)
        plt.close(fig)
        print("      Saved 01_median_price_per_sqft_top15.png")
    except Exception as exc:
        print(f"      [warn] bar chart failed: {exc}")

    # 4b. Histogram — overall sale price distribution, log-x
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(df["price"], bins=60, color="#3a7bd5", edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Sale price (log scale, USD)")
        ax.set_ylabel("Number of sales")
        ax.set_title("Sale price distribution — King County (last 12 months)")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "02_price_distribution_log.png", dpi=140)
        plt.close(fig)
        print("      Saved 02_price_distribution_log.png")
    except Exception as exc:
        print(f"      [warn] histogram failed: {exc}")

    # 4c. Scatter — sqft_living vs price, colored by zip (top 10 zips only)
    try:
        top10_zips = summary.head(10)["zipcode"].tolist()
        scatter_df = df[df["zipcode"].isin(top10_zips)].copy()
        scatter_df["zipcode"] = scatter_df["zipcode"].astype(str)
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.scatterplot(
            data=scatter_df,
            x="sqft_living",
            y="price",
            hue="zipcode",
            palette="tab10",
            alpha=0.7,
            s=35,
            ax=ax,
        )
        ax.set_xlabel("Living area (sqft)")
        ax.set_ylabel("Sale price (USD)")
        ax.set_title("Sqft vs sale price — top 10 ZIPs by median price")
        ax.legend(title="ZIP", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "03_sqft_vs_price_top10_zips.png", dpi=140)
        plt.close(fig)
        print("      Saved 03_sqft_vs_price_top10_zips.png")
    except Exception as exc:
        print(f"      [warn] scatter plot failed: {exc}")

    # 4d. Monthly trend line — median sale price across all of King County
    try:
        monthly = (
            df.set_index("sale_date")["price"]
            .resample("MS")
            .median()
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(monthly.index, monthly.values, marker="o", color="#d6336c", linewidth=2)
        ax.set_xlabel("Month")
        ax.set_ylabel("Median sale price (USD)")
        ax.set_title("Median sale price by month — King County")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "04_monthly_median_trend.png", dpi=140)
        plt.close(fig)
        print("      Saved 04_monthly_median_trend.png")
    except Exception as exc:
        print(f"      [warn] monthly trend failed: {exc}")


# ---------------------------------------------------------------------------
# Stage 5 — exports (zip CSV + one-page snapshot)
# ---------------------------------------------------------------------------

def export_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write the zip-summary CSV and the one-page market snapshot PNG."""
    print("[5/5] Writing exports ...")

    # 5a. Zip summary CSV
    try:
        summary_path = OUTPUTS_DIR / "zipcode_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"      Saved {summary_path.name}")
    except Exception as exc:
        print(f"      [warn] zip summary CSV failed: {exc}")

    # 5b. One-page market snapshot — single PNG combining headline stats and
    # the top-15 zip bar chart.
    try:
        total_sales = len(df)
        median_price = df["price"].median()
        median_ppsf = df["price_per_sqft"].median()
        date_min = df["sale_date"].min().date()
        date_max = df["sale_date"].max().date()

        fig = plt.figure(figsize=(11, 8.5))  # letter-ish portrait-friendly
        gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.6, 1.6])

        # Header
        header_ax = fig.add_subplot(gs[0, :])
        header_ax.axis("off")
        header_ax.text(
            0.0, 0.85,
            "King County Residential Market Snapshot",
            fontsize=20, fontweight="bold",
        )
        header_ax.text(
            0.0, 0.45,
            f"Window: {date_min} to {date_max}   |   {total_sales:,} sales analyzed",
            fontsize=12, color="#555",
        )
        header_ax.text(
            0.0, 0.10,
            f"Median price: ${median_price:,.0f}   |   "
            f"Median $/sqft: ${median_ppsf:,.0f}   |   "
            f"ZIPs covered: {summary['zipcode'].nunique()}",
            fontsize=12, color="#222",
        )

        # Top 10 zips by median price — table
        table_ax = fig.add_subplot(gs[1, 0])
        table_ax.axis("off")
        table_ax.set_title("Top 10 ZIPs by median price", fontsize=12, loc="left")
        top10 = summary.head(10).copy()
        cell_text = [
            [
                str(int(row.zipcode)),
                f"${row.median_price:,.0f}",
                f"${row.median_price_per_sqft:,.0f}",
                f"{int(row.sales_count)}",
            ]
            for row in top10.itertuples()
        ]
        tbl = table_ax.table(
            cellText=cell_text,
            colLabels=["ZIP", "Med. price", "Med. $/sqft", "Sales"],
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

        # Top 15 zips $/sqft chart
        chart_ax = fig.add_subplot(gs[1, 1])
        top15 = summary.head(15).copy().sort_values("median_price_per_sqft")
        chart_ax.barh(
            top15["zipcode"].astype(str),
            top15["median_price_per_sqft"],
            color=sns.color_palette("viridis", len(top15)),
        )
        chart_ax.set_title("Top 15 ZIPs — median $/sqft", fontsize=12, loc="left")
        chart_ax.set_xlabel("$ / sqft")
        chart_ax.tick_params(axis="y", labelsize=8)

        # Monthly trend
        trend_ax = fig.add_subplot(gs[2, :])
        monthly = (
            df.set_index("sale_date")["price"]
            .resample("MS").median().dropna()
        )
        trend_ax.plot(monthly.index, monthly.values, marker="o",
                      color="#d6336c", linewidth=2)
        trend_ax.set_title("Monthly median sale price — King County", fontsize=12, loc="left")
        trend_ax.set_ylabel("Median price (USD)")
        trend_ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        fig.tight_layout()
        snap_png = OUTPUTS_DIR / "market_snapshot.png"
        fig.savefig(snap_png, dpi=150)

        # Save the same figure to PDF for a clean one-pager.
        snap_pdf = OUTPUTS_DIR / "market_snapshot.pdf"
        fig.savefig(snap_pdf)
        plt.close(fig)
        print(f"      Saved {snap_png.name} and {snap_pdf.name}")
    except Exception as exc:
        print(f"      [warn] market snapshot failed: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("King County residential market analysis")
    print("=" * 70)

    # Stage 1 — download (hard fail if we can't get data)
    try:
        csv_path = download_dataset()
    except Exception as exc:
        print(f"[1/5] FAILED: {exc}")
        return 1

    # Stage 2 — load & clean (also hard fail; everything else depends on df)
    try:
        df = load_and_clean(csv_path)
    except Exception as exc:
        print(f"[2/5] FAILED: {exc}")
        traceback.print_exc()
        return 2

    # Stage 3 — zip summary (hard fail if grouping breaks)
    try:
        summary = summarize_by_zip(df)
    except Exception as exc:
        print(f"[3/5] FAILED: {exc}")
        traceback.print_exc()
        return 3

    # Stages 4 & 5 — non-critical: continue past any single chart failing
    try:
        make_visualizations(df, summary)
    except Exception as exc:
        print(f"[4/5] FAILED at top level: {exc}")
        traceback.print_exc()

    try:
        export_outputs(df, summary)
    except Exception as exc:
        print(f"[5/5] FAILED at top level: {exc}")
        traceback.print_exc()

    print("\nDone. See outputs/ for charts, zipcode_summary.csv, and the snapshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
