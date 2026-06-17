"""
Parcel-level redevelopment pro-forma — a ranked acquisition list.

Where the Developer Opportunity Score ranks ZIPs by proxies, this ranks
individual parcels by estimated development MARGIN:

    margin = GDV − (acquisition + demolition + construction + soft + selling)
    GDV    = buildable units × finished-unit size × new-construction $/living-sqft

Each candidate is a real recent residential sale, so the sale price is a
realistic acquisition cost and the parcel's lot size / zoning / assessed split
are known.

Data (all from the King County Assessor bulk extracts in data/kc_extract/ plus
the joined market cache):
  - kc_market_joined.csv  : PIN, zip5, SalePrice, LOTSQFT, ZONING, JURIS, dates
  - EXTR_RPAcct_NoName    : ApprLandVal / ApprImpsVal  -> teardown signal
  - EXTR_Parcel           : Unbuildable / hazards / NbrBldgSites -> feasibility
  - EXTR_ResBldg          : YrBuilt / SqFtTotLiving -> age + new-build comps

Three feasibility layers (your "is it even possible to develop"):
  1. units buildable from zoning + lot size (HB 1110 middle housing credited)
  2. real teardown signal (improvement-to-value ratio + building age)
  3. physical gating (rural, Unbuildable, steep slope / wetland / flood, etc.)

ALL ECONOMICS ARE ESTIMATES driven by the assumptions block below — tune them.
"""

from __future__ import annotations

import csv
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: F401  (kept for parity with style)
import matplotlib.pyplot as plt
import pandas as pd

import chart_style as cs
from dev_score_and_shoreline import (
    LUX_LOWDENSITY, SHORELINE_ZIPS, classify_zoning,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_DIR = DATA_DIR / "kc_extract"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_current"
JOINED_CACHE = DATA_DIR / "kc_market_joined.csv"
ENC = "latin-1"
csv.field_size_limit(10_000_000)

# ---------------------------------------------------------------------------
# Pro-forma assumptions  (edit these — they drive every dollar figure)
# ---------------------------------------------------------------------------
BUILD_COST_PER_SQFT = 300      # hard construction $/finished sqft (Seattle infill)
SOFT_COST_PCT = 0.25           # design + permits + fees + financing, as % of hard
DEMO_COST = 35_000             # demolition of the existing structure (per teardown)
SELLING_COST_PCT = 0.06        # commissions + closing on GDV
HB1110_BASE_UNITS = 4          # city residential baseline (transit could push to 6)
MIN_LOT_PER_UNIT = 1_000       # physical floor: sqft of lot required per unit
UNITS_CAP = 60                 # sanity cap for very large multifamily lots
NEW_BUILD_YEARS = 5            # a sale is a "new-construction comp" if built <= N yrs prior
TEARDOWN_MAX_IMPS_RATIO = 0.55 # improvement <= 55% of value  => teardown candidate
TEARDOWN_MIN_AGE = 35          # OR existing building older than this (years)

# Finished-unit size by product type (sqft)
UNIT_SQFT = {"sfr": 2600, "townhome": 1700, "multifamily": 1100}

# Lowrise/midrise minimum lot area per unit (sqft) — rough Seattle equivalents
MF_LOT_PER_UNIT = {"LR3": 800, "LR2": 1100, "LR1": 1600, "MR": 500, "HR": 250}

# Physical / regulatory constraints that make a parcel hard or impossible to build
HAZARD_COLS = ["SteepSlopeHazard", "Wetland", "Stream", "HundredYrFloodPlain",
               "LandslideHazard", "ErosionHazard", "CoalMineHazard"]


# ---------------------------------------------------------------------------
# Enrichment loaders (only for the PINs present in the market cache)
# ---------------------------------------------------------------------------
def _pin(mj: str, mn: str) -> str:
    return f"{mj.strip().zfill(6)}{mn.strip().zfill(4)}"


def load_account(pins: set[str]) -> dict[str, tuple[float, float]]:
    """PIN -> (land value, improvement value)."""
    out: dict[str, tuple[float, float]] = {}
    with (EXTRACT_DIR / "EXTR_RPAcct_NoName.csv").open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            p = _pin(row["Major"], row["Minor"])
            if p not in pins:
                continue
            try:
                out[p] = (float(row["ApprLandVal"] or 0), float(row["ApprImpsVal"] or 0))
            except ValueError:
                pass
    return out


def load_parcel_constraints(pins: set[str]) -> dict[str, dict]:
    """PIN -> feasibility constraint fields."""
    out: dict[str, dict] = {}
    with (EXTRACT_DIR / "EXTR_Parcel.csv").open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            p = _pin(row["Major"], row["Minor"])
            if p not in pins:
                continue
            hazard = any(row.get(c, "N").strip().upper() == "Y" for c in HAZARD_COLS)
            try:
                pct_unusable = float(row.get("PcntUnusable", 0) or 0)
            except ValueError:
                pct_unusable = 0.0
            try:
                nbr_sites = int(row.get("NbrBldgSites", 0) or 0)
            except ValueError:
                nbr_sites = 0
            out[p] = {
                "unbuildable": row.get("Unbuildable", "False").strip() == "True",
                "pct_unusable": pct_unusable,
                "hazard": hazard,
                "odd_shape": row.get("RestrictiveSzShape", "0").strip() == "1",
                "nbr_bldg_sites": nbr_sites,
            }
    return out


def load_resbldg(pins: set[str]) -> dict[str, dict]:
    """PIN -> building summary (oldest/newest year, total living sqft)."""
    out: dict[str, dict] = {}
    with (EXTRACT_DIR / "EXTR_ResBldg.csv").open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            p = _pin(row["Major"], row["Minor"])
            if p not in pins:
                continue
            try:
                yr = int(row["YrBuilt"] or 0)
            except ValueError:
                yr = 0
            try:
                living = float(row["SqFtTotLiving"] or 0)
            except ValueError:
                living = 0.0
            rec = out.setdefault(p, {"yr_old": 9999, "yr_new": 0, "living": 0.0})
            if 1800 < yr < rec["yr_old"]:
                rec["yr_old"] = yr
            if yr > rec["yr_new"]:
                rec["yr_new"] = yr
            rec["living"] += living
    return out


# ---------------------------------------------------------------------------
# Buildable-unit model
# ---------------------------------------------------------------------------
def estimate_units(zoning: str, juris: str, lot_sqft: float) -> tuple[int, str]:
    """(units, product_type) buildable, from zoning + jurisdiction + lot size."""
    tier, _ = classify_zoning(zoning, juris)
    if tier in ("rural", "unknown") or not lot_sqft or lot_sqft <= 0:
        return 0, "none"
    acres = lot_sqft / 43_560
    base = re.sub(r"\(.*?\)", "", str(zoning).upper()).replace(" ", "").strip()
    j = (juris or "").upper().strip()
    phys_cap = max(1, int(lot_sqft // MIN_LOT_PER_UNIT))

    if tier == "multifamily":
        for k, per_unit in MF_LOT_PER_UNIT.items():
            if base.startswith(k):
                u = max(1, int(lot_sqft // per_unit))
                return min(u, UNITS_CAP), ("townhome" if k.startswith("LR") else "multifamily")
        m = re.match(r"^(?:SR|R)-?(\d+(?:\.\d+)?)", base)   # genuine units/acre (12..48)
        if m and "." not in m.group(1) and 12 <= float(m.group(1)) < 100:
            u = max(1, int(round(acres * float(m.group(1)))))
            return min(u, UNITS_CAP), "multifamily"
        u = max(2, int(lot_sqft // 1200))              # generic mid-density fallback
        return min(u, UNITS_CAP), "multifamily"

    if tier == "sfr_mid":                              # city SFR — HB 1110 middle housing
        u = min(HB1110_BASE_UNITS, phys_cap)
        return u, ("townhome" if u > 1 else "sfr")

    # tier == "sfr_low": luxury (1 rebuild) or unincorporated units/acre
    if j in LUX_LOWDENSITY:
        return 1, "sfr"
    m = re.match(r"^R-?(\d+(?:\.\d+)?)$", base)         # full-match unincorporated R-n
    if m and j == "KING COUNTY":
        raw = m.group(1)
        if "." not in raw and float(raw) < 100:        # units/acre (not a min-lot code)
            u = min(max(1, int(round(acres * float(raw)))), phys_cap, UNITS_CAP)
            return u, ("townhome" if u > 1 else "sfr")
    return 1, "sfr"


# ---------------------------------------------------------------------------
# Build candidate table + pro-forma
# ---------------------------------------------------------------------------
def build_candidates() -> pd.DataFrame:
    df = pd.read_csv(JOINED_CACHE, dtype={"PIN": str, "zip5": str})
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    df["LOTSQFT"] = pd.to_numeric(df["LOTSQFT"], errors="coerce")
    pins = set(df["PIN"])
    print(f"[data] {len(df):,} candidate sales; enriching from extracts ...")

    acct = load_account(pins)
    cons = load_parcel_constraints(pins)
    resb = load_resbldg(pins)
    print(f"       account={len(acct):,}  parcel={len(cons):,}  resbldg={len(resb):,}")

    df["land_val"] = df["PIN"].map(lambda p: acct.get(p, (0, 0))[0])
    df["imps_val"] = df["PIN"].map(lambda p: acct.get(p, (0, 0))[1])
    total_av = df["land_val"] + df["imps_val"]
    df["imps_ratio"] = (df["imps_val"] / total_av).where(total_av > 0)

    df["yr_built"] = df["PIN"].map(lambda p: resb.get(p, {}).get("yr_old"))
    df["yr_built_new"] = df["PIN"].map(lambda p: resb.get(p, {}).get("yr_new"))
    df["living_sqft"] = df["PIN"].map(lambda p: resb.get(p, {}).get("living", 0.0))
    df["has_building"] = df["PIN"].isin(resb)

    for key, default in [("unbuildable", False), ("pct_unusable", 0.0),
                         ("hazard", False), ("odd_shape", False)]:
        df[key] = df["PIN"].map(lambda p, k=key, d=default: cons.get(p, {}).get(k, d))

    # ---- buildable units / product ----
    units_prod = df.apply(lambda r: estimate_units(r["ZONING"], r.get("JURIS", ""),
                                                    r["LOTSQFT"]), axis=1)
    df["est_units"] = [u for u, _ in units_prod]
    df["product"] = [p for _, p in units_prod]

    # ---- finished-unit value: new-construction $/living-sqft per ZIP ----
    df["sale_year"] = df["SaleDate"].dt.year
    new_mask = (
        (df["yr_built_new"] > 0)
        & (df["sale_year"] - df["yr_built_new"] <= NEW_BUILD_YEARS)
        & (df["living_sqft"] > 500)
    )
    df["ppsf_living"] = (df["SalePrice"] / df["living_sqft"]).where(df["living_sqft"] > 500)
    zip_new_ppsf = df[new_mask].groupby("zip5")["ppsf_living"].median()
    county_new_ppsf = df.loc[new_mask, "ppsf_living"].median()
    zip_any_ppsf = df.groupby("zip5")["ppsf_living"].median()
    print(f"[comps] new-construction $/living-sqft: county median "
          f"${county_new_ppsf:,.0f}  ({int(new_mask.sum()):,} new-build comps)")

    def value_per_unit(row) -> float:
        ppsf = zip_new_ppsf.get(row["zip5"])
        if pd.isna(ppsf):
            ppsf = zip_any_ppsf.get(row["zip5"], county_new_ppsf)
        if pd.isna(ppsf):
            ppsf = county_new_ppsf
        return UNIT_SQFT.get(row["product"], UNIT_SQFT["townhome"]) * ppsf

    df["value_per_unit"] = df.apply(value_per_unit, axis=1)

    # ---- pro-forma ----
    unit_sqft = df["product"].map(lambda p: UNIT_SQFT.get(p, UNIT_SQFT["townhome"]))
    df["gdv"] = df["est_units"] * df["value_per_unit"]
    hard = df["est_units"] * unit_sqft * BUILD_COST_PER_SQFT
    soft = hard * SOFT_COST_PCT
    demo = df["has_building"].map(lambda b: DEMO_COST if b else 0)
    selling = df["gdv"] * SELLING_COST_PCT
    df["total_cost"] = df["SalePrice"] + hard + soft + demo + selling
    df["est_margin"] = df["gdv"] - df["total_cost"]
    df["margin_pct"] = (df["est_margin"] / df["gdv"]).where(df["gdv"] > 0)
    df["profit_per_unit"] = (df["est_margin"] / df["est_units"]).where(df["est_units"] > 0)

    # ---- eligibility: feasible AND a real teardown/vacant candidate ----
    age = df["sale_year"] - df["yr_built"]
    df["is_teardown"] = (
        (~df["has_building"])                       # vacant land
        | (df["imps_ratio"] <= TEARDOWN_MAX_IMPS_RATIO)
        | (age >= TEARDOWN_MIN_AGE)
    )
    df["feasible"] = (
        (df["est_units"] >= 1)
        & (~df["unbuildable"])
        & (~df["hazard"])
        & (df["pct_unusable"] < 50)
    )
    df["eligible"] = df["feasible"] & df["is_teardown"]
    return df


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
EXPORT_COLS = [
    "PIN", "address", "zip5", "JURIS", "ZONING", "product", "est_units",
    "LOTSQFT", "yr_built", "imps_ratio", "SalePrice", "value_per_unit",
    "gdv", "total_cost", "est_margin", "margin_pct", "profit_per_unit",
    "is_teardown", "feasible", "odd_shape",
]


def export_csv(df: pd.DataFrame) -> None:
    out = df[df["eligible"]].sort_values("est_margin", ascending=False).copy()
    cols = [c for c in EXPORT_COLS if c in out.columns]
    path = OUTPUTS_DIR / "redevelopment_targets.csv"
    out[cols].rename(columns={"SalePrice": "acquisition", "LOTSQFT": "lot_sqft"}) \
       .to_csv(path, index=False)
    print(f"[out] {len(out):,} eligible targets -> {path.name}")


def chart_top_targets(df: pd.DataFrame) -> None:
    cs.apply()
    sub = df[df["eligible"] & df["zip5"].isin(SHORELINE_ZIPS)
             & (df["est_margin"] > 0)].copy()
    sub = sub.sort_values("est_margin", ascending=False).head(15).iloc[::-1]
    if sub.empty:
        print("  [warn] no positive-margin targets in focus ZIPs")
        return
    labels = [f"{(a or 'vacant lot')[:24]}  ·  {z}"
              for a, z in zip(sub["address"].fillna("vacant lot"), sub["zip5"])]
    margins = (sub["est_margin"] / 1e6).tolist()
    mpct = sub["margin_pct"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(13, 9))
    norm = plt.Normalize(min(mpct), max(mpct))
    bars = ax.barh(labels, margins, color=[cs.VALUE_CMAP(norm(m)) for m in mpct],
                   edgecolor="white", linewidth=0.6)
    for b, m, u, p in zip(bars, margins, sub["est_units"], mpct):
        ax.text(b.get_width() + max(margins) * 0.012,
                b.get_y() + b.get_height() / 2,
                f"${m:.2f}M  ·  {int(u)}u  ·  {p*100:.0f}%",
                va="center", ha="left", fontsize=9, color=cs.MUTED)
    ax.set_xlabel("Estimated development margin ($M)")
    ax.set_xlim(0, max(margins) * 1.20)
    cs.grid_x_only(ax)
    cs.title_block(
        fig,
        "Top Redevelopment Targets — Your Focus ZIPs",
        "Recent sales ranked by estimated margin  ·  label = $margin · units · margin%",
    )
    cs.footer(fig, f"GDV − (acquisition+demo+build+soft+selling)  ·  build "
                   f"${BUILD_COST_PER_SQFT}/sqft, soft {SOFT_COST_PCT:.0%}  ·  ESTIMATES")
    fig.subplots_adjust(top=0.86, left=0.30, right=0.97, bottom=0.10)
    fig.savefig(OUTPUTS_DIR / "10_redevelopment_targets.png")
    plt.close(fig)
    print("  Saved 10_redevelopment_targets.png")


def chart_margin_by_zip(df: pd.DataFrame) -> None:
    cs.apply()
    elig = df[df["eligible"] & (df["est_margin"] > 0)]
    by_zip = elig.groupby("zip5").agg(
        med_margin=("est_margin", "median"),
        n=("est_margin", "size"),
    )
    by_zip = by_zip[by_zip["n"] >= 10].sort_values("med_margin").tail(20)
    if by_zip.empty:
        print("  [warn] no ZIPs with enough eligible targets")
        return
    zips = by_zip.index.tolist()
    vals = (by_zip["med_margin"] / 1e6).tolist()
    colors = [cs.ACCENT if z in SHORELINE_ZIPS else cs.PRIMARY for z in zips]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(zips, vals, color=colors, edgecolor="white", linewidth=0.6)
    for b, v, n in zip(bars, vals, by_zip["n"]):
        ax.text(b.get_width() + max(vals) * 0.012, b.get_y() + b.get_height() / 2,
                f"${v:.2f}M  (n={int(n)})", va="center", ha="left",
                fontsize=9, color=cs.MUTED)
    ax.set_xlabel("Median estimated margin per project ($M)")
    ax.set_xlim(0, max(vals) * 1.16)
    cs.grid_x_only(ax)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in SHORELINE_ZIPS:
            lbl.set_fontweight("bold")
            lbl.set_color(cs.INK)
    cs.title_block(
        fig,
        "Where Redevelopment Pencils — Median Margin by ZIP",
        "Median estimated project margin across eligible teardown candidates "
        "(≥10 per ZIP)",
    )
    cs.footer(fig, "Orange = your focus ZIPs  ·  estimates from the pro-forma "
                   "assumptions block")
    fig.subplots_adjust(top=0.86, left=0.09, right=0.96, bottom=0.09)
    fig.savefig(OUTPUTS_DIR / "11_margin_by_zip.png")
    plt.close(fig)
    print("  Saved 11_margin_by_zip.png")


def main() -> int:
    if not JOINED_CACHE.exists():
        print(f"Missing {JOINED_CACHE} — run build_joined_from_extracts.py first.")
        return 1
    print("=" * 72)
    print("Parcel-level redevelopment pro-forma")
    print("=" * 72)
    try:
        df = build_candidates()
    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        return 1

    elig = df[df["eligible"]]
    print(f"\n[summary] {len(elig):,} eligible targets of {len(df):,} sales  "
          f"({df['feasible'].sum():,} feasible, {df['is_teardown'].sum():,} teardown-ish)")
    pos = elig[elig["est_margin"] > 0]
    print(f"          {len(pos):,} with positive estimated margin  "
          f"(median ${pos['est_margin'].median():,.0f})")
    print("\n  Top 12 targets by estimated margin:")
    show = (elig.sort_values("est_margin", ascending=False).head(12)
            [["zip5", "address", "ZONING", "est_units", "product",
              "SalePrice", "gdv", "est_margin", "margin_pct"]])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False))

    export_csv(df)
    try:
        chart_top_targets(df)
        chart_margin_by_zip(df)
    except Exception as exc:
        print(f"[warn] chart failed: {exc}")
        traceback.print_exc()
    print("\nDone. See outputs_current/redevelopment_targets.csv and charts 10–11.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
