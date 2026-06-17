"""
Build data/kc_market_joined.csv from the King County Assessor BULK EXTRACTS
instead of the live GIS REST API.

Why this exists
---------------
The live GIS Sales layer (KingCo_PropertyInfo MapServer/3) lags badly — as of
mid-2026 it topped out at Nov-2025 sales with zero 2026 records. The Assessor's
downloadable extracts (refreshed ~weekly) carry sales through the current month.
This script reproduces the same joined schema the downstream scripts expect, so
dev_score_and_shoreline.py and shoreline_zip_map.py (and the cache branch added
to current_market_analysis.py) all run on fresh 2026 data with no other changes.

Hybrid data sourcing
--------------------
Only the GIS *Sales* layer (Layer 3) lags; the GIS *Parcel* layer (Layer 2) is
current. So we take the fresh 2026 SALES (dates + prices) from the bulk extract,
but pull the PARCEL ATTRIBUTES (situs ZIP, lot sqft, present-use, situs address)
from the live GIS Parcel layer for just the sold PINs. The bulk Account extract
address is the TAXPAYER MAILING address (owners live nationwide -> hundreds of
bogus ZIPs), so it is NOT used for the property location.

Inputs (unzipped into data/kc_extract/ from the .zip extracts):
    EXTR_RPSale.csv          - sales         (Major, Minor, DocumentDate, SalePrice, ...)
    EXTR_Parcel.csv          - parcel filter (Major, Minor, PropType) -> residential pre-filter
GIS:
    KingCo_PropertyInfo MapServer/2 (Parcels) -> ZIP5, LOTSQFT, PREUSE_DESC, ADDR_FULL ...

Output schema (matches the GIS-built cache exactly):
    PIN, SaleDate, SalePrice, Principal_Use, Property_Type, address,
    ZIP5, LOTSQFT, PREUSE_DESC, PROPTYPE, CTYNAME, POSTALCTYNAME, ADDR_FULL,
    zip5, price_per_lot_sqft

Methodology parity with the GIS pipeline:
    - residential only        (parcel PropType == 'R')
    - last 12 months           (relative to the max DocumentDate in the data)
    - all sales in window kept  (not deduped to one-per-PIN)
    - parcel attrs + situs ZIP from the live GIS Parcel layer (same as original)

Valid-sale filter (the raw extract, unlike the GIS Sales layer, includes every
recorded deed — quitclaims, gifts, estate settlements, partial-interest, etc.).
We keep only true arms-length market sales using the King County assessor's
standard criteria:
    - SaleInstrument == 3   (Statutory Warranty Deed)
    - SaleReason     == 1   (None / no special reason)
    - SalePrice      >= 10,000
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_DIR = DATA_DIR / "kc_extract"
OUT = DATA_DIR / "kc_market_joined.csv"

SALES_CSV = EXTRACT_DIR / "EXTR_RPSale.csv"
PARCEL_CSV = EXTRACT_DIR / "EXTR_Parcel.csv"
ACCT_CSV = EXTRACT_DIR / "EXTR_RPAcct_NoName.csv"

PARCEL_LAYER = (
    "https://gismaps.kingcounty.gov/arcgis/rest/services/Property/"
    "KingCo_PropertyInfo/MapServer/2/query"
)
PARCEL_PIN_BATCH = 400
REQUEST_TIMEOUT = 60

MIN_PRICE = 10_000
LOOKBACK_MONTHS = 12
ENC = "latin-1"

# King County "valid sale" codes (see EXTR_LookUp.csv).
# LUType 6 SaleInstrument: 2 = Warranty Deed, 3 = Statutory Warranty Deed.
# Recent sales are coded 2 far more often than 3, so accept the warranty-deed
# family and exclude quitclaims/trustee/executor/bargain-&-sale/generic deeds.
VALID_INSTRUMENTS = {"2", "3"}
VALID_REASON = "1"       # LUType 5: None (arms-length)

# Ratio-study guard: drop sales priced implausibly far below the parcel's total
# assessed value (partial-interest / non-market transfers that still recorded as
# warranty deeds). Sales on unassessed parcels (total == 0) are kept.
MIN_SALE_TO_ASSESSED = 0.40

# csv has huge free-text fields in places; lift the field-size cap.
csv.field_size_limit(10_000_000)


def pin(major: str, minor: str) -> str:
    """King County 10-digit PIN = Major(6) + Minor(4), zero-padded."""
    return f"{major.strip().zfill(6)}{minor.strip().zfill(4)}"


def load_residential_pins() -> dict[str, tuple[str, str]]:
    """PIN -> (CurrentZoning, DistrictName/jurisdiction) for PropType=='R' parcels.

    Zoning + jurisdiction come from the bulk Parcel extract (the live GIS query
    does not expose zoning). Jurisdiction is needed to disambiguate R-codes that
    mean different things by city (e.g. Shoreline R-6 = 6 units/acre vs Medina
    R16 = 16,000 sqft minimum lot)."""
    out: dict[str, tuple[str, str]] = {}
    total = 0
    with PARCEL_CSV.open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if row["PropType"].strip() == "R":
                out[pin(row["Major"], row["Minor"])] = (
                    row["CurrentZoning"].strip(),
                    row["DistrictName"].strip(),
                )
    print(f"[parcel] {len(out):,} residential PINs (of {total:,} total)")
    return out


def load_assessed_values() -> dict[str, float]:
    """PIN -> total assessed value (ApprLandVal + ApprImpsVal) from Account extract."""
    out: dict[str, float] = {}
    with ACCT_CSV.open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            try:
                land = float(row["ApprLandVal"].strip() or 0)
                imps = float(row["ApprImpsVal"].strip() or 0)
            except ValueError:
                continue
            out[pin(row["Major"], row["Minor"])] = land + imps
    print(f"[account] assessed values for {len(out):,} PINs")
    return out


def fetch_parcel_attrs(pins: list[str]) -> pd.DataFrame:
    """Live GIS Parcel layer attrs for the sold PINs (current, unlike sales layer)."""
    print(f"[gis] fetching parcel attrs for {len(pins):,} PINs ...")
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
        print(f"    batch {i+1:>3}/{batches}  total {len(rows):,}")
        time.sleep(0.05)
    return pd.DataFrame(rows)


def build() -> int:
    if not SALES_CSV.exists():
        print(f"Missing {SALES_CSV} — unzip the extracts into {EXTRACT_DIR} first.")
        return 1

    resid_pins = load_residential_pins()
    assessed = load_assessed_values()

    print("[sales] streaming sales, keeping residential arms-length ...")
    rows: list[dict] = []
    seen = kept = 0
    drop_price = drop_resid = drop_instrument = drop_reason = drop_ratio = 0
    with SALES_CSV.open(encoding=ENC, newline="") as f:
        for row in csv.DictReader(f):
            seen += 1
            try:
                price = float(row["SalePrice"].strip() or 0)
            except ValueError:
                continue
            if price < MIN_PRICE:
                drop_price += 1
                continue
            if row["SaleInstrument"].strip() not in VALID_INSTRUMENTS:
                drop_instrument += 1
                continue
            if row["SaleReason"].strip() != VALID_REASON:
                drop_reason += 1
                continue
            p = pin(row["Major"], row["Minor"])
            zoning_juris = resid_pins.get(p)
            if zoning_juris is None:
                drop_resid += 1
                continue
            total_av = assessed.get(p, 0.0)
            if total_av > 0 and price < MIN_SALE_TO_ASSESSED * total_av:
                drop_ratio += 1
                continue
            ds = row["DocumentDate"].strip()
            try:
                sale_date = pd.Timestamp(ds)  # parses MM/DD/YYYY
            except (ValueError, TypeError):
                continue
            rows.append({
                "PIN": p,
                "SaleDate": sale_date,
                "SalePrice": price,
                "Principal_Use": "RESIDENTIAL",
                "Property_Type": row.get("PropertyType", "").strip(),
                "ZONING": zoning_juris[0],
                "JURIS": zoning_juris[1],
            })
            kept += 1
    print(f"[sales] scanned {seen:,} rows -> {kept:,} residential arms-length sales")
    print(f"        dropped: price<{MIN_PRICE} {drop_price:,} | "
          f"non-warranty instrument {drop_instrument:,} | "
          f"non-arms-length reason {drop_reason:,} | non-residential {drop_resid:,} | "
          f"sale<{MIN_SALE_TO_ASSESSED:.0%} assessed {drop_ratio:,}")

    df = pd.DataFrame(rows)
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    df = df.dropna(subset=["SaleDate", "PIN", "SalePrice"])

    # Trim to last 12 months relative to max date (same rule as GIS pipeline)
    max_date = df["SaleDate"].max()
    cutoff = max_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    before = len(df)
    df = df[df["SaleDate"] > cutoff]
    print(f"[trim] {before:,} -> {len(df):,} rows in window "
          f"{cutoff.date()} -> {max_date.date()}")

    # Enrich with SITUS parcel attributes from the live GIS Parcel layer
    unique_pins = df["PIN"].astype(str).unique().tolist()
    parcels = fetch_parcel_attrs(unique_pins)
    if parcels.empty:
        print("[gis] FAILED: no parcel attributes returned")
        return 2
    df = df.merge(parcels, on="PIN", how="inner")
    print(f"[join] {len(df):,} sales matched to GIS parcels")

    # Derived columns to match the cache schema
    df["zip5"] = df["ZIP5"].astype(str).str.strip().str.zfill(5)
    df = df[df["zip5"].str.match(r"^\d{5}$")]
    df["address"] = df["ADDR_FULL"]
    df["LOTSQFT"] = pd.to_numeric(df["LOTSQFT"], errors="coerce")
    df["price_per_lot_sqft"] = df.apply(
        lambda r: (r["SalePrice"] / r["LOTSQFT"])
        if r["LOTSQFT"] and r["LOTSQFT"] > 0 else None,
        axis=1,
    )

    cols = ["PIN", "SaleDate", "SalePrice", "Principal_Use", "Property_Type",
            "address", "ZIP5", "LOTSQFT", "PREUSE_DESC", "PROPTYPE",
            "CTYNAME", "POSTALCTYNAME", "ADDR_FULL", "ZONING", "JURIS",
            "zip5", "price_per_lot_sqft"]
    df = df[cols]
    df.to_csv(OUT, index=False)
    print(f"[out] wrote {len(df):,} rows -> {OUT}")
    print(f"      ZIPs: {df['zip5'].nunique()}   "
          f"median price: ${df['SalePrice'].median():,.0f}   "
          f"usable LOTSQFT: {df['price_per_lot_sqft'].notna().sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
