# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real estate investment research tools for analyzing properties in the Seattle/Shoreline area. All tools are self-contained HTML/CSS/JS files — open directly in any modern browser, no build step or server required.

## Workflow

Always commit and push all changes to git after every update. Do not wait for the user to ask — commit and push automatically once changes are complete.

## Application Files

| File | Purpose |
|------|---------|
| `index.html` | **Primary app.** Property tax analyzer with GIS search, CSV upload parcel mapper, corner lot detection, and interactive map |
| `corner_lot_finder.html` | Standalone corner lot finder for Chris Haynes PCP dataset (uses same GIS endpoints as `index.html` but scoped to a single CSV) |
| `owner_lookup.html` | PIN matcher for King County bulk data CSVs (Parcel + Real Property Sales + Real Property Account). Uses streaming file reader for multi-GB Sales file |
| `corner_lot_targets.html` | Static corner lot target list for developer acquisition |
| `corner-lot-outreach/corner_lot_outreach_plan.html` | Business playbook (non-technical) for licensed-broker corner-lot acquisition and developer sprint. Reference document, not an app. Lives in its own `corner-lot-outreach/` folder for project assets |
| `agent-website/index.html` | Real estate agent marketing site template (standalone, based on Lella Norberg design) — unrelated to the analysis tools |

All HTML files open directly via `file://` &mdash; no server, no build, no package manager. Leaflet is the only external dependency and loads from CDN.

## Architecture: `index.html`

Single-file app (~1800 lines) with no external dependencies except Leaflet (loaded via CDN). Major subsystems:

**1. GIS Property Search** (`run()` / `fetchBatch()`)
- King County ArcGIS REST API, Layer 2 (Parcels) + Layer 3 (Sales) + Roads Layer 6
- Endpoint: `https://gismaps.kingcounty.gov/arcgis/rest/services/Property/KingCo_PropertyInfo/MapServer/2/query`
- Paginates in batches of 1000 via `resultOffset`; stops when `exceededTransferLimit` is false
- Filters by property type (single-family or all residential) using `PREUSE_DESC` / `PROPTYPE`
- After fetching parcels: fetches road network, detects corner lots spatially, then fetches sales data for owner names

**2. Corner Lot Detection** (`computeCornerLots()`)
- Spatial analysis comparing parcel edge geometry against road centerlines
- A corner lot requires: two different streets with sufficient frontage (22m+), streets at >32 degree angle, road segments crossing within 10m, and a parcel vertex within 16m of that crossing
- Uses meter-scale approximation at lat ~47.7 (1 deg lon ~ 74,900m, 1 deg lat ~ 111,139m)

**3. CSV Upload & Parcel Mapper** (`handleCsvFile()` / `renderCsvMap()`)
- Drag-and-drop or file picker for CSV files containing parcel data
- Auto-detects PIN column from common names: PIN, APN, APN - UNFORMATTED, PARCEL, etc.
- Zero-pads PINs to 10 digits (King County standard; Excel strips leading zeros)
- Batch-queries GIS API (50 PINs per request) for parcel geometries
- Renders found parcels on a separate Leaflet map with popups

**4. Analysis & Rendering** (`analyze()` / `renderTable()` / `renderMapParcels()`)
- Computes median assessed value, flags properties below configurable threshold (default 15%)
- Table supports sorting by any column, text search (address/ZIP/city/owner), flagged-only and corner-lot-only filters
- Map view shows parcel polygons color-coded by flagged status with corner lot borders

**Key data flow:** Fetch parcels (with geometry rings) -> fetch roads -> compute corners -> fetch sales -> analyze (median/flagging) -> render table + map

## Architecture: `owner_lookup.html`

Different model than `index.html`: instead of the live GIS API, this tool joins three King County bulk-data CSV exports locally in the browser. It's the path for batch work on tens of thousands of rows without API rate limits.

- **Three required uploads** (any order): Parcel CSV, Real Property Sales CSV, Real Property Account CSV. Each maps to a distinct slot in the UI (`loadParcels()`, `loadSales()`, `loadAcct()`).
- **Sales file is multi-GB.** `loadSales()` uses `file.stream().getReader()` + chunked line parsing rather than `FileReader.readAsText()` — loading the whole string would OOM. Maintains a `leftover` buffer across chunks so rows aren't split at chunk boundaries. Most recent sale per PIN wins.
- **Join key is 10-digit zero-padded PIN.** `normPin()` handles both string PINs and concatenated Major/Minor integer fields (Sales data uses Major+Minor; Parcel data uses a single PIN field).
- **Output** is a joined table (filterable by absentee-owner flag, exportable to CSV) &mdash; this is how you produce the taxpayer-mailing-address list referenced by `corner-lot-outreach/corner_lot_outreach_plan.html`.

## King County GIS API Notes

- PIN field is a 10-digit zero-padded string (e.g., `0164000381`)
- Parcel geometry comes as `rings` arrays of `[lon, lat]` coordinate pairs (must flip to `[lat, lon]` for Leaflet)
- Sales layer (Layer 3) provides `buyername` and `SaleDate` — most recent sale per PIN is used for owner name
- API has no auth but may have CORS issues in some browsers; Chrome/Edge work best

## Data Files (gitignored)

`.gitignore` excludes `*.csv`, `*.pdf`, `*.xlsx`, `*.xls`, and `memory/`. Team CSVs and King County bulk exports live alongside the HTML files but never commit. Key datasets in use:

**Team / input CSVs:**
- `Chris Haynes Filtered Properties.csv` — uses `APN - UNFORMATTED` as PIN column
- `seattle_shoreline_property_tax_analysis.csv` — uses `PIN` column, exported by `index.html`
- `Chris Haynes Poor Condition Properties 03.18.csv` — large PCP dataset (~1.9 MB)
- `Haynes Team Corner Lot Search Owners 04.15.26.csv` — current corner-lot target list referenced in the outreach plan

**King County bulk exports (zipped):**
- `Parcel.zip`, `Real Property Sales.zip`, `Real Property Account.zip`, `Tax Data.zip` — the canonical King County data dumps. Unzip and feed to `owner_lookup.html` to produce owner mailing lists. The Sales CSV specifically is large enough that non-streaming parsers will crash the tab.
