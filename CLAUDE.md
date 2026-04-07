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
| `corner_lot_targets.html` | Static corner lot target list for developer acquisition |
| `corner_lot_finder.html` | Standalone corner lot finder for Chris Haynes PCP dataset |

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

## King County GIS API Notes

- PIN field is a 10-digit zero-padded string (e.g., `0164000381`)
- Parcel geometry comes as `rings` arrays of `[lon, lat]` coordinate pairs (must flip to `[lat, lon]` for Leaflet)
- Sales layer (Layer 3) provides `buyername` and `SaleDate` — most recent sale per PIN is used for owner name
- API has no auth but may have CORS issues in some browsers; Chrome/Edge work best

## Data Files (gitignored)

CSV, PDF, XLSX files are in `.gitignore`. Key datasets:
- `Chris Haynes Filtered Properties.csv` — uses `APN - UNFORMATTED` as PIN column
- `seattle_shoreline_property_tax_analysis.csv` — uses `PIN` column, exported by this app
- `Chris Haynes Poor Condition Properties 03.18.csv` — large PCP dataset (~1.9 MB)
