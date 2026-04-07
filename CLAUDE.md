# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real estate investment research project for analyzing properties in the Seattle/Shoreline area. It consists of:
- A single-file web application (`property_analyzer.html`) for querying King County GIS data
- CSV datasets of property data for offline analysis
- A CMA (Comparative Market Analysis) PDF document

## Workflow

Always commit and push all changes to git after every update. Do not wait for the user to ask — commit and push automatically once changes are complete.

## Running the Application

Open `property_analyzer.html` directly in any modern browser — no build step, server, or dependencies required.

## Architecture: `property_analyzer.html`

A self-contained HTML/CSS/JS file with no external dependencies. Key components:

**Data Source:** King County GIS ArcGIS REST API
- Endpoint: `https://gismaps.kingcounty.gov/arcgis/rest/services/Property/KingCo_PropertyInfo/MapServer/2/query`
- Fetches in batches of 1000 records; paginates automatically until all records are retrieved

**Filter Logic:**
- Targets single-family homes only (use description contains "Single Family", excludes vacant land)
- Flags properties with assessed value ≥ threshold% below the statistical median of all fetched properties
- Default threshold: 15% below median

**Default Geographic Bounds (Seattle/Shoreline):**
- North: 47.7225 (NE 145th St), South: 47.6612 (NE 45th St)
- West: -122.3180 (Roosevelt Way NE), East: -122.2570 (Lake Washington)

**Features:** sortable table, address/ZIP search, flagged-only toggle, CSV export, progress tracking during fetch

## Data Files

| File | Description |
|------|-------------|
| `Chris Haynes Filtered Properties.csv` | Pre-filtered property subset (~80 KB) |
| `Chris Haynes Poor Condition Properties 03.18.csv` | Properties in poor condition (~1.9 MB) |
| `seattle_shoreline_property_tax_analysis.csv` | Full analysis output (~4 MB) |
| `Copy of Final Property Master Review Sheet 2026.xlsx` | Master property review spreadsheet |
| `joanne mcelroy cma 3-18.pdf` | CMA — general area analysis (Mar 18) |
| `CMA 14549 8th Ave NE 3-19-26.pdf` | CMA — 14549 8th Ave NE (Mar 19) |
| `CMA 14549 8th Ave NE 3-21-26.pdf` | CMA — 14549 8th Ave NE (Mar 21, updated) |
| `Plat, Parcel, Survey Maps.pdf` | Plat and parcel survey maps |
