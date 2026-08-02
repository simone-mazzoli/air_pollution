# Visualization Diagnostics

## Purpose

This directory contains diagnostic plotting scripts for sensor and reference-station locations. These scripts are not required by the modeling pipeline, but they help inspect spatial coverage and data availability.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `plot_sensors_map_leaflet.py` | Creates an interactive Folium/Leaflet map with humidity sensors, low-cost PM sensors, and UBA reference stations for a hard-coded month. | Monthly PM, monthly UBA, and monthly humidity CSVs. | `data/processed/plots/sensors_map_leaflet_<MONTH>.html` | Diagnostic |
| `plot_sensors_map_static.py` | Creates a static Matplotlib map over German state borders for a hard-coded month. Fetches/caches the state GeoJSON if needed. | Monthly PM, monthly UBA, monthly humidity CSVs, and optional cached state GeoJSON. | `data/processed/plots/sensors_map_static_<MONTH>.png`; `data/processed/germany_states.geojson` | Diagnostic |

## Pipeline position

Run these after [`../02_scripts_cleaning`](../02_scripts_cleaning/README.md) has produced monthly PM, humidity, and UBA station outputs. They are optional diagnostics and do not feed directly into calibration, Earth Engine downloads, or modeling.

## Data flow

Monthly low-cost PM means + monthly humidity means + monthly UBA reference means + German state polygons -> visual spatial coverage maps.

## Running the scripts

```bash
python 05_scripts_visual/plot_sensors_map_leaflet.py
python 05_scripts_visual/plot_sensors_map_static.py
```

Neither script defines command-line arguments. The month is controlled by the `MONTH` constant inside each file.

## Important assumptions and caveats

- Both scripts currently set `MONTH = "2024-01"` in code.
- Both scripts expect the PM monthly file at `data/processed/monthly_avg/all_pm_sensors/germany_monthly_avg_ALLPM_<MONTH>.csv`.
- The current PM cleaning script writes merged PM monthly output as `data/processed/monthly_avg/all_pm_sensors/<YYYY-MM>.csv`, so these visualization scripts likely need path updates before they run against current outputs.
- The static map fetches German state borders from `isellsoap/deutschlandGeoJSON` and caches them at `data/processed/germany_states.geojson`.
