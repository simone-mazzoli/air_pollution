# Pipeline Overview

## Current repository structure

| Path | Purpose 
| --- | --- 
| `01_scripts_gathering/` | Download Sensor.Community archives and UBA reference data. 
| `02_scripts_cleaning/` | Clean raw sensor archives and aggregate PM, humidity, and UBA data. 
| `03_scripts_calibration/` | Validate hourly data, calibrate low-cost PM, and create spatial splits. 
| `04_GEE/` | Download and inspect Sentinel-2 Earth Engine patches. 
| `05_scripts_visual/` | Optional diagnostic maps for sensor/reference coverage. 
| `06_models/02_resnet/` | Current pretrained ResNet high-resolution image baseline. 

## End-to-end flow

1. Download Sensor.Community monthly ZIP archives for PM and humidity/weather sensor types.
2. Fetch UBA station metadata and daily PM10/PM2.5 reference measurements.
3. Clean Sensor.Community PM readings into hourly, daily, and monthly PM aggregates.
4. Clean humidity/weather readings into hourly, daily, and monthly humidity aggregates, including clipped-humidity features.
5. Aggregate UBA daily PM reference data into monthly station means.
6. Resolve UBA stations and low-cost sensors to German states.
7. Calibrate low-cost PM to UBA reference distributions and write annual corrected sensor targets.
8. Download annual Sentinel-2 median composite patches for calibrated sensor locations.
9. Validate and normalize image patches.
10. Train the current high-resolution ResNet baseline over non-test held-out folds.

## Main external datasets

- Sensor.Community monthly CSV archives from `https://archive.sensor.community/csv_per_month/`.
- UBA air-quality station and measurement APIs.
- German state boundary GeoJSON from `isellsoap/deutschlandGeoJSON`, cached as `data/processed/germany_states.geojson`.
- Google Earth Engine Sentinel-2 Surface Reflectance Harmonized collection `COPERNICUS/S2_SR_HARMONIZED`.
- BIFOLD BigEarthNet ResNet50 model metadata and weights from Hugging Face, plus ConfigILM normalization constants.

## Implementation status

| Stage | Current status |
| --- | --- |
| Sensor.Community download | Implemented for 2024 monthly ZIPs with resume and validation. |
| UBA reference retrieval | Implemented for 2024 station metadata and daily PM10/PM2.5 measurements. |
| PM cleaning | Implemented for hourly/daily/monthly aggregation with coverage and plausibility filters. |
| Humidity cleaning | Implemented for hourly/daily/monthly aggregation and clipped relative humidity features. |
| UBA monthly aggregation | Implemented. |
| UBA station state assignment | Implemented, including coordinate fallback for federal/non-state-coded stations. |
| Low-cost PM calibration | Implemented as the established percentile/range-mapping branch plus an isolated regression reference-adjustment alternative. |
| Sensor state assignment | Implemented from SDS011 node coordinates with polygon assignment, 1 km boundary fallback, and outside-Germany exclusion. |
| Coverage and distance diagnostics | Partially implemented; two diagnostic scripts have likely incorrect `BASE_DIR` path roots. |
| Sentinel-2 patch download | Implemented for high-resolution and low-resolution multispectral streams. |
| Patch validation/normalization | Implemented for preview grids and PyTorch dataset utilities. |
| ResNet training | Partially implemented as a high-resolution PM10-only baseline. |
| Final sealed Sachsen-Anhalt evaluation | Not implemented in the current code. |
| Low-resolution/two-stream modeling | Not implemented in the current trainer. |
| Continuous pollution mapping | Not implemented in the current code. |
| Environmental-justice analysis | Not implemented in the current code. |

## Train/validation/test logic

The current code treats `Sachsen-Anhalt` as the sealed CNN test Land. In `03_scripts_calibration/03_calibrate_pm_loo.py`, Sachsen-Anhalt UBA stations are excluded from calibration, and Sachsen-Anhalt sensors are excluded from non-test fold outputs if `data/processed/sensor_land.csv` exists. A separate `Sachsen-Anhalt_TEST` annual file can be produced using coefficients fit only on non-Sachsen-Anhalt data.

The alternative regression calibration lives under `03_scripts_calibration/regression_reference_adjustment/` and prepares corrected annual PM labels only. It evaluates OLS and Huber fits over non-Sachsen-Anhalt inner folds, selects a method/radius from those folds, then fits final PM10 and PM2.5 equations on all non-Sachsen-Anhalt data. The same final coefficients are applied to non-Sachsen-Anhalt CNN training labels and Sachsen-Anhalt test labels. The 2024 audit recommends the shared OLS 20 km label set, while documenting that the global affine fits substantially compress annual target variation.

For cross-validation, German states are mapped into fold groups by `LAND_TO_FOLD`. Some folds combine states, such as Berlin-Brandenburg, Bremen-Niedersachsen, Hamburg-Schleswig-Holstein, and Saarland-Rheinland-Pfalz. The current ResNet script trains on ten non-test folds and validates on the held-out non-test fold, then averages metrics across folds that are run.

The final sealed Sachsen-Anhalt model evaluation is described by the split design but is not implemented in the current repository state.

## Important inconsistencies and caveats

- `03_scripts_calibration/01_validate_new_months.py` and `03_scripts_calibration/02_radius_distance_ablation.py` appear to point at `03_scripts_calibration/data/processed` instead of root `data/processed`.
- `05_scripts_visual` expects older PM merged filenames. Current PM cleaning writes `data/processed/monthly_avg/all_pm_sensors/<YYYY-MM>.csv`.
- Several docstrings and comments outside the refreshed calibration docs may still reference old script names, including `fetch_satellite.py`, `visualize_patches.py`, `get_uba_stations.py`, `correct_pm.py`, and `calibrate_pm.py`.
- `06_models/02_resnet/03_train_resnet.py` currently trains only `PM10_corrected`, although its docstring still describes two-output PM10/PM2.5 regression.
- `requirements.txt` appears to be a captured `pip freeze` with a shell prompt line at the top, not a minimal clean dependency manifest.
- A Python bytecode cache file under `03_scripts_calibration/__pycache__/` and ResNet debug images/results are tracked.

## Paths and credentials

Most scripts build paths relative to the repository root and read or write under `data/processed` or `data/raw`. The `.gitignore` excludes `data/`, `papers/`, `task/`, `venv/`, `.vscode/`, and `client_id_GEE.txt`.

`client_id_GEE.txt` is expected at the repository root if a Google Earth Engine service account JSON key is used. It is ignored by git in the current `.gitignore`.
