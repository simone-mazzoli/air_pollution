# Calibration and Spatial Splits

## Purpose

This directory validates cleaned hourly data, assigns stations and sensors to German states, calibrates low-cost PM annual targets against UBA reference data, and checks spatial coverage for held-out-state modeling.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `01_validate_new_months.py` | Checks expected PM/humidity hourly files, schemas, value ranges, and monthly coverage before calibration. | `data/processed/hourly/pm/all_pm_sensors/*.parquet`; `data/processed/hourly/humidity/all_sensors/*.parquet`; node coordinate parquet files. | Printed validation report. | Validation |
| `02_radius_distance_ablation.py` | Diagnostic for whether nearest-UBA-station distance harms calibration transfer. Fits against nearest station and validates against the second-nearest station. | Hourly PM/humidity files; node coordinates; UBA daily reference data; optional `data/processed/calibration/global_kappa.json`. | Printed distance-bin table; `data/processed/calibration/radius_diagnostic.csv` | Diagnostic / ablation |
| `03_calibrate_pm_loo.py` | Creates annual calibrated PM targets using leave-one-fold-out calibration across non-test folds, with Sachsen-Anhalt excluded from fitting and written as a separate test-label file. | `data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet`; PM node coordinates; `data/processed/daily_avg/uba/pm_reference_stations_2024.csv`; `data/processed/uba/station_land.csv`; optional `data/processed/sensor_land.csv`. | `data/processed/calibration/linear_by_fold.json`; `data/processed/calibration/linear_test_land.json`; `data/processed/corrected/fold/<fold>/annual/2024.csv`; `data/processed/corrected/fold/Sachsen-Anhalt_TEST/annual/2024.csv` | Main pipeline step |
| `04_resolve_sensor_land.py` | Assigns SDS011 sensor locations to a German Land using state polygons, a documented near-boundary fallback, and outside-Germany exclusion. | `data/processed/hourly/pm/nodes/sds011_<YYYY-MM>.parquet`; `data/processed/germany_states.geojson`. | `data/processed/sensor_land.csv`; assignment diagnostics JSON/CSV and map. | Main pipeline step for modeling splits |
| `regression_reference_adjustment/calibrate_pm_regression_loo.py` | Alternative regression-based SDS011 reference adjustment using OLS/Huber and radius comparison, isolated from percentile outputs. | Merged hourly PM files; SDS011 node coordinates; UBA daily PM; `sensor_land.csv`; `uba/station_land.csv`. | `data/processed/calibration/regression_reference_adjustment/*`; `data/processed/corrected/regression_reference_adjustment/*` | Alternative calibration branch |
| `05_check_land_coverage.py` | Computes sensor density and buffered area coverage for each fold group to help choose or audit a held-out state. | `data/processed/germany_states.geojson`; `data/processed/sensor_land.csv`; fold annual CSVs if coordinates are missing. | Printed coverage table. | Diagnostic |

## Pipeline position

Run this after [`../02_scripts_cleaning`](../02_scripts_cleaning/README.md). The intended order is:

1. `01_validate_new_months.py` before expensive calibration, if its paths are corrected or run from the expected location.
2. `04_resolve_sensor_land.py` to create `sensor_land.csv` from SDS011 node coordinates.
3. `03_calibrate_pm_loo.py` for the established percentile/range-mapping calibration.
4. Optionally run `regression_reference_adjustment/calibrate_pm_regression_loo.py` for the isolated regression alternative.
5. `05_check_land_coverage.py` and `02_radius_distance_ablation.py` as diagnostics.

The next stage is [`../04_GEE`](../04_GEE/README.md), which downloads Sentinel-2 patches for calibrated sensor locations.

## Data flow

Hourly low-cost PM -> daily low-cost PM means using MEZ dates -> nearest UBA station matching -> fold-specific calibration coefficients -> annual corrected PM targets per sensor.

Station coordinates/codes -> station Land lookup -> folds for calibration. Sensor coordinates -> sensor Land lookup -> fold assignments for CNN training and the sealed Sachsen-Anhalt test set.

## Running the scripts

```bash
python 03_scripts_calibration/01_validate_new_months.py
python 03_scripts_calibration/02_radius_distance_ablation.py
python 03_scripts_calibration/02_radius_distance_ablation.py --months 2024-01 2024-02 --year 2024
python 03_scripts_calibration/03_calibrate_pm_loo.py
python 03_scripts_calibration/03_calibrate_pm_loo.py --months 2024-01 2024-02 --year 2024 --refit
python 03_scripts_calibration/03_calibrate_pm_loo.py --apply-fold Bayern
python 03_scripts_calibration/04_resolve_sensor_land.py
python 03_scripts_calibration/04_resolve_sensor_land.py --boundaries data/processed/germany_states.geojson --name-column name
python 03_scripts_calibration/regression_reference_adjustment/calibrate_pm_regression_loo.py --method both --radius-km 5 10 20 --refit
python 03_scripts_calibration/05_check_land_coverage.py
```

## Important assumptions and caveats

- `03_calibrate_pm_loo.py` sets `TEST_LAND = "Sachsen-Anhalt"` and excludes that Land's UBA stations from calibration. If `data/processed/sensor_land.csv` exists, Sachsen-Anhalt sensors are also removed from all non-test fold outputs.
- Calibration uses fold groups that combine some small or neighboring states, for example Berlin-Brandenburg and Bremen-Niedersachsen.
- The current calibration fit uses robust percentile-range matching, not the older OLS per-sensor approach described in some diagnostic comments.
- The regression alternative is a label-preparation pipeline only. It does not train the CNN, and it writes only under `regression_reference_adjustment/` output roots.
- `04_resolve_sensor_land.py` now reads SDS011 node coordinates directly. It assigns points covered by German Land polygons, accepts only near-boundary fallbacks within `1 km`, and excludes points farther outside Germany rather than forcing them to the nearest Land.
- Annual targets require at least `182` valid days and drop sensors whose corrected annual mean exceeds `50 ug/m3` for any corrected pollutant.
- `01_validate_new_months.py` and `02_radius_distance_ablation.py` currently set `BASE_DIR` to `03_scripts_calibration`, so their `data/processed` paths appear to resolve under `03_scripts_calibration/data/processed` instead of the repository root. This README documents intended behavior, but those scripts may need path correction before use.
- Some comments still reference older names such as `calibrate_pm_leave_one_fold_out.py`, `calibrate_pm.py`, and `global_kappa.json`; the main current calibration script is `03_calibrate_pm_loo.py`.
