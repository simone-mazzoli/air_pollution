# Nearby Reference Regression Experiments

These experiments tested a simple idea: for each low-cost SDS011 sensor, can a
nearby official UBA station help us correct the sensor's PM10/PM2.5 readings?

The answer was mostly no for annual labels. OLS and Huber regression reduced
the huge raw SDS011 error, but the corrected annual labels became too close to a
constant mean and lost most spatial variation. That makes them weak targets for
an image model.

## Main Scripts

| Script | What it does |
| --- | --- |
| `audit_2024_data_completeness.py` | Checks whether the 2024 inputs are complete enough for these experiments. | 
| `calibrate_pm_regression_loo.py` | Fits OLS/Huber nearby-reference corrections and writes diagnostic outputs. |
| `compare_label_construction_methods.py` | Compares raw SDS011, constant mean, percentile/range mapping, OLS, and Huber on the same held-out folds. |
| `eda_close_reference_inventory.py` | Counts close SDS011-UBA pairs and nearby weather coverage. |
| `fit_close_reference_weather_models.py` | Tests close-reference daily models with PM, humidity, and temperature. |

## Inputs

These scripts expect processed data such as:

- `data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet`
- `data/processed/hourly/pm/nodes/sds011_<YYYY-MM>.parquet`
- `data/processed/daily_avg/uba/pm_reference_stations_2024.csv`
- `data/processed/uba/station_land.csv`
- `data/processed/sensor_land.csv`

`sensor_land.csv` is created by
`02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py`.

They default to `data/processed/` under the repository root. Most scripts also
accept `--processed-dir` if your processed data live somewhere else.

## Useful Commands

Small smoke test:

```bash
python3 03_scripts_calibration/experiments/nearby_reference_regression/calibrate_pm_regression_loo.py \
  --months 2024-04 2024-05 2024-06 \
  --method ols \
  --radius-km 10 \
  --refit
```

Full method comparison:

```bash
python3 03_scripts_calibration/experiments/nearby_reference_regression/compare_label_construction_methods.py
```

Weather-aware diagnostics:

```bash
python3 01_scripts_gathering/sensors-related/03_download_uba_measurements.py --station-metadata
python3 01_scripts_gathering/download_extract_hyras.py
python3 03_scripts_calibration/experiments/nearby_reference_regression/fit_close_reference_weather_models.py
```

The weather download can fetch large HYRAS NetCDF files.

## Outputs

Most outputs go under:

```text
data/processed/calibration/regression_reference_adjustment/
data/processed/corrected/regression_reference_adjustment/
```

The key written-up result is in [RESULTS.md](RESULTS.md). The short project
summary is in
[../../SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](../../SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md).

## Important Interpretation

Do not describe this branch as successful sensor calibration. It is useful
because it shows why the regression approach is not enough for reference-quality
annual labels.
