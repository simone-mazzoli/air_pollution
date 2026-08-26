# Pipeline Overview

This file describes the current workflow in the order the project was built.

```text
ground measurements + satellite/context patches -> model -> PM2.5 estimates
```

The final CNN training branch uses official EEA reference-station labels. The
Sensor.Community branch is kept because it supports the report conclusion that
the tested low-cost annual calibration approaches were not suitable as final
training labels.

## 1. Input Data

| Input | Script | Main output |
| --- | --- | --- |
| Sensor.Community monthly ZIP files | `01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py` | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` |
| UBA station metadata | `01_scripts_gathering/sensors-related/02_uba_stations_metadata.py` | `data/processed/uba_stations_germany.csv` |
| UBA daily PM measurements | `01_scripts_gathering/sensors-related/03_download_uba_measurements.py` | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` |
| HYRAS weather | `01_scripts_gathering/sensors-related/05_download_extract_hyras.py` | `data/processed/daily_weather/hyras_2024_sds011.parquet` |
| EEA PM data | `01_scripts_gathering/01_get_eea_pm.py` | `data/processed/eea/airbase_raw/` |

UBA and EEA stations provide official reference measurements. Sensor.Community
gives denser spatial coverage but noisier measurements, so it is used for
calibration checks rather than final CNN training labels.

## 2. Cleaning And Aggregation

| Step | Script | Main output |
| --- | --- | --- |
| Clean PM sensors | `02_scripts_cleaning/sensors-related/01_process_pm_sensors.py` | hourly parquet, daily CSV and monthly CSV files |
| Clean humidity sensors | `02_scripts_cleaning/sensors-related/02_process_humidity_sensors.py` | hourly, daily and monthly humidity tables |
| Aggregate UBA PM by month | `02_scripts_cleaning/sensors-related/03_aggregate_uba_monthly.py` | `data/processed/monthly_avg/uba/pm_reference_stations_<YYYY-MM>.csv` |
| Assign UBA stations to states | `02_scripts_cleaning/sensors-related/04_locate_DEUB_UBA_stations.py` | `data/processed/uba/station_land.csv` |
| Assign Sensor.Community sensors to states | `02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py` | `data/processed/sensor_land.csv` |
| Build EEA daily labels | `02_scripts_cleaning/01_build_eea_labels.py` | `data/processed/daily_avg/eea/pm_reference_stations_2024.csv` |

Annual aggregation averages daily measurements into one value per station for
2024.

## 3. Low-Cost Sensor Calibration

Calibration experiments are under `03_scripts_calibration/`.

| Step | Path | Main output |
| --- | --- | --- |
| Nearby-reference calibration tests | `03_scripts_calibration/experiments/nearby_reference_regression/` | calibration result tables |
| Clustered-sensor checks | `03_scripts_calibration/experiments/clustered_sensors/` | exploratory calibration outputs |
| Data and distance diagnostics | `03_scripts_calibration/experiments/diagnostics/` | diagnostic outputs |
| Calibration summary | `03_scripts_calibration/build_sensor_calibration_summary.py` | `03_scripts_calibration/sensor_community_calibration_summary.csv` |

The tested Sensor.Community calibration approaches were not sufficiently
accurate and spatially informative for use as reference-equivalent annual
labels. The final CNN training branch therefore uses official EEA labels.

## 4. Satellite Patch Collection

| Step | Script | Main output |
| --- | --- | --- |
| Download EEA station patches | `04_GEE/01_download_eea_patches.py` | `data/processed/satellite_eea/<stream_folder>/<station_code>.npy` |
| Download dense-grid patches | `04_GEE/02_download_dense_grid_patches.py` | `data/processed/satellite_grid/<stream_folder>/<grid_id>.npy` |

The Earth Engine exports include Sentinel-2, Sentinel-5P, aerosol-wide context
and Copernicus DEM streams. Export dimensions are nominal projected footprints,
not exact physical ground dimensions.

## 5. Model Training And Evaluation

| Step | Script | Main output |
| --- | --- | --- |
| Assign EEA geographic folds | `06_models/00_assign_folds.py` | `data/processed/eea/station_fold.csv` |
| Run geographic CV | `06_models/01_train_cv.py` | `06_models/results/<experiment>/eea_cv_results.json` |
| Train selected final model | `06_models/02_train_final.py` | `06_models/results/cnn_deep_wide/final_model.pt` |
| Evaluate sealed TEST stations | `06_models/03_predict_test.py` | `06_models/results/cnn_deep_wide/test_predictions.csv` |

The selected final model is `cnn_deep_wide`, a scratch CNN with high-resolution
Sentinel-2, wider Sentinel-2 context, Sentinel-5P/context streams and elevation
inputs. The 24-epoch final training run comes from the original pre-TEST
model-selection record.

Later learning-curve runs are kept as supplementary reproducibility diagnostics.
They are not the basis for changing model selection.

## 6. Dense Prediction And Analysis

| Step | Script | Main output |
| --- | --- | --- |
| Predict dense grid | `07_prediction_analysis/02_predict_grid.py` | `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv` |
| Analyze TEST predictions | `07_prediction_analysis/01_analyze_test_predictions.py` | figures and tables under `07_prediction_analysis/outputs/` |
| Build Kreis exposure table | `08_kreislevel_data/02_build_kreis_exposure.py` | `08_kreislevel_data/kreis_exposure_socioeconomic.csv` |
| Map Kreis socioeconomic results | `08_kreislevel_data/03_map_pollution_inequality.py` | `08_kreislevel_data/figures/pollution_inequality_maps/` |
| Build final report figures | `09_report_figures/build_report_figures.py` | `Air_pollution_report/Figures/generated/` |
| Build preliminary analysis figures | `09_report_figures/build_preliminary_analysis_figures.py` | preliminary figures and supporting CSVs |

The dense continuous map is interpolated from grid predictions. Displayed map
pixels are not independent model predictions. The Kreis socioeconomic analysis
is descriptive and ecological, using modeled area-level ambient PM2.5 rather
than population-weighted personal exposure.

## Data Location

Most scripts expect a local `data/` directory at the repository root. The
repository keeps only `data/README.md`; large raw downloads and processed arrays
must be stored locally outside Git.
