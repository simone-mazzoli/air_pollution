# Pipeline Overview

This file explains the workflow in the order we usually think about it. The
main idea is:

```text
ground measurements + satellite patches -> model -> pollution estimates in new places
```

The repository is still a research project, so some stages are finished, some
are experimental, and some are plans.

We tested the signal and validity of the low-cost sensors. The final model
branch uses EEA reference stations, satellite/context data, geographic
cross-validation, a sealed TEST region, dense prediction, and an exploratory
Kreis-level socioeconomic analysis.

## 1. Input Data

| Input | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Sensor.Community monthly ZIP files | `01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py` | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` | implemented |
| UBA station metadata | `01_scripts_gathering/sensors-related/02_uba_stations_metadata.py` | `data/processed/uba_stations_germany.csv` | implemented |
| UBA daily PM measurements | `01_scripts_gathering/sensors-related/03_download_uba_measurements.py` | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` | implemented |
| EEA PM data | `01_scripts_gathering/01_get_eea_pm.py` | `data/processed/eea/airbase_raw/` | implemented for the final model branch |
| HYRAS weather | `01_scripts_gathering/sensors-related/05_download_extract_hyras.py` | `data/processed/daily_weather/hyras_2024_sds011.parquet` | experimental, used for weather diagnostics |

UBA stations are official German reference stations. We use them because their
measurements are much more reliable than low-cost sensors. Sensor.Community
gives many more locations, but the PM sensors are noisy.

## 2. Cleaning And Aggregation

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Clean PM sensors | `02_scripts_cleaning/sensors-related/01_process_pm_sensors.py` | hourly parquet files, daily CSVs, monthly CSVs | implemented |
| Clean humidity sensors | `02_scripts_cleaning/sensors-related/02_process_humidity_sensors.py` | hourly/daily/monthly humidity tables | implemented |
| Aggregate UBA PM by month | `02_scripts_cleaning/sensors-related/03_aggregate_uba_monthly.py` | `data/processed/monthly_avg/uba/pm_reference_stations_<YYYY-MM>.csv` | implemented |
| Assign UBA stations to states | `02_scripts_cleaning/sensors-related/04_locate_DEUB_UBA_stations.py` | `data/processed/uba/station_land.csv` | implemented |

Annual aggregation means averaging daily values into one value per location for
the year. The current code uses 2024.

## 3. Reference And Sensor Labels

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Assign Sensor.Community sensors to states | `02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py` | `data/processed/sensor_land.csv` | implemented |
| Sensor calibration experiments | `03_scripts_calibration/experiments/` | calibration result tables under `data/processed/calibration/` and `03_scripts_calibration/` | implemented |
| Summarize sensor calibration experiments | `03_scripts_calibration/build_sensor_calibration_summary.py` | `03_scripts_calibration/sensor_community_calibration_summary.csv` | implemented |

Calibration means trying to correct a sensor so that its values match a trusted
reference measurement. We tested several Sensor.Community calibration ideas
because those sensors cover many more places than UBA stations. The result was
mostly negative: the tested methods did not give reference-quality annual
labels. See
[03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md).

The Sensor.Community branch was used to test whether low-cost annual labels
were reliable enough. The final CNN training branch does not depend on these
calibrated sensor labels.

## 4. Satellite Patch Collection

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Download Sentinel-2 patches for sensor labels | `04_GEE/01_download_satellite_patches.py` | `data/processed/satellite/high_res_multispec/*.npy`, `low_res_multispec/*.npy`, `manifest.csv` | implemented for the sensor branch |
| Inspect downloaded patches | `04_GEE/02_inspect_patches.py` | `data/processed/satellite/preview_patches.png` | implemented |
| Download Sentinel-5P sensor patches | `04_GEE/03_download_s5p_patches.py` | S5P patch arrays under `data/processed/satellite/` | experimental |
| Download national Sentinel-5P rasters/crops | `04_GEE/04_download_s5p_nation.py` | national rasters and cropped arrays | experimental |
| Download Sentinel-2 patches for reference stations | `04_GEE/05_download_satellite_patches_uba.py` | station-centered Sentinel-2 arrays | implemented |
| Download Sentinel-5P patches for UBA stations | `04_GEE/06_download_s5p_patches_uba.py` | station-centered S5P arrays | experimental |

Sentinel-2 gives local visual context such as land cover, roads, vegetation, and
urban structure. Sentinel-5P is coarser but measures atmospheric products such
as NO2. Earth Engine scripts require Google Earth Engine access. If
`client_id_GEE.txt` exists at the repository root and contains a service-account
JSON key, the scripts use it. Otherwise they try interactive Earth Engine login.

These scripts can create many `.npy` files and may take a long time.

## 5. Model Training

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Assign EEA geographic folds | `06_models/00_assign_folds.py` | `data/processed/eea/station_fold.csv` | implemented |
| Run geographic CV | `06_models/01_train_cv.py` | `06_models/results/<experiment>/eea_cv_results.json` and predictions CSVs | implemented |
| Train selected final model | `06_models/02_train_final.py` | `06_models/results/cnn_deep_wide/final_model.pt` | already done |
| Evaluate sealed TEST stations | `06_models/03_predict_test.py` | `06_models/results/cnn_deep_wide/test_predictions.csv` | already done |

The final selected model is `cnn_deep_wide`, a scratch CNN with high-resolution
and low-resolution Sentinel-2 inputs plus contextual data. It predicts annual
PM2.5 from official EEA reference-station labels.

## 6. Geographic Validation

Geographic validation means training in some places and validating somewhere
else. This matters because air pollution has strong spatial patterns. Randomly
splitting nearby locations can make results look better than they really are.

The final reference-station validation uses European geographic folds for
development and keeps northern/eastern Germany as the sealed TEST region.

## 7. Final Grid Prediction

Completed for the selected `cnn_deep_wide` model. The saved dense-grid
prediction CSV is:

```text
07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv
```

## 8. Comparison With EEA Grid

Planned/experimental. The EEA data scripts are present, but the repository does
not yet contain a completed comparison workflow.

## 9. Socioeconomic Analysis

Completed as an exploratory Kreis-level analysis for the report. The main joined
table is:

```text
08_kreislevel_data/kreis_exposure_socioeconomic.csv
```

## Data Location Reminder

Most scripts look for `data/` at the repository root. `data/` can be a real
folder or a symlink to an external drive. The repository ignores that directory,
so team members need to get or generate their own local data.

## Main Known Gaps

- Large input data are stored outside Git, so a fresh local checkout cannot
  rerun every step by itself.
- The final `cnn_deep_wide` model selection is recorded in
  `06_models/results/cnn_deep_wide/final_model_selection.json`.
- Some optional Sensor.Community experiments still expect
  `data/processed/sensor_land.csv`.
