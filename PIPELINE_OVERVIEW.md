# Pipeline Overview

This file explains the workflow in the order we usually think about it. The
main idea is:

```text
ground measurements + satellite patches -> model -> pollution estimates in new places
```

The repository is still a research project, so some stages are finished, some
are experimental, and some are plans.

We are parallely testing the signal and validity of the sensors, while also creating a branch off using all EEA stations to estimate a held-out region.

## 1. Input Data

| Input | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Sensor.Community monthly ZIP files | `01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py` | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` | implemented |
| UBA station metadata | `01_scripts_gathering/sensors-related/02_uba_stations_metadata.py` | `data/processed/uba_stations_germany.csv` | implemented |
| UBA daily PM measurements | `01_scripts_gathering/sensors-related/03_download_uba_measurements.py` | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` | implemented |
| EEA PM data | `01_scripts_gathering/get_eea_pm.py` | `data/processed/eea/airbase_raw/` | experimental |
| HYRAS weather | `01_scripts_gathering/download_extract_hyras.py` | `data/processed/daily_weather/hyras_2024_sds011.parquet` | experimental, used for weather diagnostics |

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
| Assign Sensor.Community sensors to states | `03_scripts_calibration/active/04_resolve_sensor_land.py` | `data/processed/sensor_land.csv` | implemented |
| Build current annual proxy labels | `03_scripts_calibration/active/03_calibrate_pm_loo.py` | `data/processed/corrected/fold/*/annual/2024.csv` | implemented |
| Summarize sensor calibration experiments | `03_scripts_calibration/build_sensor_calibration_summary.py` | `03_scripts_calibration/sensor_community_calibration_summary.csv` | implemented |

Calibration means trying to correct a sensor so that its values match a trusted
reference measurement. We tested several Sensor.Community calibration ideas
because those sensors cover many more places than UBA stations. The result was
mostly negative: the tested methods did not give reference-quality annual
labels. See
[03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md).

Spatial folds are geographic train/validation splits. Here, German states are
grouped into folds, and `Sachsen-Anhalt` is treated as the sealed test state in
the current sensor-label workflow. The final reference-station validation design
is still being evaluated.

## 4. Satellite Patch Collection

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Download Sentinel-2 patches for sensor labels | `04_GEE/01_download_satellite_patches.py` | `data/processed/satellite/high_res_multispec/*.npy`, `low_res_multispec/*.npy`, `manifest.csv` | implemented |
| Inspect downloaded patches | `04_GEE/02_inspect_patches.py` | `data/processed/satellite/preview_patches.png` | implemented |
| Download Sentinel-5P sensor patches | `04_GEE/03_download_s5p_patches.py` | S5P patch arrays under `data/processed/satellite/` | experimental |
| Download national Sentinel-5P rasters/crops | `04_GEE/04_download_s5p_nation.py` | national rasters and cropped arrays | experimental |
| Download Sentinel-2 patches for UBA stations | `04_GEE/05_download_satellite_patches_uba.py` | station-centered Sentinel-2 arrays | implemented/experimental |
| Download Sentinel-5P patches for UBA stations | `04_GEE/06_download_s5p_patches_uba.py` | station-centered S5P arrays | experimental |

Sentinel-2 gives local visual context such as land cover, roads, vegetation, and
urban structure. Sentinel-5P is coarser but measures atmospheric products such
as NO2. Earth Engine scripts require Google Earth Engine access. If
`client_id_GEE.txt` exists at the repository root and contains a service-account
JSON key, the scripts use it; otherwise they try interactive Earth Engine login.

These scripts can create many `.npy` files and may take a long time.

## 5. Model Training

| Step | Folder/script | Output | Status |
| --- | --- | --- | --- |
| Inspect normalization constants | `06_models/02_resnet/01_find_stats.py` | printed band order and mean/std values | utility |
| Normalize/check patches | `06_models/02_resnet/02_sat_normalize.py` | debug patch grids | implemented |
| Train ResNet baseline | `06_models/02_resnet/03_train_resnet.py` | `resnet_cv_results.json`, predictions CSVs | implemented baseline |

The current trainer uses high-resolution Sentinel-2 patches and predicts
`PM10_corrected`. It does not yet represent the final modeling direction. The
next modeling question is whether training on official reference measurements,
with Sensor.Community possibly added as weak labels, improves performance on
held-out reference stations.

## 6. Geographic Validation

Geographic validation means training in some places and validating somewhere
else. This matters because air pollution has strong spatial patterns. Randomly
splitting nearby locations can make results look better than they really are.

Current code supports state-based folds for the sensor-label pipeline. A final
reference-station validation setup is still a project decision, not a finished
result.

## 7. Final Grid Prediction

Planned. The idea is to run a trained model across a regular grid of locations
so we can produce a pollution surface instead of predictions only at known
stations or sensors.

## 8. Comparison With EEA Grid

Planned/experimental. The EEA data scripts are present, but the repository does
not yet contain a completed comparison workflow.

## 9. Socioeconomic Analysis

Planned/early. There is a preliminary INKAR socioeconomic download script in
`01_scripts_gathering/sensors-related/05_get_socioeconomic_data_PRELIMINARY.py`.
Treat it as exploratory until the analysis design is clearer.

## Data Location Reminder

Most scripts look for `data/` at the repository root. `data/` can be a real
folder or a symlink to an external drive. The repository ignores that directory,
so team members need to get or generate their own local data.

## Main Known Gaps

- The exact final reference-station training and validation setup is still being
  evaluated.
- The current ResNet script trains PM10 only.
- The low-resolution stream, Sentinel-5P streams, and final map prediction are
  not fully integrated into the main trainer.
- `05_scripts_visual/` still expects an older monthly PM filename and may need a
  small path update before use.
