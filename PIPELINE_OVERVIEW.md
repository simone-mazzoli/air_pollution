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

| Input | Folder/script | Output | 
| --- | --- | --- | --- |
| Sensor.Community monthly ZIP files | `01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py` | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` | 
| UBA station metadata | `01_scripts_gathering/sensors-related/02_uba_stations_metadata.py` | `data/processed/uba_stations_germany.csv` | 
| UBA daily PM measurements | `01_scripts_gathering/sensors-related/03_download_uba_measurements.py` | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` | 
| EEA PM data | `01_scripts_gathering/get_eea_pm.py` | `data/processed/eea/airbase_raw/` | 
| HYRAS weather | `01_scripts_gathering/sensor-related/05_download_extract_hyras.py` | `data/processed/daily_weather/hyras_2024_sds011.parquet` | 

UBA stations are official German reference stations. We use them because their
measurements are much more reliable than low-cost sensors. Sensor.Community
gives many more locations, but the PM sensors are noisy. The Hyras weather data 
is used for the calibration experiment. The EEA PM data provides PM data from 
all of Europe, which ends up being the training data. 

## 2. Cleaning And Aggregation

| Step | Folder/script | Output |
| --- | --- | --- | --- |
| Clean PM sensors | `02_scripts_cleaning/sensors-related/01_process_pm_sensors.py` | hourly parquet files, daily CSVs, monthly CSVs | 
| Clean humidity sensors | `02_scripts_cleaning/sensors-related/02_process_humidity_sensors.py` | hourly/daily/monthly humidity tables | 
| Aggregate UBA PM by month | `02_scripts_cleaning/sensors-related/03_aggregate_uba_monthly.py` | `data/processed/monthly_avg/uba/pm_reference_stations_<YYYY-MM>.csv` | 
| Assign UBA stations to states | `02_scripts_cleaning/sensors-related/04_locate_DEUB_UBA_stations.py` | `data/processed/uba/station_land.csv` | 
| Create daily measurements EEA stations | `02_scripts_clearning/01_build_eea_labels.py` | `data/processed/daily_avg/eea/pm_reference_stations_2024.csv` |

Annual aggregation means averaging daily values into one value per location for
the year. The current code uses 2024.

## 3. Reference And Sensor Labels

| Step | Folder/script  | Status |
| --- | --- | --- | --- |
| Tests whether averaging nearby Sensor.Community sensors makes them reliable enough. | `experiments/clustered_sensors/` | experimental |
| Older checks for input coverage, distance effects, and fold coverage. | `experiments/diagnostics/` | experimental |
| Tests where nearby UBA stations were used to correct SDS011 readings with OLS, Huber, and weather-aware models. |`experiments/nearby_reference_regression/` | experimental | 

Calibration means trying to correct a sensor so that its values match a trusted
reference measurement. We tested several Sensor.Community calibration ideas
because those sensors cover many more places than UBA stations. The result was
mostly negative: the tested methods did not give reference-quality annual
labels. See
[03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md).

Spatial folds are geographic train/validation splits. Here, German states are
grouped into folds, and `Sachsen-Anhalt` is treated as the sealed test state in
the current sensor-label workflow. The calibration attempt was unsuccesfull and 
the Sensor.Community data was excluded from training. 

## 4. Satellite Patch Collection

| Step | Folder/script | Output |
| --- | --- | --- | --- |
| Download Sentinel-2 and Sentinel-5P patches and Coperinus elevation for sensor labels | `04_GEE/01_download_eea_patches.py` | `data/processed/satellite_eea/<stream_folder>/<station_code>.npy`, `manifest_<stream>.csv` |
| Download the same data for test Laender in 10 km grids | `04_GEE/02_download_dense_grid_patches.py` | `data/processed/satellite_grid/<stream_folder>/<grid_id>.npy`, `data/processed/eea/grid_points.csv` | 

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
