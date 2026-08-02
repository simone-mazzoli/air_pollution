# 01 Data Gathering

This folder downloads the outside data we need before any modeling can happen.
Most outputs go under `data/`, which is not committed to Git.

Run commands from the repository root.

## Main Scripts

| Script | What it downloads | Main output |
| --- | --- | --- |
| `sensors-related/01_sensor_community_all_sensors.py` | Monthly Sensor.Community ZIP archives for PM and humidity/weather sensors. | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` |
| `sensors-related/02_uba_stations_metadata.py` | UBA station metadata for stations inside Germany. | `data/processed/uba_stations_germany.csv` |
| `sensors-related/03_download_uba_measurements.py` | Daily UBA PM10 and PM2.5 measurements for 2024. | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` |

UBA stations are official reference stations. Sensor.Community sensors are
volunteer low-cost sensors. We download both because UBA is reliable and
Sensor.Community is much denser.

## Usual Order

```bash
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 01 02 03
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 04 05 06 --workers 2
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 10 11 12 --types sds011 pms5003 dht22 bme280

python3 01_scripts_gathering/sensors-related/02_uba_stations_metadata.py
python3 01_scripts_gathering/sensors-related/03_download_uba_measurements.py
```

The Sensor.Community downloader expects months as `01` through `12`. The year is
currently fixed to 2024 in the script.

## Other Data Scripts

| Script | Status | Notes |
| --- | --- | --- |
| `get_eea_pm.py` | experimental | Downloads EEA PM data. Example: `python3 01_scripts_gathering/get_eea_pm.py --country DE --year 2024`. |
| `download_extract_hyras.py` | experimental | Downloads/extracts HYRAS weather for calibration diagnostics. This can download large NetCDF files. |
| `sensors-related/04_get_admin_boundaries.py` | optional | Downloads administrative boundaries for later spatial joins. |
| `sensors-related/05_get_socioeconomic_data_PRELIMINARY.py` | early | Preliminary socioeconomic data fetch. Not part of the current model run. |

## Data Folder

These scripts write into `data/raw/` and `data/processed/` relative to the
repository. If your data are on an external drive, make `data/` a symlink or
mount/link that points there. Do not commit the downloaded ZIPs or generated
processed tables.
