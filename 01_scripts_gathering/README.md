# Data Gathering

## Purpose

This directory gathers the external tabular data used by the pipeline: Sensor.Community monthly sensor archives and UBA reference-station metadata and PM measurements for 2024.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `01_sensor_community_all_sensors.py` | Downloads monthly Sensor.Community ZIP archives, with resume support and ZIP validation. Includes PM and humidity sensor types by default. | Sensor.Community archive at `https://archive.sensor.community/csv_per_month/`; month arguments; optional sensor-type arguments. | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` | Main pipeline step |
| `02_uba_stations_metadata.py` | Fetches UBA air-quality station metadata and keeps stations inside the Germany bounding box that were active during 2024. | UBA stations API. | `data/processed/uba_stations_germany.csv` | Main pipeline step |
| `03_download_uba_measurements.py` | Downloads daily PM10 and PM2.5 reference measurements for the filtered UBA station list, then pivots them to one row per station/date. | `data/processed/uba_stations_germany.csv`; UBA measures CSV API. | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` | Main pipeline step |

## Pipeline position

Run this stage before cleaning and calibration. The usual order is:

1. `01_sensor_community_all_sensors.py` for the required months and sensor types.
2. `02_uba_stations_metadata.py`.
3. `03_download_uba_measurements.py`.

The next stage is [`../02_scripts_cleaning`](../02_scripts_cleaning/README.md), which aggregates Sensor.Community archives and UBA daily reference values.

## Data flow

Sensor.Community monthly archive URLs -> local ZIP files under `data/raw/`.

UBA station API -> Germany-bounded active station list -> UBA daily PM10/PM2.5 values -> wide daily station table.

## Running the scripts

```bash
python 01_scripts_gathering/01_sensor_community_all_sensors.py 01 02 03
python 01_scripts_gathering/01_sensor_community_all_sensors.py 04 05 06 --workers 2
python 01_scripts_gathering/01_sensor_community_all_sensors.py 10 11 12 --types sds011 pms5003 dht22 bme280
python 01_scripts_gathering/02_uba_stations_metadata.py
python 01_scripts_gathering/03_download_uba_measurements.py
```

`01_sensor_community_all_sensors.py` accepts months as `01` through `12`, `--types`, and `--workers`. The UBA scripts do not define command-line arguments; their year and paths are constants in the files.

## Important assumptions and caveats

- The year is hard-coded to `2024` in all three scripts.
- The repository ignores `data/`, so raw and processed outputs may live outside this checkout or on an external drive.
- `02_uba_stations_metadata.py` filters stations to a shared Germany bounding box: latitude `47.2` to `55.1`, longitude `5.8` to `15.1`.
- `03_download_uba_measurements.py` queries UBA component id `1` for PM10 and `9` for PM2.5 using daily mean scope `1`.
- `03_download_uba_measurements.py` skips re-download if its final CSV already exists.
- An exception message in `03_download_uba_measurements.py` says to run `get_uba_stations.py`; the current script name is `02_uba_stations_metadata.py`.
