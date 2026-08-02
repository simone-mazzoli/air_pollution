# Cleaning and Aggregation

## Purpose

This directory converts raw Sensor.Community and UBA downloads into analysis-ready hourly, daily, and monthly tables. It also prepares UBA station state assignments needed by calibration.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `01_process_pm_sensors.py` | Cleans PM sensor archives, filters to Germany, applies plausibility checks, aggregates to hourly/daily/monthly, and merges PM sensor types by priority. | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` for PM sensors. | `data/processed/hourly/pm/<sensor_type>_<YYYY-MM>.parquet`; `data/processed/hourly/pm/nodes/<sensor_type>_<YYYY-MM>.parquet`; `data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet`; `data/processed/daily_avg/<sensor_type>_<YYYY-MM>.csv`; `data/processed/monthly_avg/<sensor_type>_<YYYY-MM>.csv`; `data/processed/monthly_avg/all_pm_sensors/<YYYY-MM>.csv` | Main pipeline step |
| `02_process_humidity_sensors.py` | Cleans humidity/weather archives, derives clipped humidity features, aggregates to hourly/daily/monthly, and merges humidity sensor types by priority. | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` for `bme280` and `dht22`. | `data/processed/hourly/humidity/<sensor_type>_<YYYY-MM>.parquet`; `data/processed/hourly/humidity/nodes/<sensor_type>_<YYYY-MM>.parquet`; `data/processed/hourly/humidity/all_sensors/<YYYY-MM>.parquet`; `data/processed/daily_avg/humidity/<sensor_type>_<YYYY-MM>.csv`; `data/processed/monthly_avg/humidity/<sensor_type>_<YYYY-MM>.csv`; `data/processed/monthly_avg/humidity/all_sensors/<YYYY-MM>.csv` | Main pipeline step |
| `03_aggregate_uba_monthly.py` | Aggregates UBA daily PM reference data to one monthly file per month. | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` | `data/processed/monthly_avg/uba/pm_reference_stations_<YYYY-MM>.csv` | Main pipeline step |
| `04_locate_DEUB_UBA_stations.py` | Resolves UBA station codes to German state names, using coordinates for federal or non-state-coded stations. | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv`; `data/processed/germany_states.geojson` or fetched GeoJSON. | `data/processed/uba/station_land.csv` | Main pipeline step for calibration |

## Pipeline position

Run this after [`../01_scripts_gathering`](../01_scripts_gathering/README.md):

1. `01_process_pm_sensors.py`.
2. `02_process_humidity_sensors.py`.
3. `03_aggregate_uba_monthly.py`.
4. `04_locate_DEUB_UBA_stations.py`.

The next stage is [`../03_scripts_calibration`](../03_scripts_calibration/README.md), which validates hourly files, calibrates annual PM targets, and assigns sensor states.

## Data flow

Raw Sensor.Community PM readings -> Germany-bounded plausible readings -> node-hour means -> node-day means -> node-month means -> merged all-PM-sensor monthly/hourly tables.

Raw humidity/weather readings -> Germany-bounded range-filtered readings -> `humidity_clip90` and high-humidity flags -> node-hour/day/month humidity tables -> merged all-humidity-sensor tables.

UBA daily PM station measurements -> monthly station means with separate PM10 and PM2.5 day counts.

UBA station codes and coordinates -> station-to-Land lookup for leave-one-Land-out calibration.

## Running the scripts

```bash
python 02_scripts_cleaning/01_process_pm_sensors.py
python 02_scripts_cleaning/01_process_pm_sensors.py --months 2024-01 2024-02 --types sds011 sps30 --force
python 02_scripts_cleaning/01_process_pm_sensors.py --months 2024-01 --no-merge

python 02_scripts_cleaning/02_process_humidity_sensors.py
python 02_scripts_cleaning/02_process_humidity_sensors.py --months 2024-01 2024-02 --types bme280 dht22 --force
python 02_scripts_cleaning/02_process_humidity_sensors.py --months 2024-01 --no-merge

python 02_scripts_cleaning/03_aggregate_uba_monthly.py
python 02_scripts_cleaning/04_locate_DEUB_UBA_stations.py
```

## Important assumptions and caveats

- PM columns are `P1` for PM10 and `P2` for PM2.5.
- PM cleaning requires both `P1` and `P2`, drops rows outside the Germany bounding box, drops non-positive or very high values above `1000`, and drops rows where `P2 > P1`.
- PM coverage thresholds are at least `5` readings per hour, `18` hours per day, and `10` days per month.
- PM sensor-type merge priority is `sds011`, `sps30`, `pms7003`, `pms5003`, `pms6003`, `pms3003`, `pms1003`.
- Humidity coverage thresholds match the PM script. `bme280` is preferred over `dht22` during merged humidity output creation.
- Humidity values are clipped at `90%` per reading before aggregation into `humidity_clip90`.
- Timestamps are treated as naive UTC in Sensor.Community cleaning. Later calibration shifts them by one hour to match UBA MEZ.
- `04_locate_DEUB_UBA_stations.py` can fetch and cache `data/processed/germany_states.geojson` if it is missing.
