# 02 Cleaning And Aggregation

This folder turns raw downloads into tables that later scripts can use.
Cleaning mostly means filtering impossible values, grouping readings by sensor
and time, and writing smaller files under `data/processed/`.

Run commands from the repository root.

## Main Scripts

| Script | What it does | Main output |
| --- | --- | --- |
| `sensors-related/01_process_pm_sensors.py` | Cleans PM sensors, keeps Germany-bounding-box readings, and creates hourly/daily/monthly PM tables. | `data/processed/hourly/pm/`, `data/processed/daily_avg/`, `data/processed/monthly_avg/` |
| `sensors-related/02_process_humidity_sensors.py` | Cleans humidity/weather sensors and creates hourly/daily/monthly humidity tables. | `data/processed/hourly/humidity/`, `data/processed/daily_avg/humidity/`, `data/processed/monthly_avg/humidity/` |
| `sensors-related/03_aggregate_uba_monthly.py` | Aggregates daily UBA reference data to monthly station means. | `data/processed/monthly_avg/uba/pm_reference_stations_<YYYY-MM>.csv` |
| `sensors-related/04_locate_DEUB_UBA_stations.py` | Assigns each UBA station to a German federal state. | `data/processed/uba/station_land.csv` |
| `build_eea_pm_labels.py` | builds the EU-wide annual PM label file from the EEA raw parquet, one row per (station, valid day). | `data/processed/daily_avg/eea/pm_reference_stations_2024.csv` |


## Usual Order

```bash
python3 02_scripts_cleaning/sensors-related/01_process_pm_sensors.py
python3 02_scripts_cleaning/sensors-related/02_process_humidity_sensors.py
python3 02_scripts_cleaning/sensors-related/03_aggregate_uba_monthly.py
python3 02_scripts_cleaning/sensors-related/04_locate_DEUB_UBA_stations.py
```

For a smaller test run:

```bash
python3 02_scripts_cleaning/sensors-related/01_process_pm_sensors.py --months 2024-01 2024-02 --types sds011 --force
python3 02_scripts_cleaning/sensors-related/02_process_humidity_sensors.py --months 2024-01 2024-02 --types bme280 dht22 --force
```

Use `--no-merge` only if you do not want the combined monthly/hourly files that
later stages normally expect.

## A Few Terms

Annual aggregation means turning daily values into one value for the year.
Hourly, daily, and monthly aggregation are the same idea at shorter time scales.

For Sensor.Community SDS011 PM sensors, `P1` is treated as PM10 and `P2` as
PM2.5.

## Important Details

- PM cleaning drops non-positive readings, very high values above `1000`, and
  rows where `P2 > P1`.
- PM and humidity cleaning both require enough readings per hour/day/month so a
  single sparse sensor does not dominate.
- Humidity readings are also clipped at 90% to reduce the effect of saturated
  humidity sensors.
- UBA station-state assignment may fetch German state boundaries if
  `data/processed/germany_states.geojson` is missing.

The next folder is [03_scripts_calibration](../03_scripts_calibration/README.md).
