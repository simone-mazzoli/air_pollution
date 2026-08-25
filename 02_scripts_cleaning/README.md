# 02 Cleaning And Aggregation
This folder turns raw downloads into tables that later scripts can use.
Cleaning mostly means filtering impossible values, grouping readings by sensor
and time, and writing smaller files under `data/processed/`.
Run commands from the repository root.

Sensor.Community/UBA cleaning scripts live in `sensors-related`
(low-cost branch, not used in the final model). This page covers the EEA
label-building and socioeconomic scripts.

## Main Scripts
| Script | What it does | Main output |
| --- | --- | --- |
| `01_build_eea_labels.py` | Builds the EU-wide annual PM label file from the EEA raw parquet, one row per (station, valid day). | `data/processed/daily_avg/eea/pm_reference_stations_2024.csv` |

## Usual Order
```bash
python3 02_scripts_cleaning/01_build_eea_labels.py
```

## Notes
- `01_build_eea_labels.py` keeps a station if PM10 or PM2.5 has at least
  `--min-frac` (default 90%) of the year's days valid, matching
  `count_eea_stations.py`'s threshold from `01_scripts_gathering`. Only valid,
  non-missing observations (`Validity > 0`, value not the -999 missing marker,
  value >= 0) are kept.
