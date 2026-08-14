# 02 Cleaning And Aggregation
This folder turns raw downloads into tables that later scripts can use.
Cleaning mostly means filtering impossible values, grouping readings by sensor
and time, and writing smaller files under `data/processed/`.
Run commands from the repository root.

Sensor.Community/UBA cleaning scripts live in `sensors-related/README.md`
(low-cost branch, not used in the final model). This page covers the EEA
label-building and socioeconomic scripts.

## Main Scripts
| Script | What it does | Main output |
| --- | --- | --- |
| `01_build_eea_labels.py` | Builds the EU-wide annual PM label file from the EEA raw parquet, one row per (station, valid day). | `data/processed/daily_avg/eea/pm_reference_stations_2024.csv` |
| `02_build_socioeconomic_kreis_2024.py` | Combines manually collected 2024 socioeconomic source files (population, median age, deaths, unemployment, wages) into a single Kreis-level table, joined on AGS. | `data/processed/socioeconomic/socioeconomic_kreis_2024.csv` |

## Usual Order
```bash
python3 02_scripts_cleaning/01_build_eea_labels.py
python3 02_scripts_cleaning/02_build_socioeconomic_kreis_2024.py
```

## Notes
- `01_build_eea_labels.py` keeps a station if PM10 or PM2.5 has at least
  `--min-frac` (default 90%) of the year's days valid, matching
  `count_eea_stations.py`'s threshold from `01_scripts_gathering`. Only valid,
  non-missing observations (`Validity > 0`, value not the -999 missing marker,
  value >= 0) are kept.
- `02_build_socioeconomic_kreis_2024.py` needs
  `data/processed/admin_boundaries/vg250_kreis.geojson` (from
  `01_scripts_gathering/sensors-related/04_get_admin_boundaries.py`) as the
  AGS/NUTS crosswalk, plus the raw Eurostat/VGRdL xlsx files under
  `data/processed/socioeconomic/`. Unemployment (Arbeitslosenquote) is
  downloaded fresh from the Bundesagentur für Arbeit rather than read from a
  local file. The wage column is gross wages per employee, not disposable
  household income -- named accordingly, not a drop-in income measure.

The next folder is [03_scripts_calibration](../03_scripts_calibration/README.md).
