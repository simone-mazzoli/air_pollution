# 01 Data Gathering
This folder downloads the outside data we need before any modeling can happen.
Most outputs go under `data/`, which is not committed to Git.
Run commands from the repository root.

## Two Branches
This folder has two separate data-gathering paths that don't feed each other:
- **Low-cost sensors** (Sensor.Community + UBA-Germany-only) — the original,
  chronologically first direction. Not used in the final model. See
  `sensors-related/README.md`.
- **EEA reference stations** (Europe-wide, via `airbase`) — what the final
  model actually trains and validates on. Documented below.

## EEA Reference Station Branch (used in the final model)
Reference-grade PM10/PM2.5 stations across all EEA-reporting countries, via the
`airbase` package.

| Script | What it does | Main output |
| --- | --- | --- |
| `01_get_eea_pm.py` | Downloads verified PM10/PM2.5 parquet files per country via `airbase`. | `data/processed/eea/airbase_raw/<CC>/*.parquet` |
| `02_count_eea_stations.py` | Filters to stations with at least 90% valid days in the year; pulls station coordinates from the `airbase` metadata. | `data/processed/eea/stations_to_download.csv` |

`02_count_eea_stations.py`'s output (`stations_to_download.csv`) is the input
list for the patch downloader (documented elsewhere).

### Usual Order
```bash
python3 01_scripts_gathering/01_get_eea_pm.py
python3 01_scripts_gathering/02_count_eea_stations.py
```
`01_get_eea_pm.py --country` defaults to all EEA-reporting countries; pass
specific 2-letter codes (e.g. `--country DE FR`) to restrict it.

## Data Folder
These scripts write into `data/raw/` and `data/processed/` relative to the
repository. If your data are on an external drive, make `data/` a symlink or
mount/link that points there. Do not commit the downloaded ZIPs or generated
processed tables.
