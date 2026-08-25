# Sensors-Related: Low-Cost Sensor Branch
This is the original, chronologically first data-gathering direction: volunteer
low-cost sensors (Sensor.Community) plus Germany-only official reference
stations (UBA). **Not used in the final model**. Low-cost sensors turned out
not usable even after calibration, and Germany-only reference stations (~400)
were too few on their own. See `../README.md` for the branch that replaced this
one (EEA reference stations).

## Main Scripts
| Script | What it downloads | Main output |
| --- | --- | --- |
| `01_sensor_community_all_sensors.py` | Monthly Sensor.Community ZIP archives for PM and humidity/weather sensors. | `data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip` |
| `02_uba_stations_metadata.py` | UBA station metadata for stations inside Germany. | `data/processed/uba_stations_germany.csv` |
| `03_download_uba_measurements.py` | Daily UBA PM10 and PM2.5 measurements for 2024. | `data/processed/daily_avg/uba/pm_reference_stations_2024.csv` |

UBA stations are official reference stations; Sensor.Community sensors are
volunteer low-cost sensors.

### Usual Order
```bash
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 01 02 03
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 04 05 06 --workers 2
python3 01_scripts_gathering/sensors-related/01_sensor_community_all_sensors.py 10 11 12 --types sds011 pms5003 dht22 bme280
python3 01_scripts_gathering/sensors-related/02_uba_stations_metadata.py
python3 01_scripts_gathering/sensors-related/03_download_uba_measurements.py
```
The Sensor.Community downloader expects months as `01` through `12`. The year is
currently fixed to 2024 in the script.

## Other Scripts
| Script | Status | Notes |
| --- | --- | --- |
| `04_get_admin_boundaries.py` | optional | Downloads administrative boundaries for later spatial joins. |
| `05_download_extract_hyras.py` | experimental | Downloads/extracts HYRAS weather for calibration diagnostics. This can download large NetCDF files. |
