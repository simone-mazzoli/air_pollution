# Active Calibration Scripts

These are the calibration scripts still used by the current pipeline.

```bash
python3 03_scripts_calibration/active/04_resolve_sensor_land.py
python3 03_scripts_calibration/active/03_calibrate_pm_loo.py
```

`04_resolve_sensor_land.py` writes `data/processed/sensor_land.csv`.
`03_calibrate_pm_loo.py` writes annual label files under
`data/processed/corrected/fold/`.

Most teammates should start with the parent
[calibration README](../README.md), then come here only if they need the exact
active scripts.
