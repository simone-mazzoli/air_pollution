# Older Calibration Diagnostics

These scripts are useful checks, but they are not part of the normal current
training run.

```bash
python3 03_scripts_calibration/experiments/diagnostics/01_validate_new_months.py
python3 03_scripts_calibration/experiments/diagnostics/02_radius_distance_ablation.py
python3 03_scripts_calibration/experiments/diagnostics/05_check_land_coverage.py
```

They read `data/processed/` under the repository root.
