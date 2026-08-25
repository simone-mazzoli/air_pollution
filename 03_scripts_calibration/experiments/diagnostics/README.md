# Older Calibration Diagnostics

These scripts are useful checks, but they are not part of the normal current
training run.

```bash
python3 03_scripts_calibration/experiments/diagnostics/01_validate_new_months.py
python3 03_scripts_calibration/experiments/diagnostics/02_radius_distance_ablation.py
python3 03_scripts_calibration/experiments/diagnostics/03_check_land_coverage.py
```

## What do each of the diagnostic files do 
| File | What does it do | 
|--- | --- | 
|`01_validate_new_months.py`| Are all files in the correct format, and do the valuees make sense| 
|`02_radius_distance_ablation.py`| Whether distance to the nearest reference station affects calibration transfer.  Check if the distance to the nearest reference station affects calibration transfers | 
|`03_check_land_coverage.py`| Compute how covered each of the Folds are.| 

### Data 
They read `data/processed/` under the repository root.
