# Calibration Scripts

This is a quick guide to what each calibration script is for.

## Scripts Used By The Current Pipeline

| Script | What it does | Main output |
| --- | --- | --- |
| `active/04_resolve_sensor_land.py` | Assigns SDS011 sensors to German federal states. | `data/processed/sensor_land.csv` |
| `active/03_calibrate_pm_loo.py` | Builds the current annual Sensor.Community proxy labels. | `data/processed/corrected/fold/*/annual/2024.csv` |
| `build_sensor_calibration_summary.py` | Rebuilds the small CSV table used in the report. | `sensor_community_calibration_summary.csv` |

## Experiments We Keep For The Report

| Script | Question it tested | Result |
| --- | --- | --- |
| `experiments/nearby_reference_regression/calibrate_pm_regression_loo.py` | Can nearby UBA stations correct SDS011 annual labels with OLS or Huber regression? | Error drops a lot compared with raw sensors, but annual spatial variation collapses. |
| `experiments/nearby_reference_regression/compare_label_construction_methods.py` | How do raw, constant mean, percentile mapping, OLS, and Huber compare on the same folds? | Constant mean has the best annual RMSE; regression does not beat it. |
| `experiments/nearby_reference_regression/fit_close_reference_weather_models.py` | Does adding weather help close-reference calibration? | Some daily improvements, but no convincing annual calibration. |
| `experiments/clustered_sensors/cluster_test.py` | Does averaging nearby sensors make them good enough? | Current cluster results still do not beat the baseline. |

## Older Diagnostic Scripts

| Script | What we used it to check |
| --- | --- |
| `experiments/nearby_reference_regression/eda_close_reference_inventory.py` | How many SDS011 sensors are close enough to UBA stations, and what weather data are nearby. |
| `experiments/nearby_reference_regression/audit_2024_data_completeness.py` | Whether the 2024 inputs had enough complete rows for the regression experiments. |
| `experiments/diagnostics/01_validate_new_months.py` | Whether newly processed monthly files have expected columns and coverage. |
| `experiments/diagnostics/02_radius_distance_ablation.py` | Whether distance to the nearest reference station affects calibration transfer. |
| `experiments/diagnostics/05_check_land_coverage.py` | How much sensor coverage each geographic fold has. |
