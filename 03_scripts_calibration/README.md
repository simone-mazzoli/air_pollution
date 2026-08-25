# 03 Calibration

This folder is where the scripts are kept that we use in our attempt to calibrate the Sensor.Community data. 

Sensor.Community gives us many more locations than UBA stations, but the SDS011
PM sensors are low-cost devices. We tested whether their readings could be
corrected well enough to act like official annual PM10/PM2.5 measurements.

## What The Experiment Folders Contain

| Folder | Why we keep it |
| --- | --- |
| `experiments/clustered_sensors/` | Tests whether averaging nearby Sensor.Community sensors makes them reliable enough. |
| `experiments/diagnostics/` | Older checks for input coverage, distance effects, and fold coverage. |
| `experiments/nearby_reference_regression/` | Tests where nearby UBA stations were used to correct SDS011 readings with OLS, Huber, and weather-aware models. |

We keep these scripts because they support the results summarized in the report.
They are not needed for a normal current training run.

## Main Conclusion

The tested calibration approaches did not turn Sensor.Community measurements
into reliable reference-equivalent annual labels.

In plain language:

- Raw SDS011 readings have large errors and malfunction tails.
- Percentile/range mapping makes the national annual distribution look more
  plausible, but it does not prove each sensor is locally accurate.
- OLS and Huber regressions reduce raw error, but the annual labels become too
  close to a constant mean and lose most spatial variation.
- Weather-aware models show some daily signal in small close-reference subsets,
  especially for PM2.5, but they do not solve annual calibration.
- Clustered sensors may reduce random noise, but current results do not show
  that clusters can replace UBA stations.

Read
[SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md)
for the short result summary with numbers.

## Regenerating The Summary Table

The small CSV used for report writing is kept in Git:

```bash
python3 03_scripts_calibration/build_sensor_calibration_summary.py
```

It reads existing result files under `data/processed/calibration/` and the
clustered-sensor result CSV. It does not reprocess raw data or refit models.

## Data And Paths

Most calibration scripts expect `data/processed/` under the repository root.
The nearby-reference experiment scripts also support `--processed-dir` if your
processed data are somewhere else.

If `data/` is stored on an external drive, make sure the repository root still
has a `data/` folder or symlink.
