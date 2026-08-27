# PM2.5 completeness data-quality check

This supplementary check compares the saved PM2.5 model station set with PM2.5
daily completeness in the processed EEA labels. It does not change
preprocessing, folds, model selection, TEST membership, predictions or
checkpoints.

## Inputs

1. Daily EEA labels: `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`.
2. Development predictions: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv`.
3. TEST predictions: `06_models/results/cnn_deep_wide/test_predictions.csv`.
4. EEA station metadata: `data/processed/eea/airbase_raw/metadata.csv`.
5. Satellite arrays under `data/processed/satellite_eea/`.

## Thresholds

The implemented preprocessing threshold was `int(0.90 * 366)`, or at least
329 valid days. A literal 90 percent threshold for leap-year 2024 would require
at least 330 valid days.

The station-level file names the implemented boolean
`passes_implemented_329_day_threshold`. The stricter comparison is named
`passes_literal_90pct_330_day_threshold`.

## Results

The saved model station set contains 1,869 development stations and 164 sealed
TEST stations, with no station-code overlap.

Stations below the implemented 329-day threshold: 108 total, including 96
development stations and 12 TEST stations. These stations passed the original
joint PM10/PM2.5 preprocessing rule through PM10 completeness and still had a
non-missing annual PM2.5 label.

Stations below the literal 330-day threshold: 111 total, including 99
development stations and 12 TEST stations. Three development stations had
exactly 329 valid PM2.5 days; no TEST station had exactly 329 valid PM2.5 days.

Recomputed annual PM2.5 means from the daily file match the saved model targets;
the largest absolute difference was below `0.000002` ug/m3. This is therefore a
completeness-threshold issue, not a target-recalculation issue.

## Sensitivity

Using saved predictions only, a strict `>=330` PM2.5-day subset gives:

| split | all stations | strict `>=330` stations | all RMSE | strict RMSE | relative change |
| --- | ---: | ---: | ---: | ---: | ---: |
| development CV | 1,869 | 1,770 | 3.822866 | 3.757936 | -1.70% |
| sealed TEST | 164 | 152 | 2.374988 | 2.396637 | +0.91% |

This is a saved-prediction sensitivity analysis. It is not a retraining result
on a different station set.

## Outputs

- `analysis_outputs/data_quality/pm25_completeness/pm25_completeness_by_station.csv`
- `analysis_outputs/data_quality/pm25_completeness/pm25_completeness_split_summary.csv`
