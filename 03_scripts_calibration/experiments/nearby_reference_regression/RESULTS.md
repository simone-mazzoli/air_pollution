# PM Label-Construction Results

Status: This file documents an intermediate annual calibration comparison. The
original label recommendation was superseded by the later weather-aware,
background-station and clustered-sensor checks. The final project does not use
Sensor.Community measurements as CNN labels. See
`../../SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md`.

## Run Summary

The 2024 PM label comparison used the corrected sensor-to-Land assignments and
all merged hourly PM months, `2024-01` through `2024-12`.

Core settings:

- sensor filter: SDS011 only
- daily completeness: at least `18` hourly observations per sensor-day
- annual completeness: at least `182` valid days per sensor-year
- sealed test Land: `Sachsen-Anhalt`
- inner validation: leave-one-fold-out over non-Sachsen-Anhalt folds
- matching radii compared: `5`, `10`, `20` km

The corrected `sensor_land.csv` policy assigned sensors only when they were
covered by a German Land polygon or within a `1 km` near-boundary tolerance.
Points farther outside Germany were excluded.

State-assignment counts:

- direct polygon assignments: `6,009`
- accepted boundary fallbacks: `41`
- excluded outside-Germany sensors: `1,312`
- final assigned SDS011 locations: `6,050`

## Methods Compared

The fair comparison harness evaluated each method on the same held-out annual
sensor-station rows for each radius, fold, and pollutant:

- `raw SDS011 baseline`
- `constant-mean baseline`
- `original percentile-mapping method`
- `OLS regression adjustment`
- `Huber regression adjustment`

The original percentile-mapping method was reproduced inside the common
corrected SDS011-only fold harness. Existing percentile-output summary numbers
were not used directly because the older script was not fully comparable: it
used a 12-hour daily threshold and did not filter `sensor_type` before daily
aggregation.

Main evaluation level:

1. build daily nearby sensor-UBA pairs;
2. aggregate to annual sensor-station values;
3. compute station-balanced annual metrics so each held-out UBA station has
   equal total weight.

## Held-Out Reference Accuracy

Combined across PM10 and PM2.5, the constant-mean baseline had the lowest
held-out RMSE. This is the key caution: low RMSE can be achieved by predicting
almost the same value everywhere.

| Method | Radius | Mean RMSE | Mean MAE | Folds better than raw | Folds better than constant mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant-mean baseline | 5 km | `2.319` | `1.732` | `22 / 22` | `0 / 22` |
| constant-mean baseline | 10 km | `2.350` | `1.754` | `22 / 22` | `0 / 22` |
| constant-mean baseline | 20 km | `2.374` | `1.785` | `22 / 22` | `0 / 22` |
| OLS regression adjustment | 20 km | `2.508` | `1.867` | `22 / 22` | `2 / 22` |
| OLS regression adjustment | 5 km | `2.511` | `1.879` | `22 / 22` | `7 / 22` |
| OLS regression adjustment | 10 km | `2.577` | `1.906` | `22 / 22` | `4 / 22` |
| Huber regression adjustment | 5 km | `2.591` | `1.845` | `22 / 22` | `9 / 22` |
| Huber regression adjustment | 20 km | `2.663` | `1.951` | `22 / 22` | `4 / 22` |
| Huber regression adjustment | 10 km | `2.696` | `1.929` | `22 / 22` | `4 / 22` |
| original percentile-mapping method | 5 km | `15.044` | `4.803` | `22 / 22` | `0 / 22` |
| original percentile-mapping method | 20 km | `16.710` | `4.445` | `22 / 22` | `0 / 22` |
| original percentile-mapping method | 10 km | `16.987` | `5.210` | `22 / 22` | `0 / 22` |
| raw SDS011 baseline | 5 km | `32.990` | `10.495` | `0 / 22` | `0 / 22` |
| raw SDS011 baseline | 20 km | `36.635` | `9.432` | `0 / 22` | `0 / 22` |
| raw SDS011 baseline | 10 km | `36.965` | `11.190` | `0 / 22` | `0 / 22` |

At the currently selected OLS 20 km setting:

- raw SDS011 baseline RMSE: `36.635`
- constant-mean baseline RMSE: `2.374`
- OLS regression adjustment RMSE: `2.508`
- OLS 20 km improvement over constant mean: `-0.135`

OLS 20 km improves strongly over raw SDS011, but it does not meaningfully beat
the constant-mean baseline.

## Spatial-Signal Preservation

All affine methods are monotone transforms of raw annual SDS011 values, so their
rank correlation with held-out UBA references is the same as raw SDS011 within a
given radius. The difference is how much target variation they preserve.

At 20 km:

| Method | Pollutant | Spearman with UBA | Prediction/reference std ratio | Prediction interdecile range | Predicted high-low contrast |
| --- | --- | ---: | ---: | ---: | ---: |
| raw SDS011 baseline | PM10 | `0.179` | `6.517` | `21.805` | `3.756` |
| original percentile-mapping method | PM10 | `0.179` | `2.917` | `9.750` | `1.717` |
| Huber regression adjustment | PM10 | `0.179` | `0.172` | `0.578` | `0.099` |
| OLS regression adjustment | PM10 | `0.179` | `0.146` | `0.492` | `0.088` |
| constant-mean baseline | PM10 | undefined | `0.000` | `0.000` | `0.000` |
| raw SDS011 baseline | PM2.5 | `0.194` | `7.892` | `13.980` | `-0.211` |
| original percentile-mapping method | PM2.5 | `0.194` | `3.656` | `6.455` | `-0.108` |
| Huber regression adjustment | PM2.5 | `0.194` | `0.258` | `0.463` | `-0.003` |
| OLS regression adjustment | PM2.5 | `0.194` | `0.160` | `0.283` | `-0.003` |
| constant-mean baseline | PM2.5 | undefined | `0.000` | `0.000` | `0.000` |

The regression adjustments are close to constant annual labels. OLS 20 km keeps
only about `15%` to `16%` of the held-out UBA station standard deviation and
nearly eliminates the high-versus-low concentration contrast.

## Annual CNN-Label Distributions

Final non-Sachsen-Anhalt annual label distributions at 20 km:

| Method | Pollutant | Retained sensors | Mean | Std | P10 | Median | P90 | Interdecile range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw SDS011 baseline | PM10 | `4,245` | `13.593` | `31.731` | `3.798` | `9.814` | `17.573` | `13.775` |
| UBA annual reference distribution | PM10 | `351` | `13.551` | `2.664` | `10.292` | `13.403` | `16.907` | `6.615` |
| original percentile-mapping method | PM10 | `4,194` | `13.880` | `3.623` | `10.630` | `13.379` | `16.709` | `6.080` |
| OLS regression adjustment | PM10 | `4,245` | `14.077` | `0.707` | `13.859` | `13.993` | `14.166` | `0.307` |
| Huber regression adjustment | PM10 | `4,245` | `12.963` | `0.820` | `12.710` | `12.865` | `13.066` | `0.356` |
| raw SDS011 baseline | PM2.5 | `4,245` | `7.714` | `24.529` | `2.098` | `5.254` | `9.108` | `7.010` |
| UBA annual reference distribution | PM2.5 | `319` | `8.407` | `1.505` | `6.669` | `8.392` | `10.077` | `3.408` |
| original percentile-mapping method | PM2.5 | `4,220` | `8.822` | `3.021` | `6.952` | `8.393` | `10.087` | `3.135` |
| OLS regression adjustment | PM2.5 | `4,245` | `8.514` | `0.480` | `8.404` | `8.466` | `8.541` | `0.137` |
| Huber regression adjustment | PM2.5 | `4,245` | `7.768` | `0.736` | `7.599` | `7.694` | `7.810` | `0.210` |

The original percentile-mapping method preserves a plausible annual target
range close to the UBA annual station distribution. The regression adjustments
compress annual labels heavily.

## Recommendation

Decision categories:

| Candidate | Category | Reason |
| --- | --- | --- |
| original percentile-mapping method | best range-preserving candidate within this sub-experiment; not used for final model labels | Preserves plausible annual target range and spatial label variation, despite weak paired held-out RMSE. |
| OLS regression adjustment, 20 km | compressed sensitivity candidate within this sub-experiment; not used for final model labels | Useful as a low-error, regression-to-the-mean sensitivity label set, but not suitable as main labels because it does not beat constant mean and strongly compresses targets. |
| other OLS/Huber regression adjustments | not recommended due to target compression | They improve over raw SDS011 but remain close to constant-mean predictions and preserve little annual variation. |
| constant-mean baseline | not recommended due to target compression | Best RMSE, but no spatial signal and not a usable CNN target. |
| raw SDS011 baseline | not recommended due to error | Preserves rank and variation, but has very large held-out error and malfunction tails. |

Answer to the main question:

- OLS 20 km should not be treated as the main label set.
- The original percentile-mapping method is the best range-preserving candidate
  within this sub-experiment, but it is not used for final CNN training.
- OLS 20 km is a compressed, reference-agreement-oriented candidate within this
  sub-experiment.
- The constant-mean baseline shows that regression RMSE improvements over raw
  are misleading unless target variation and rank preservation are checked.

## Key Output Files

- `data/processed/calibration/regression_reference_adjustment/method_comparison/fold_metrics.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/aggregate_metrics.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/signal_preservation_metrics.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/annual_label_distributions.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/method_rankings.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/comparison_summary.md`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/comparison_metadata.json`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/plots/`
