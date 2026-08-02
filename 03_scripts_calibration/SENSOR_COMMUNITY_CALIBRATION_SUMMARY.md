# Sensor.Community Calibration Summary

This is the short version of what we learned from the Sensor.Community
calibration work.

## Question

Can we turn 2024 Sensor.Community SDS011 PM readings into annual PM10/PM2.5
labels that are close enough to official UBA reference measurements?

Answer: not with the approaches tested here.

## Setup

The main comparison used:

- SDS011 PM sensors only;
- 2024 data;
- at least 18 valid hours per sensor-day;
- at least 182 valid days per sensor-year;
- German-state-based validation folds;
- `Sachsen-Anhalt` kept out of fitting/selection in the current sensor-label
  workflow;
- station-balanced annual errors, so one UBA station with many nearby sensors
  does not dominate the score.

A reference station is an official monitoring station. A held-out fold is a
region we do not train on, then use for validation.

## Approaches We Tried

- Raw SDS011 annual means.
- Percentile/range mapping, which maps the sensor network into a more plausible
  national annual distribution.
- Nearby-reference OLS and Huber regression, which fit one global correction
  from nearby SDS011-UBA daily pairs.
- Daily weather-aware regression with humidity and temperature.
- Background-station-only sensitivity checks.
- Clustered-sensor means and medians.
- A constant-mean baseline.

OLS means ordinary least squares regression. Huber is a regression method that
down-weights large outliers. The constant-mean baseline predicts almost the same
annual value everywhere; it is not useful as a model target, but it tells us how
hard it is to beat a very simple annual prediction.

## Result Table

The generated CSV has one row per main approach:
`sensor_community_calibration_summary.csv`.

| Approach | Pollutant | Level | Headline result | Beat baseline? | What we do with it |
| --- | --- | --- | --- | --- | --- |
| raw SDS011 baseline | PM10 | annual | mean RMSE 46.310 | no | diagnostic only |
| raw SDS011 baseline | PM2.5 | annual | mean RMSE 26.959 | no | diagnostic only |
| constant-mean baseline | PM10 | annual | mean RMSE 2.911 | baseline | comparison only |
| constant-mean baseline | PM2.5 | annual | mean RMSE 1.836 | baseline | comparison only |
| national percentile/range mapping | PM10 | annual | mean RMSE 20.860 | no | possible proxy label, not reference-grade |
| national percentile/range mapping | PM2.5 | annual | mean RMSE 12.559 | no | possible proxy label, not reference-grade |
| OLS regression adjustment | PM10 | annual | mean RMSE 3.098 | no | sensitivity experiment |
| OLS regression adjustment | PM2.5 | annual | mean RMSE 1.919 | no | sensitivity experiment |
| Huber regression adjustment | PM10 | annual | mean RMSE 3.157 | no | kept for the report |
| Huber regression adjustment | PM2.5 | annual | mean RMSE 2.169 | no | kept for the report |
| daily weather-aware regression | PM10 | daily | RMSE improvement vs constant 2.523 | yes | daily diagnostic only |
| daily weather-aware regression | PM2.5 | daily | RMSE improvement vs constant 2.272 | yes | daily diagnostic only |
| background-station-only sensitivity | PM10 | annual | RMSE improvement vs constant -0.395 | no | sensitivity experiment |
| background-station-only sensitivity | PM2.5 | annual | RMSE improvement vs constant -0.196 | no | sensitivity experiment |
| clustered-sensor mean | PM10 | annual | RMSE 3.108 | no | diagnostic only |
| clustered-sensor median | PM10 | annual | RMSE 3.172 | no | diagnostic only |

## What The Numbers Mean

Raw SDS011 values are too noisy. They have weak annual rank correlation with UBA
stations: about `0.18` for PM10 and `0.19` for PM2.5 at 20 km. They also have
very large error and malfunction tails.

Percentile/range mapping gives a more believable national annual distribution.
That is useful, but it is not the same as local calibration. It does not prove
that one sensor's annual value is correct at its location.

OLS and Huber regression reduce the huge raw sensor error, but mostly by
shrinking predictions toward the annual mean. OLS at 20 km keeps only about
`15%` to `16%` of the UBA station standard deviation and still does not beat the
constant-mean baseline.

Weather-aware daily models show that there is some daily information in close
sensor-reference pairs. That does not become a convincing annual calibration.

Clustered sensors may reduce random device noise, but the current cluster tests
still have negative baseline skill. They do not show that grouped low-cost
sensors can replace reference stations.

## Remaining Possible Use

Sensor.Community data may still help as weak labels. Here, a weak label means a
noisy extra training target. The claim would be: adding these noisy labels helps
the model predict completely held-out UBA stations. That still needs to be shown
with model validation and should not be described as successful sensor
calibration.

## How To Rebuild The CSV

```bash
python3 03_scripts_calibration/build_sensor_calibration_summary.py
```

The script reads:

- `data/processed/calibration/regression_reference_adjustment/method_comparison/aggregate_metrics.csv`
- `data/processed/calibration/regression_reference_adjustment/method_comparison/signal_preservation_metrics.csv`
- `data/processed/calibration/regression_reference_adjustment/close_reference_weather_models/results.csv`
- `03_scripts_calibration/experiments/clustered_sensors/cluster_test_results.csv`
