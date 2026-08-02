# Regression Reference Adjustment

## Purpose

This directory contains a parallel PM calibration branch that prepares corrected
annual PM labels for later CNN training/evaluation. It does not train the CNN.

The original pipeline in `../03_calibrate_pm_loo.py` remains the percentile/range
mapping branch and writes to:

- `data/processed/calibration/linear_by_fold.json`
- `data/processed/calibration/linear_test_land.json`
- `data/processed/corrected/fold/`

This branch fits regressions between nearby matched SDS011
sensor-days and UBA reference-station days:

```text
corrected PM = intercept + slope * raw SDS011 PM
```

The regression branch is isolated under:

- `data/processed/calibration/regression_reference_adjustment/`
- `data/processed/corrected/regression_reference_adjustment/`

The isolation is deliberate: these outputs must not overwrite the established
percentile-calibration outputs.

The regression adjustment is now treated as an alternative and sensitivity
branch. The current scientific comparison favors the original percentile-mapping
method for main CNN labels because it preserves a plausible annual target range,
while the regression labels are useful mainly for checking behavior under a
low-error but compressed target.

## High-Level Flow

```text
Merged hourly SDS011 data
-> daily means on fixed MEZ dates
-> sensor and UBA Land/fold assignment
-> nearby sensor-station matching
-> inner leave-one-fold-out OLS/Huber evaluation
-> shared method/radius selection
-> final non-Sachsen-Anhalt calibration fit
-> annual train and Sachsen-Anhalt test labels
```

A separate comparison harness in
`compare_label_construction_methods.py` evaluates the regression branch against
the raw SDS011 baseline, a constant-mean baseline, and the original
percentile-mapping method under the same corrected folds and held-out
sensor-station rows.

## Inputs

Required files under `data/processed/` by default:

| Input | Purpose |
| --- | --- |
| `hourly/pm/all_pm_sensors/<YYYY-MM>.parquet` | Merged hourly PM means. Must contain `location`, `hour`, `P1`, `P2`, and `sensor_type`. |
| `hourly/pm/nodes/sds011_<YYYY-MM>.parquet` | SDS011 sensor coordinates. |
| `daily_avg/uba/pm_reference_stations_<YEAR>.csv` | Daily UBA PM10 and PM2.5 reference measurements. |
| `uba/station_land.csv` | UBA station-to-Land lookup. |
| `sensor_land.csv` | Low-cost sensor-to-Land lookup. |

Use `--processed-dir` when processed data live outside the checkout. In this
project, `data/` is commonly an ignored symlink to an external drive.

## Calibration Logic

The script loads the merged hourly PM files and filters to SDS011 before any
daily aggregation. `sensor_type` is normalized with whitespace stripping and
lowercasing, so labels such as `SDS011` and ` sds011 ` are treated consistently.

Hourly rows are treated as valid hourly means from
`../../02_scripts_cleaning/01_process_pm_sensors.py`. The script shifts UTC
hours by one fixed hour to align with UBA MEZ dates, then keeps only sensor-days
with at least `18` valid hourly observations. Annual labels require at least
`182` valid days.

For each pollutant:

- PM10 uses SDS011 `P1` and UBA `PM10`.
- PM2.5 uses SDS011 `P2` and UBA `PM2.5`.

Nearby matched sensors are linked to the nearest UBA station within the requested
radius. These are geographic nearest-neighbor matches, not co-located regulatory
collocations. Typical comparison radii are `5`, `10`, and `20` km.

Corrected daily values are clipped at zero. Annual outputs preserve the existing
malfunction sanity policy: labels are dropped if any corrected annual pollutant
exceeds `50 ug/m3`.

## OLS And Huber

Both methods fit the same equation:

```text
corrected PM = intercept + slope * raw SDS011 PM
```

They use identical training and evaluation rows. OLS estimates the intercept and
slope by ordinary least squares. Huber uses iteratively reweighted least squares
to down-weight large residuals while fitting the same two parameters.

## Fold Logic

Sachsen-Anhalt is the sealed final test Land.

For each non-Sachsen-Anhalt inner validation fold:

1. Exclude all Sachsen-Anhalt low-cost sensors and UBA stations.
2. Exclude the validation fold's low-cost sensors from fitting.
3. Exclude the validation fold's UBA stations from fitting.
4. Fit PM10 and PM2.5 coefficients from the remaining folds.
5. Evaluate held-out performance only on nearby matched rows in the validation fold.
6. Write fold label files for non-Sachsen-Anhalt sensors using that fold's coefficients.

The script raises an error if Sachsen-Anhalt or the current validation fold
appears in fitting rows.

## Final Calibration

After cross-validation, the default shared configuration is selected by lowest
mean inner-fold corrected RMSE across pollutants, with mean corrected MAE as a
tie-breaker. Sachsen-Anhalt performance is not used for selection. Explicit
overrides are available through `--final-method` and `--final-radius-km`.

Final fitting then:

1. fits one PM10 equation on all non-Sachsen-Anhalt training data;
2. fits one PM2.5 equation on all non-Sachsen-Anhalt training data;
3. applies the same final PM10 and PM2.5 equations to non-Sachsen-Anhalt CNN
   training sensors;
4. applies those identical equations to Sachsen-Anhalt CNN test sensors.

This coefficient consistency is important: the test labels are not refit or
calibrated separately.

## Outputs

Calibration diagnostics:

| Output | Contents |
| --- | --- |
| `data/processed/calibration/regression_reference_adjustment/cv_results.csv` | One row per `method x radius x fold x pollutant`. |
| `data/processed/calibration/regression_reference_adjustment/cv_summary.csv` | Mean and standard deviation across folds. |
| `data/processed/calibration/regression_reference_adjustment/selection_summary.csv` | Shared method/radius ranking used for final selection. |
| `data/processed/calibration/regression_reference_adjustment/fold_coefficients.json` | Per-fold fitted coefficients. |
| `data/processed/calibration/regression_reference_adjustment/final_coefficients.json` | Final selected method/radius and coefficients. |
| `data/processed/calibration/regression_reference_adjustment/run_metadata.json` | Run settings, split rules, leakage assertions, and output roots. |
| `data/processed/calibration/regression_reference_adjustment/final_label_manifest.json` | Freeze manifest with source settings, row counts, coefficients, revision, and file hashes. |

Corrected annual labels:

| Output | Contents |
| --- | --- |
| `data/processed/corrected/regression_reference_adjustment/folds/radius_<R>km/<fold>/<method>/annual/<YEAR>.csv` | Inner-fold label files for CNN cross-validation. |
| `data/processed/corrected/regression_reference_adjustment/final/radius_<R>km/train/<method>/annual/<YEAR>.csv` | Final non-Sachsen-Anhalt CNN training labels. |
| `data/processed/corrected/regression_reference_adjustment/final/radius_<R>km/test_Sachsen-Anhalt/<method>/annual/<YEAR>.csv` | Final sealed Sachsen-Anhalt CNN test labels. |

Annual label CSVs contain:

- `location`
- `PM10_raw`
- `PM10_corrected`
- `PM2.5_raw`
- `PM2.5_corrected`
- `n_days_total`
- `lat`
- `lon`
- `land`
- `fold`

## Metrics

Held-out validation metrics are annual and station-balanced. Daily pairs are
first averaged to one annual row per low-cost sensor and matched UBA station.
Each UBA station then contributes equal total weight, so stations with many
nearby low-cost sensors do not dominate.

`cv_results.csv` includes MAE, RMSE, bias, R2, fitted intercept/slope, paired
sensor-days, unique low-cost locations, represented UBA stations, and matching
distance summaries.

### Why the constant-mean baseline matters

The annual UBA reference field is relatively narrow. A method can therefore get
low held-out RMSE by predicting nearly the same annual concentration everywhere.
That behavior is useful as a diagnostic but weak as a CNN target, because it
removes the spatial variation the image model is supposed to learn.

For that reason the comparison reports two families of evidence:

- reference-agreement metrics, especially station-balanced annual RMSE and MAE;
- target-signal diagnostics, including annual label standard deviation,
  interdecile range, rank correlation, and high-versus-low station contrast.

The constant-mean baseline anchors the first family. It shows how much error is
available to reduce without learning any spatial signal. The target-signal
diagnostics anchor the second family. They show whether a candidate label set
still contains enough spatial and distributional variation to be meaningful for
CNN training.

## Example Commands

Inspect CLI options:

```bash
python 03_scripts_calibration/regression_reference_adjustment/calibrate_pm_regression_loo.py --help
```

Smoke-test a few months:

```bash
python 03_scripts_calibration/regression_reference_adjustment/calibrate_pm_regression_loo.py \
  --months 2024-04 2024-05 2024-06 \
  --method ols \
  --radius-km 10 \
  --refit
```

Run the full regression calibration across candidate radii and methods:

```bash
python 03_scripts_calibration/regression_reference_adjustment/calibrate_pm_regression_loo.py \
  --method both \
  --radius-km 5 10 20 \
  --refit
```

Run the fair label-construction comparison:

```bash
python 03_scripts_calibration/regression_reference_adjustment/compare_label_construction_methods.py
```

The comparison writes to:

```text
data/processed/calibration/regression_reference_adjustment/method_comparison/
```

Force a final candidate after reviewing inner-fold results:

```bash
python 03_scripts_calibration/regression_reference_adjustment/calibrate_pm_regression_loo.py \
  --method both \
  --radius-km 5 10 20 \
  --final-method ols \
  --final-radius-km 20 \
  --refit
```

## Current 2024 Freeze Recommendation

The completed 2024 scientific comparison does not recommend the shared OLS
20 km configuration as the main label set. OLS 20 km strongly improves over the
raw SDS011 baseline, but it does not beat the constant-mean baseline on combined
held-out RMSE and it compresses annual target variation heavily.

Current recommendation:

- main CNN labels: original percentile-mapping method;
- sensitivity labels: OLS regression adjustment at 20 km;
- not recommended as main labels: Huber regression adjustment and other OLS
  radii, because they remain close to mean predictions.

The OLS 20 km final coefficients are still documented for reproducibility:

- PM10: intercept `13.7745556932`, slope `0.0222735557`
- PM2.5: intercept `8.3626846893`, slope `0.0195873056`
- final non-Sachsen-Anhalt train labels: `4,245`
- final Sachsen-Anhalt test labels: `60`

The detailed comparison is summarized in `RESULTS.md`.

## External Weather And UBA Metadata Pass

For close-reference weather diagnostics, run the small acquisition pass:

```bash
python 01_scripts_gathering/03_download_uba_measurements.py --station-metadata
python 01_scripts_gathering/download_extract_hyras.py
```

This uses HYRAS-DE v6-1 daily 2024 TAS and HURS NetCDF files from the DWD
Climate Data Center and UBA Air Data API v4 through
`https://luftdaten.umweltbundesamt.de/api-proxy`. The pass writes:

- `data/processed/daily_weather/hyras_2024_sds011.parquet`
- `data/processed/uba/station_metadata.csv`
- `data/processed/calibration/regression_reference_adjustment/external_weather_metadata/results.csv`
- `data/processed/calibration/regression_reference_adjustment/external_weather_metadata/REPORT.md`

HYRAS is a consistent 1 km gridded ambient-weather product in EPSG:3035; it is
not the same thing as same-location Sensor.Community BME280/DHT22 device
weather. This acquisition pass does not fit calibration models, write corrected
PM labels, or train the CNN.

## Limitations and Notes

- Nearby matched sensors are not truly co-located with UBA stations.
- Humidity is not corrected in this branch.
- One global relationship is shared across SDS011 sensors within each
  fold/method/pollutant.
- The fitted 2024 affine models have small slopes and substantially compress
  annual target variation relative to raw SDS011 annual means. This is a known
  limitation of the global regression label set, not a hidden post-processing
  step.
- Full runs require `sensor_land.csv` and `uba/station_land.csv` from the
  corrected Land-assignment workflow.
- Existing percentile-mapping outputs are not overwritten by this branch.
