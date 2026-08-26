# Air Pollution Deep Learning Project

We are trying to predict long-term particulate air pollution across Germany from
satellite imagery and ground measurements. The main pollutants are:

- `PM10`: particles smaller than 10 micrometers.
- `PM2.5`: finer particles smaller than 2.5 micrometers.

Both matter for health, and both are measured by official monitoring stations.

## Research Question

Can we use satellite data and ground measurements to estimate annual air
pollution in places where there is no official station?

The current modeling direction is to train and evaluate against official
reference measurements from UBA/EEA stations. Sensor.Community data may still be
tested as extra noisy training data, but we are not treating it as proven ground
truth.

## Data Sources

| Source | What it is | Why we use it |
| --- | --- | --- |
| UBA reference stations | Official German air-quality stations from the Umweltbundesamt. | These are our most trusted PM10/PM2.5 measurements. |
| EEA / AirBase data | European reference-station data and gridded products. | Useful for comparison and possible wider validation. |
| Sensor.Community | Volunteer low-cost sensors, mostly SDS011 PM sensors plus weather sensors. | Much denser spatial coverage than UBA, but individual readings are noisy. |
| Sentinel-2 | Optical satellite images with 10-20 m bands. | Gives land cover and local spatial context around each station or sensor. |
| Sentinel-5P | Coarser atmospheric satellite products such as NO2. | Gives broader air-quality context that Sentinel-2 cannot see directly. |

## Current Approach

The final model branch trains on official EEA reference-station labels and
predicts annual PM2.5. The selected model is `cnn_deep_wide`, a scratch CNN that
uses Sentinel-2 patches, Sentinel-5P context, and elevation data. A patch is a
small satellite image window centered on a measurement location.

We validate geographically with European folds during development. Northern and
eastern Germany are kept as a sealed TEST region. This helps us test whether the
model generalizes to new regions instead of memorizing local patterns.

Earlier work tried to train mainly on corrected Sensor.Community annual labels.
Those labels are now treated cautiously. The calibration experiments showed that
the sensors are useful to investigate, but the tested corrections did not make
them reference-equivalent annual measurements.

## Repository Structure

| Folder | What happens there |
| --- | --- |
| `01_scripts_gathering/` | Download raw Sensor.Community, UBA, EEA, HYRAS, and related external data. |
| `02_scripts_cleaning/` | Convert raw downloads into hourly, daily, monthly, and annual tables, including sensor state assignment. |
| `03_scripts_calibration/` | Keep Sensor.Community calibration experiments and their summaries. |
| `04_GEE/` | Download Sentinel-2 and Sentinel-5P patches with Google Earth Engine. |
| `05_scripts_visual/` | Optional maps for checking sensor and station coverage. |
| `06_models/` | Current EEA reference-station model training, saved results, and TEST prediction code. |
| `07_prediction_analysis/` | TEST diagnostics and dense-grid prediction outputs. |
| `08_kreislevel_data/` | Exploratory Kreis-level socioeconomic analysis. |
| `Air_pollution_report/` | Current LaTeX report files and report figures. |

## Setup On A New Computer

Start from the repository root:

```bash
cd /path/to/air_pollution
```

Install Python dependencies in your own environment. The current
`requirements.txt` looks like a captured environment rather than a carefully
trimmed list, so expect some package-version work if you are setting up from
scratch.

Most scripts assume they are run from the repository root and build paths like
`data/raw/...` and `data/processed/...`.

## Data Directory

Large data files are not stored in Git. The code currently expects a `data/`
directory at the repository root. On the current Mac setup, `data/` can be a
symlink to an external drive. For example:

```bash
ln -s /Volumes/EXTERNAL_DRIVE/air_pollution_data data
```

On Windows or Linux, the same idea applies: make sure `data/` exists at the
repository root, either as a normal folder or as a link to wherever the large
data live. Some experiment scripts also accept `--processed-dir`, but most of
the main pipeline simply looks for `data/` next to the code.

Expected high-level layout:

```text
data/
├── raw/
├── processed/
│   ├── hourly/
│   ├── daily_avg/
│   ├── monthly_avg/
│   ├── calibration/
│   ├── corrected/
│   └── satellite/
```

Do not commit raw downloaded archives, Sentinel patches, model checkpoints,
large arrays, or large intermediate tables. The small calibration summary CSV in
`03_scripts_calibration/` is intentionally kept in Git because it is a report
summary, not raw data.

## Short Pipeline

The project now has two main branches:

1. Sensor.Community branch: clean low-cost sensor data, assign sensors to German
   states, run calibration experiments, and conclude that the tested annual
   labels are not used as final CNN training labels.
2. Final modeling branch: use EEA reference stations plus satellite and context
   data, run geographic CV, select `cnn_deep_wide`, evaluate the sealed TEST
   region, predict a dense grid, and run an exploratory Kreis-level
   socioeconomic analysis.

See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the longer version.

## Project Status

Implemented:

- 2024 Sensor.Community and UBA data gathering.
- PM and humidity cleaning.
- Sensor and station state assignment.
- Current annual proxy-label generation.
- Sentinel patch download for sensor/station locations.
- Geographic CV on EEA reference stations.
- Final selected `cnn_deep_wide` PM2.5 model.
- Sealed TEST evaluation.
- Dense-grid PM2.5 prediction.
- Exploratory Kreis-level socioeconomic analysis.

Still under evaluation or planned:

- Whether Sensor.Community helps as weak labels. A weak label is a noisy label
  that may still help training if the final model improves on held-out reference
  stations.
- Comparison with EEA gridded products.
- High-versus-low PM patch examples for the report, which need data stored on
  the university server.

## Where To Read More

- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md): full workflow and current status.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): what can be checked from this
  checkout and what needs external data.
- [03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md): what we learned from the sensor calibration tests.
- [06_models/README.md](06_models/README.md): current model training code.
