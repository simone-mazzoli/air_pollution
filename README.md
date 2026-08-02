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

The current baseline trains a ResNet model on Sentinel image patches. A patch is
a small satellite image window centered on a measurement location. We validate
geographically: some German federal states are left out during training and used
for validation. This helps us test whether the model generalizes to new regions
instead of memorizing local patterns.

Earlier work tried to train mainly on corrected Sensor.Community annual labels.
Those labels are now treated cautiously. The calibration experiments showed that
the sensors are useful to investigate, but the tested corrections did not make
them reference-equivalent annual measurements.

## Repository Structure

| Folder | What happens there |
| --- | --- |
| `01_scripts_gathering/` | Download raw Sensor.Community, UBA, EEA, HYRAS, and related external data. |
| `02_scripts_cleaning/` | Convert raw downloads into hourly, daily, monthly, and annual tables. |
| `03_scripts_calibration/` | Assign sensors to German states, make current proxy labels, and keep calibration experiments. |
| `04_GEE/` | Download Sentinel-2 and Sentinel-5P patches with Google Earth Engine. |
| `05_scripts_visual/` | Optional maps for checking sensor and station coverage. |
| `06_models/02_resnet/` | Current ResNet training and patch-normalization code. |

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

1. Download Sensor.Community and UBA/EEA data.
2. Clean the raw sensor archives into hourly, daily, and monthly tables.
3. Assign UBA stations and Sensor.Community sensors to German federal states.
4. Build annual labels used by the current modeling experiments.
5. Download Sentinel image patches for those locations.
6. Normalize patches and train the ResNet baseline.
7. Compare predictions against geographically held-out reference stations.

See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the longer version.

## Project Status

Implemented:

- 2024 Sensor.Community and UBA data gathering.
- PM and humidity cleaning.
- Sensor and station state assignment.
- Current annual proxy-label generation.
- Sentinel patch download for sensor/station locations.
- A high-resolution ResNet PM10 baseline.

Still under evaluation or planned:

- Whether Sensor.Community helps as weak labels. A weak label is a noisy label
  that may still help training if the final model improves on held-out reference
  stations.
- Final reference-station training/evaluation design.
- Continuous gridded prediction.
- Comparison with EEA gridded products.
- Socioeconomic or environmental-justice analysis.

## Where To Read More

- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md): full workflow and current status.
- [03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md](03_scripts_calibration/SENSOR_COMMUNITY_CALIBRATION_SUMMARY.md): what we learned from the sensor calibration tests.
- [06_models/02_resnet/README.md](06_models/02_resnet/README.md): current model training code.
