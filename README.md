# Air Pollution Deep Learning Project

This repository contains the code, saved results and final report for the
Master's project:

**Mapping Air Pollution and Exposure Inequality in Germany from Satellite
Imagery**

The project predicts annual PM2.5 concentrations from official monitoring
stations and satellite/context inputs. The final selected model is
`cnn_deep_wide`.

## Workflow

The project has two main parts.

1. Low-cost Sensor.Community data were gathered, cleaned and tested against UBA
   reference stations. The tested annual calibration approaches were not accurate
   enough for final model labels.
2. The final model branch uses official EEA reference-station labels, downloads
   satellite and context patches, runs geographic cross-validation, trains the
   selected model, evaluates the sealed TEST region, predicts a dense grid and
   joins the predictions to Kreis-level socioeconomic data.

See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the step-by-step workflow.

## Repository Structure

| Path | Contents |
| --- | --- |
| `01_scripts_gathering/` | Scripts for external Sensor.Community, UBA, EEA, HYRAS and boundary data. |
| `02_scripts_cleaning/` | Cleaning and aggregation scripts for station and sensor data. |
| `03_scripts_calibration/` | Low-cost sensor calibration experiments and summaries. |
| `04_GEE/` | Google Earth Engine scripts for satellite/context patch exports. |
| `05_scripts_visual/` | Small visual checks for station coverage. |
| `06_models/` | Model definitions, training scripts and saved model results. |
| `07_prediction_analysis/` | TEST diagnostics and dense-grid prediction outputs. |
| `08_kreislevel_data/` | Kreis-level exposure and socioeconomic analysis. |
| `09_report_figures/` | Scripts that build report figures from saved data. |
| `analysis_outputs/` | Data-quality and preliminary-analysis outputs used by the report. |
| `Air_pollution_report/` | LaTeX report source, bibliography, figures and final PDF. |
| `data/` | Placeholder README for local raw and processed data. |

## Main Saved Results

Important saved result files are kept in the repository so the report can be
checked without retraining.

- Final model-selection record:
  `06_models/results/cnn_deep_wide/final_model_selection.json`
- Final checkpoint:
  `06_models/results/cnn_deep_wide/final_model.pt`
- Cross-validation predictions and metrics:
  `06_models/results/cnn_deep_wide/`
- Frozen ResNet comparison results:
  `06_models/results/resnet_frozen/`
- TEST predictions:
  `06_models/results/cnn_deep_wide/test_predictions.csv`
- Dense-grid predictions:
  `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv`
- Kreis-level socioeconomic outputs:
  `08_kreislevel_data/`
- Low-cost sensor calibration summary:
  `03_scripts_calibration/sensor_community_calibration_summary.csv`

## Data

Large raw downloads, processed satellite arrays and model-ready patch folders
are not committed. Use the scripts under `01_scripts_gathering/`,
`02_scripts_cleaning/` and `04_GEE/` to download or construct them when needed.

The expected local layout is described in [data/README.md](data/README.md). A
full rerun needs those external data. The saved result files listed above are
included for inspection of the final report results.

## Report

The final report source is:

```bash
Air_pollution_report/main_merged.tex
```

The current compiled report is:

```bash
Air_pollution_report/build/main_merged.pdf
```

To rebuild it locally:

```bash
cd Air_pollution_report
latexmk -pdf -outdir=build main_merged.tex
```

## Reproducibility

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains which results can be checked
from this checkout and which steps require external data.
