# Reproducibility Notes

This checkout contains code and several saved results, but it does not contain
the large raw data. On this machine, `data` is a symlink to:

```text
/Volumes/Extreme SSD/air_pollution_data
```

That target is not available in the current checkout. A full rerun needs the
data folder from the university server or another local copy.

## Saved Results That Can Be Checked Locally

- CV predictions and result files under `06_models/results/`.
- Main CV comparison tables under `06_models/results/summary/`.
- The final selected checkpoint:
  `06_models/results/cnn_deep_wide/final_model.pt`.
- Saved TEST predictions:
  `06_models/results/cnn_deep_wide/test_predictions.csv`.
- The pre-TEST final model-selection record:
  `06_models/results/cnn_deep_wide/final_model_selection.json`.
- Dense-grid predictions:
  `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv`.
- Socioeconomic result tables under `08_kreislevel_data/`.
- Sensor.Community calibration summary files under `03_scripts_calibration/`.
- Current report figures under `Air_pollution_report/Figures/generated/`.

## Data Needed For A Full Rerun

- Raw EEA daily data and processed EEA station tables.
- Raw and processed Sensor.Community data.
- UBA reference data used by the calibration branch.
- Google Earth Engine exports and satellite patch arrays.
- Model-ready patch folders for EEA stations and dense-grid prediction.
- Kreis boundary geometry used for map regeneration.
- JupyterHub-only patch data for the high-versus-low PM examples.

## What Can Be Recomputed From This Checkout

- Summary tables from saved CV result folders.
- Learning-curve figures from saved CV history CSVs.
- Report-facing figures that only use saved result CSVs and existing local map
  boundaries.
- The LaTeX report, if the local TeX setup is installed.

## What Cannot Be Recomputed Here

- Full model training or cross-validation.
- Final TEST prediction from the checkpoint if the patch arrays are missing.
- Dense-grid prediction if the model-ready grid patch arrays are missing.
- Google Earth Engine patch exports.
- Sensor.Community raw-data cleaning.
- The socioeconomic summary map if Kreis geometry is not present.

The main saved result used in the report is the `cnn_deep_wide` model. Its final
training used 24 epochs, recorded before the TEST set was opened.
