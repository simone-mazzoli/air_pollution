# Reproducibility Notes

This checkout contains code, the final report and saved result files. It does
not contain the large raw downloads or satellite patch arrays needed for a full
rerun.

## Saved Results In This Checkout

- Final model-selection record:
  `06_models/results/cnn_deep_wide/final_model_selection.json`
- Final checkpoint:
  `06_models/results/cnn_deep_wide/final_model.pt`
- Geographic CV predictions, folds, histories and metrics:
  `06_models/results/cnn_deep_wide/`
- Frozen ResNet comparison results:
  `06_models/results/resnet_frozen/`
- Summary comparison tables:
  `06_models/results/summary/`
- Final TEST predictions:
  `06_models/results/cnn_deep_wide/test_predictions.csv`
- Dense-grid predictions:
  `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv`
- Kreis-level socioeconomic results:
  `08_kreislevel_data/`
- Sensor.Community calibration summary:
  `03_scripts_calibration/sensor_community_calibration_summary.csv`
- PM2.5 completeness and preliminary-analysis outputs:
  `analysis_outputs/`
- Final report figures:
  `Air_pollution_report/Figures/generated/`

These files allow the main reported results to be inspected without retraining.

## External Data Not Included

A full rerun needs local copies of:

- raw and processed EEA station data
- raw and processed Sensor.Community data
- UBA reference data for the calibration branch
- HYRAS weather data used in calibration diagnostics
- Google Earth Engine satellite/context patch arrays
- dense-grid patch arrays
- large boundary/source files not stored in Git

The expected local layout is described in `data/README.md`.

## Model Selection And Later Reruns

The final model is `cnn_deep_wide`. The final training run used 24 epochs,
taken from the original pre-TEST model-selection record in
`06_models/results/cnn_deep_wide/final_model_selection.json`.

Later learning-curve and reproducibility runs are saved because they support the
training-diagnostic figures and chronology. They are supplementary reruns and do
not change the pre-TEST model selection.

Model runs use seed 123. Since deterministic cuDNN execution was not enabled,
reruns are not guaranteed to be bit-for-bit identical. Later runs are also not
guaranteed to match exactly because they were conducted under a later
implementation state.

## Quick Checks

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall 01_scripts_gathering 02_scripts_cleaning 03_scripts_calibration 04_GEE 05_scripts_visual 06_models 07_prediction_analysis 08_kreislevel_data 09_report_figures
python3 06_models/check_shared_logic.py
```

The final report can be rebuilt with:

```bash
cd Air_pollution_report
latexmk -pdf -outdir=build main_merged.tex
```
