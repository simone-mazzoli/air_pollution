# Prediction Analysis

This folder is for post-hoc checks of the frozen final model predictions. It
does not train a model, choose a model, tune hyperparameters, or change the
sealed TEST result.

The current script reads:

```bash
06_models/results/cnn_deep_wide/test_predictions.csv
```

and writes diagnostic figures and tables under:

```text
07_prediction_analysis/outputs/
```

Run it after the final sealed TEST prediction file exists:

```bash
python3 07_prediction_analysis/analyze_test_predictions.py
```

The map figures use cached vector GeoJSON boundaries, not web map tiles. The
script can cache them once with:

```bash
python3 07_prediction_analysis/analyze_test_predictions.py --download-boundaries
```

The cached files live in `07_prediction_analysis/boundaries/`. Country outlines
come from Natural Earth 1:50m Admin 0 Countries. German Bundesland boundaries
come from geoBoundaries Open DEU ADM1. The plotting code reads those GeoJSON
files directly with the Python standard library and matplotlib, so the server
does not need a GIS stack such as cartopy or geopandas.
If those boundary files are unavailable, the script skips the geographic map
figures instead of writing plain longitude/latitude scatterplots.

The predictions and targets are station-level annual PM2.5 concentrations in
µg/m³ for the sealed north/east German TEST region. These are not dense map
predictions.

Residuals are defined as:

```text
residual = prediction - observation
```

So a positive residual means the model overpredicts PM2.5, and a negative
residual means it underpredicts PM2.5.

RMSE/MAE and R² can look contradictory in a narrow TEST region. RMSE and MAE
measure error in concentration units. R² compares the model against predicting
the TEST-set mean, which is only a mathematical reference for that held-out set.
The training-mean baseline is different: it is a deployable baseline derived
from the final training data only.

Later, dense prediction maps and comparison to an external EEA modeled PM2.5
product can be added in separate subfolders such as `dense_prediction/` and
`eea_comparison/`. That comparison should stay qualitative and external; it
should not become a target for tuning the model.
