# Results

This folder contains generated modeling outputs. Results are separated by
experiment so that frozen ResNet, partially fine-tuned ResNet, and scratch CNN
runs do not overwrite each other.

Current experiment folders:

```text
resnet_frozen/
resnet_layer4/
cnn/
```

The older `resnet/` folder may contain earlier frozen-ResNet outputs from before
the result paths were split by experiment. Keep it as historical output unless
the exact run configuration is known. Do not automatically relabel it as
`resnet_frozen`.

## Cross-Validation Outputs

`eea_cv_results.json` is written by:

```bash
python 06_models/01_train_cv.py --experiment resnet_frozen
```

It contains per-fold validation metrics, baseline metrics, station counts,
buffer-removal counts, best epoch, epochs run, batch metadata, parameter counts,
model metadata, and the config used for the run.

`cv_history.csv` is also written during CV. It has one row for each completed
epoch, so useful history remains available even if a long run is interrupted.
It stores training loss, train and validation metrics, learning rates, timing
diagnostics, the best-so-far flag, and the patience counter.

`cv_folds.csv` is the compact fold table for later analysis. It stores one row
per fold and pollutant with station counts, 100 km buffer removals, best epoch,
validation metrics, baseline RMSE, and parameter counts.

`eea_cv_predictions.csv` contains out-of-fold validation predictions with station
metadata such as station code, country, land, coordinates, and fold where those
columns are available.

`run_metadata.json` records the experiment name, model type, pollutant list,
configuration, fold setup, buffer distance, parameter counts, Git state,
Python/PyTorch/CUDA/device information, and run timestamps.

## Final Checkpoint

`final_model.pt` is written by:

```bash
python 06_models/02_train_final.py --experiment resnet_frozen
```

It contains the model weights plus the information needed to reproduce test
prediction with the same preprocessing:

- model config;
- selected streams;
- pollutant list;
- target normalization values;
- Sentinel-5P training statistics;
- arithmetic train-mean baseline concentration;
- buffer metadata;
- CV `best_epoch` values;
- final epoch count;
- parameter counts.

## Sealed Test Predictions

`test_predictions.csv` is written by:

```bash
python 06_models/03_predict_test.py --experiment resnet_frozen
```

It contains sealed TEST predictions with station metadata and true labels where
available. This file should only be produced after the model choice has been
made from development-fold CV.

Large checkpoints and generated prediction files should usually stay out of Git
unless they are intentionally needed for hand-in or reproducibility.

## Summary Outputs

After one or more CV runs are complete, run:

```bash
python 06_models/04_summarize_results.py
```

The script reads the available experiment folders and writes:

```text
summary/experiment_comparison.csv
summary/fold_comparison.csv
```

It reports experiments that are still missing instead of filling in fake values.
