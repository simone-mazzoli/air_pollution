# Results

This folder contains generated modeling outputs. Results are separated by
experiment so that frozen ResNet, partially fine-tuned ResNet, and the future
scratch CNN do not overwrite each other.

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
python 06_models/01_train_cv.py
```

It contains per-fold validation metrics, baseline metrics, station counts,
buffer-removal counts, best epoch, epochs run, batch metadata, parameter counts,
model metadata, and the config used for the run.

`eea_cv_predictions.csv` contains out-of-fold validation predictions with station
metadata such as station code, country, land, coordinates, and fold where those
columns are available.

## Final Checkpoint

`final_model.pt` is written by:

```bash
python 06_models/02_train_final.py
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
python 06_models/03_predict_test.py
```

It contains sealed TEST predictions with station metadata and true labels where
available. This file should only be produced after the model choice has been
made from development-fold CV.

Large checkpoints and generated prediction files should usually stay out of Git
unless they are intentionally needed for hand-in or reproducibility.
