# ResNet Modeling

## Purpose

This directory contains the current pretrained ResNet baseline for predicting calibrated pollution targets from Sentinel-2 patches. The implementation currently trains on high-resolution multispectral patches only.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `01_find_stats.py` | Fetches BigEarthNet/ConfigILM model metadata and prints the band order plus normalization constants for a BIFOLD BigEarthNet ResNet model. | Hugging Face `config.json`; ConfigILM `BENv2_utils.py`; optional model id argument. | Printed band order and Python mean/std lists. | Utility / validation |
| `02_sat_normalize.py` | Defines patch normalization and `SatellitePatchDataset`, filters against the satellite manifest, and can save debug grids for all folds/streams or one selected fold/stream. | `data/processed/satellite/<stream>/*.npy`; `data/processed/satellite/manifest.csv`; `data/processed/corrected/fold/*/annual/*.csv`. | Debug preview PNGs, by default under `debug_patches/` relative to the current working directory. | Validation / dataset utility |
| `03_train_resnet.py` | Fine-tunes a BigEarthNet ResNet50 high-resolution patch baseline using non-Sachsen-Anhalt held-out-fold cross-validation. | `data/processed/sensor_land.csv`; `data/processed/corrected/fold/<fold>/annual/2024.csv`; `data/processed/satellite/high_res_multispec/*.npy`; pretrained weights from Hugging Face if available. | JSON cross-validation metrics, default `resnet_cv_results.json` relative to the current working directory. | Main modeling step / experimental baseline |

## Pipeline position

Run this after [`../../04_GEE`](../../04_GEE/README.md) has produced patch files and after [`../../03_scripts_calibration`](../../03_scripts_calibration/README.md) has produced calibrated annual labels and `sensor_land.csv`.

The current implementation stops at cross-validation over the non-test folds. A final sealed Sachsen-Anhalt evaluation script is not present in this repository state.

## Data flow

Sentinel-2 `.npy` patches -> BigEarthNet band-order normalization -> PyTorch dataset -> high-resolution ResNet50 regression baseline -> fold-level validation metrics.

## Running the scripts

```bash
python 06_models/02_resnet/01_find_stats.py
python 06_models/02_resnet/01_find_stats.py BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0

python 06_models/02_resnet/02_sat_normalize.py
python 06_models/02_resnet/02_sat_normalize.py --stream-dir data/processed/satellite/high_res_multispec --locations-csv data/processed/corrected/fold/Bayern/annual/2024.csv
python 06_models/02_resnet/02_sat_normalize.py --diagnose --stream-dir data/processed/satellite/high_res_multispec --locations-csv data/processed/corrected/fold/Bayern/annual/2024.csv

python 06_models/02_resnet/03_train_resnet.py
python 06_models/02_resnet/03_train_resnet.py --folds Bayern Hessen --epochs 20 --batch 32 --out resnet_cv_results.json
```

## Important assumptions and caveats

- `03_train_resnet.py` currently uses `TARGETS = ["PM10_corrected"]`; despite the docstring mentioning two-output PM10/PM2.5 regression, the current code trains PM10 only.
- `03_train_resnet.py` uses only `data/processed/satellite/high_res_multispec`; the low-resolution stream is not used by the trainer.
- Sachsen-Anhalt is excluded from cross-validation. The script validates across non-Sachsen-Anhalt folds and does not run the final sealed Sachsen-Anhalt evaluation.
- Targets are log-transformed when `LOG_TARGET = True`, standardized using training-fold statistics, and converted back to `ug/m3` for metrics.
- Patch filenames are canonicalized to one decimal place, for example `49.0.npy`, to match files produced when `location` values were read as floats.
- `02_sat_normalize.py` filters requested patches using `manifest.csv`; `03_train_resnet.py` filters only by file existence and does not use its `MANIFEST` constant.
- The pretrained backbone download requires network access and `huggingface_hub`, `safetensors`, and `timm`. If loading fails, the code warns and continues with a random-initialized backbone.
- The checked-in `resnet_cv_results.json` currently contains a partial-looking result for one fold, not a complete cross-validation run.
