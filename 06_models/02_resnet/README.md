# 06 ResNet Baseline

This folder contains the current image model baseline. It fine-tunes a
BigEarthNet ResNet50 on satellite patches. Right now the main script predicts
`PM10_corrected` from high-resolution Sentinel-2 patches.

Run commands from the repository root.

## Main Scripts

| Script | What it does | Main output |
| --- | --- | --- |
| `sensors-related/01_find_stats.py` | Prints the BigEarthNet band order and normalization constants. | terminal output |
| `sensors-related/02_sat_normalize.py` | Loads patches, applies normalization, and can save debug preview images. | `debug_patches/` or a chosen preview image |
| `sensors-related/03_train_resnet.py` | Trains the current PM10 ResNet baseline with state-based validation folds. | `resnet_cv_results.json` by default |
| `01_eea_s2p_normalizer.py` | normalizes existing EEA S2 patches and pairs them with per-station annual PM labels for model training. | no output file (must import EEAPatchDataset, load_labels) |

## Usual Commands

```bash
python3 06_models/02_resnet/sensors-related/01_find_stats.py

python3 06_models/02_resnet/sensors-related/02_sat_normalize.py \
  --stream-dir data/processed/satellite/high_res_multispec \
  --locations-csv data/processed/corrected/fold/Bayern/annual/2024.csv

python3 06_models/02_resnet/sensors-related/03_train_resnet.py --folds Bayern Hessen --epochs 20 --batch 32
```

The full trainer can take a while, especially if it downloads pretrained weights
from Hugging Face or trains many folds.

## Inputs

- `data/processed/sensor_land.csv`
- `data/processed/corrected/fold/<fold>/annual/2024.csv`
- `data/processed/satellite/high_res_multispec/*.npy`
- optional Sentinel-5P streams if you pass `--s5p` or `--s5p-wide`

## Current Limitations

- The trainer currently predicts PM10 only, even though some older comments talk
  about PM10 and PM2.5 together.
- It uses high-resolution Sentinel-2 by default. Low-resolution and Sentinel-5P
  streams are not fully settled as the main model input.
- `Sachsen-Anhalt` is excluded from the current cross-validation folds. The
  final reference-station evaluation design is still being decided.
- The checked-in result files in this folder should be treated as old run
  outputs, not as the final project result.

## Extra Scripts And Files

| File | Notes |
| --- | --- |
| `visualize_sensors_map.py` | Optional map helper for a CSV with `lat`, `lon`, and a value column. |
| `UBA_val.py` | Reference-station validation helper; not the main training command. |
| `untested/` | Older model experiments. Read before using. |
| `best_model.pt`, `*.json`, `*.csv`, `debug_patches/` | Generated outputs from previous runs. Large new outputs should not be committed. |
