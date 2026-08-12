# 06 Models

This folder contains the modeling code for the air-pollution project. We use
European EEA/UBA reference stations to train models that predict annual PM2.5
from satellite and environmental inputs. The final application is focused on
Germany, so the sealed test region is currently northern/eastern Germany.

Large satellite patches are not stored in Git. The scripts expect them under the
project's external `data/` structure.

## Inputs

Each station can use the same set of input data:

- Sentinel-2 high-resolution patch.
- Sentinel-2 lower-resolution context patch.
- Sentinel-5P NO2 and CO patches.
- A wider Sentinel-5P aerosol input.
- DEM/elevation information.

The pretrained ResNet experiments and the future custom CNN use the same shared
data loading, folds, buffer, preprocessing, target transform, and metrics.

## Geographic Validation

The station split is geographic. `data/processed/eea/station_fold.csv` stores the
station-level assignment used by training. Do not regenerate it unless the fold
design is intentionally being changed.

During cross-validation, stations within 100 km of the held-out validation fold
are removed from training. During final training, stations within 100 km of the
sealed test stations are removed from training. The TEST stations are not used
for choosing architectures or hyperparameters.

## Models

Current comparison plan:

```text
ResNet frozen
    BigEarthNet-pretrained ResNet50
    pretrained backbone frozen

ResNet layer4
    same pretrained model
    only final ResNet stage fine-tuned

Custom CNN
    to be implemented
    trained from scratch
```

The ResNet variants share the same auxiliary Sentinel-5P, aerosol, DEM, scalar,
training, and evaluation code. The custom CNN should use those same shared parts
when it is implemented.

## Configuration

Normal runs are config-first.

- Change the selected model in `shared/config.py` with `MODEL`.
- Change the ResNet mode in `resnet/config.py` with `BACKBONE_MODE`.
- Shared settings such as pollutant, streams, batch size, seed, TTA, and the
  100 km buffer live in `shared/config.py`.
- CV can still run a subset of folds with `--folds`.

Supported ResNet modes are:

```python
BACKBONE_MODE = "frozen"
BACKBONE_MODE = "layer4"
```

`"frozen"` keeps the pretrained ResNet feature extractor fixed. `"layer4"` is a
partial fine-tuning experiment and still needs development-fold CV before it can
be considered for the sealed test set.

## Running

Run commands from the repository root. In PyCharm, set the working directory to
the repository root and run the same scripts.

```bash
python3 06_models/00_assign_folds.py
python3 06_models/01_train_cv.py
python3 06_models/02_train_final.py
python3 06_models/03_predict_test.py
```

Pipeline order:

- `00_assign_folds.py` creates `station_fold.csv` and a fold map. Do not rerun
  this unless the fold assignment should change.
- `01_train_cv.py` trains development folds and writes CV results/predictions.
- `02_train_final.py` trains on all development folds using the CV-selected epoch
  count.
- `03_predict_test.py` predicts the sealed TEST stations from the final
  checkpoint.

For an exploratory subset CV run:

```bash
python3 06_models/01_train_cv.py --folds fold1_iberia fold2_france
```

## Results

New results are separated by experiment:

```text
results/resnet_frozen/
results/resnet_layer4/
results/cnn/
```

The older `results/resnet/` directory may contain earlier frozen-ResNet outputs.
Leave those files as historical outputs unless their exact configuration is
known.

Generated checkpoints and large result files should usually stay out of Git
unless they are intentionally needed for the project hand-in or reproducibility.

## Folders

| Folder | What is in it |
| --- | --- |
| `shared/` | Data loading, fold loading, metrics, training, config, model selection, and path helpers. |
| `resnet/` | BigEarthNet-pretrained ResNet model and ResNet-specific config. |
| `cnn/` | Placeholder for the future custom CNN. |
| `results/` | Generated experiment outputs. |
| `archive/sensor_community/` | Older Sensor.Community modeling code kept for history. |

