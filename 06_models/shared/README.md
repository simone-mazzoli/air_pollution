# Shared Modeling Code

Code in `shared/` should behave the same regardless of whether the
high-resolution encoder is the ResNet or the future custom CNN.

Main files:

- `config.py` stores shared experiment settings such as pollutant, input streams,
  batch size, seed, TTA, buffer distance, and result-path helpers.
- `paths.py` defines repository-relative paths for processed data, satellite
  patches, fold files, and results.
- `folds.py` stores the fold-creation rules and loads the frozen
  `station_fold.csv`.
- `data.py` loads labels and satellite patches, applies train-only
  normalization, builds the PyTorch dataset, and applies the 100 km buffer.
- `evaluation.py` contains metrics, baseline calculations, TTA evaluation, and
  prediction-table helpers.
- `training.py` contains model-independent training utilities.
- `models.py` selects the configured model package.

Model-specific architecture decisions belong in `resnet/` or `cnn/`, not here.

