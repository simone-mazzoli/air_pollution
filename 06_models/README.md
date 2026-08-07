# 06 Models

This folder contains the modeling part of the air-pollution project. The main
task here is to predict annual PM2.5 concentration at monitoring stations from
satellite and environmental data.

PM2.5 means fine particulate matter with diameter below 2.5 micrometers. It is
one of the most important air-pollution indicators because small particles can
enter deep into the lungs. The labels used here come from official reference
stations, mainly EEA stations across Europe and UBA information for Germany. EEA
is the European Environment Agency. UBA is the German Environment Agency.

The model inputs are small image-like patches around each station. They include
Sentinel-2 multispectral imagery, Sentinel-5P atmospheric measurements, and DEM
elevation data. Sentinel-2 observes land surface reflectance at several spectral
bands. Sentinel-5P observes atmospheric gases and aerosols. DEM means digital
elevation model, which gives terrain height.

The current experiments train on Europe-wide reference stations. Northern and
eastern Germany are kept as a sealed final test region. "Sealed" means that this
region is not used for choosing the model architecture, hyperparameters, or
training settings. It is only used after development choices have been made.

The immediate comparison is:

- a BigEarthNet-pretrained ResNet50 used for transfer learning;
- a partially fine-tuned version of the same ResNet50;
- a later CNN trained from scratch.

BigEarthNet is a large remote-sensing dataset. Pretraining on it gives the
ResNet useful satellite-image features before it is trained for pollution. The
scratch CNN will show how much this pretraining helps compared with learning the
image features only from our pollution dataset. Later, the same modeling setup
can be used to make continuous pollution maps instead of only station-level
predictions.

## Directory Overview

```text
06_models/
|-- 00_assign_folds.py
|-- 01_train_cv.py
|-- 02_train_final.py
|-- 03_predict_test.py
|-- check_shared_logic.py
|-- shared/
|-- resnet/
|-- cnn/
|-- results/
`-- archive/
```

- `00_assign_folds.py` builds `data/processed/eea/station_fold.csv`, the station
  file that says which geographic fold each station belongs to. Later scripts
  read this file instead of recomputing the split.
- `01_train_cv.py` runs development cross-validation. Each development fold is
  held out once, the model is trained on the remaining development stations, and
  validation metrics are written to the matching experiment folder.
- `02_train_final.py` trains one final model on all development folds. Before
  training, it removes development stations that are within 100 km of the sealed
  TEST stations.
- `03_predict_test.py` loads the final checkpoint and evaluates the sealed
  northern/eastern Germany TEST stations.
- `check_shared_logic.py` is a small assertion-based check for modeling logic.
  It prints nothing when all checks pass.
- `shared/` contains data loading, folds, metrics, training helpers, paths, and
  shared configuration.
- `resnet/` contains the BigEarthNet-pretrained ResNet model and ResNet-specific
  settings.
- `cnn/` is reserved for the required CNN trained from scratch. Its architecture
  is not finalized yet.
- `results/` contains generated CV results, checkpoints, and prediction CSVs.
- `archive/` keeps older Sensor.Community work that is no longer part of the
  current EEA reference-station pipeline.

## Modeling Workflow

Run commands from the repository root. On JupyterHub, first make sure the repo
can see the `data/` folder in the expected location.

```bash
python 06_models/00_assign_folds.py
python check_model_data.py
python 06_models/check_shared_logic.py
python 06_models/01_train_cv.py
python 06_models/02_train_final.py
python 06_models/03_predict_test.py
```

`00_assign_folds.py` should only be run when `station_fold.csv` genuinely needs
to be created or intentionally regenerated. The fold file defines the fixed
geographic split used by cross-validation, final training, and sealed test
prediction. Changing it changes the experiment.

The steps are:

- `00_assign_folds.py`: creates the fixed station-to-fold table and a map for
  visual checking.
- `check_model_data.py`: checks whether the real data files and patch folders
  needed by the models are present and usable.
- `check_shared_logic.py`: checks that important modeling assumptions still
  behave as expected in code.
- `01_train_cv.py`: trains and validates on development folds only.
- `02_train_final.py`: trains the final model using the CV-selected epoch count.
- `03_predict_test.py`: evaluates the final checkpoint on the sealed TEST
  stations.

To run only a few development folds while testing code:

```bash
python 06_models/01_train_cv.py --folds fold1_iberia fold2_france
```

Subset runs are useful for debugging, but they are not a full CV result.

## Required Data

The current pipeline expects these files and folders:

```text
data/processed/daily_avg/eea/pm_reference_stations_2024.csv
data/processed/uba/station_land.csv
data/processed/eea/station_fold.csv

data/processed/satellite_eea/high_res_multispec/
data/processed/satellite_eea/low_res_multispec/
data/processed/satellite_eea/no2_tropomi/
data/processed/satellite_eea/co_tropomi/
data/processed/satellite_eea/aer_wide_tropomi/
data/processed/satellite_eea/dem_glo30/
```

- `pm_reference_stations_2024.csv` contains daily official reference-station
  pollution labels and station coordinates. The modeling code averages the 2024
  values per station to get annual PM10 and PM2.5 labels.
- `station_land.csv` maps German station codes to German federal states
  (Laender). This is needed because only northern/eastern German states are in
  the sealed TEST region; other German stations stay in the development folds.
- `station_fold.csv` stores one row per station with `station_code`, `country`,
  `land`, `fold`, `lat`, and `lon`. Training, final fitting, and test prediction
  all read this file.
- `high_res_multispec/` contains high-resolution Sentinel-2 patches. The model
  receives them as 10 x 120 x 120 tensors.
- `low_res_multispec/` contains lower-resolution Sentinel-2 context patches. The
  model receives them as 10 x 60 x 60 tensors.
- `no2_tropomi/` contains local Sentinel-5P NO2 patches, used as 5 x 5 inputs.
  NO2 is nitrogen dioxide and is related to combustion sources such as traffic.
- `co_tropomi/` contains local Sentinel-5P CO patches, also used as 5 x 5
  inputs. CO is carbon monoxide and gives another atmospheric signal.
- `aer_wide_tropomi/` contains wider aerosol context patches, used as 31 x 31
  inputs. Aerosol measurements are relevant because PM2.5 is particulate matter.
- `dem_glo30/` contains elevation patches, used as 60 x 60 inputs. Terrain can
  affect pollution transport and station surroundings.

Stations without every required patch are excluded from the model-ready data.
This keeps all models trained and evaluated on stations with the same available
input types.

## Geographic Folds And Leakage Prevention

Random cross-validation is a poor fit for spatial pollution data. Nearby
stations often share weather, land use, emissions, and satellite patterns. If a
nearby station is in training while its neighbor is in validation, the validation
score can look better than it should because the model has effectively seen very
similar geography.

For that reason, this project uses geographic folds. A fold is a group of
stations held out together during development. `station_fold.csv` stores the
assignment for each station.

The eight development folds are:

```text
fold1_iberia    PT, ES, AD
fold2_france    FR, NL, BE, LU
fold3_italy     IT, MT
fold4_alpine    south/west DE plus CH, AT
fold5_north     DK, SE, NO, FI, IS, IE, LT, LV, EE
fold6_balkan_e  HU, SI, HR, BA, RS, XK, ME, RO, BG
fold7_balkan_s  AL, GR, CY, TR, MK
fold8_poland    PL, CZ, SK
```

The sealed TEST region is northern/eastern Germany. In the current fold logic it
contains German stations from:

```text
Brandenburg
Mecklenburg-Vorpommern
Sachsen
Sachsen-Anhalt
Thueringen
Berlin
Hamburg
Bremen
Niedersachsen
Schleswig-Holstein
```

TEST is not a ninth development fold. It must not be used when choosing
architectures, learning rates, augmentation, or other hyperparameters.

The 100 km buffer is an extra leakage-prevention step. During CV, after one fold
is selected for validation, training stations within 100 km of any validation
station are removed from that fold's training data. During final training,
development stations within 100 km of the sealed TEST stations are removed. The
scripts record how many training stations were removed so ResNet and CNN runs
can be checked against the same geography.

## Models Being Compared

### `resnet_frozen`

`resnet_frozen` uses a BigEarthNet-pretrained ResNet50 for the high-resolution
Sentinel-2 branch. The pretrained ResNet backbone is frozen, so its weights do
not update during pollution training. Only the pollution-specific projection,
auxiliary branches, and regression head are trained.

This is transfer learning as a fixed feature extractor: the model uses
remote-sensing features learned elsewhere, then learns how to combine them for
annual PM2.5 prediction.

Current checked parameter counts:

```text
total      23,766,049
trainable     236,065
```

### `resnet_layer4`

`resnet_layer4` starts from the same pretrained ResNet50. The stem, layer1,
layer2, and layer3 remain frozen. Non-BatchNorm parameters in layer4 are
trainable. Pretrained BatchNorm layers stay frozen and in evaluation mode, even
after `model.train()` is called.

This is called partial fine-tuning because only the final stage of the
pretrained image encoder is allowed to adapt to the pollution task. The new
pollution parameters use `lr_head`, while trainable pretrained layer4
parameters use the smaller `lr_layer4`.

Current checked parameter counts:

```text
total      23,766,049
trainable  15,178,273
layer4     14,942,208
```

The much larger trainable parameter count means `resnet_layer4` can adapt more
than `resnet_frozen`, but it also has more ways to overfit. It should be judged
using development-fold CV before any sealed TEST evaluation.

### `cnn`

`cnn/` is reserved for the required CNN trained from scratch. It should use the
same station assignments, 100 km buffer, preprocessing, target transformation,
metrics, and sealed TEST setup as the ResNet experiments. Its architecture is
intentionally not finalized yet.

## Shared Multimodal Architecture

The ResNet model combines several inputs around each station:

```text
high-resolution Sentinel-2 patch  -> pretrained ResNet50 -> projection
low-resolution Sentinel-2 patch   -> small CNN branch
Sentinel-5P NO2 patch             -> small atmospheric branch
Sentinel-5P CO patch              -> small atmospheric branch
wide aerosol context              -> small CNN branch
DEM elevation patch               -> small CNN branch
scalar/context values             -> patch means and elevation

all learned representations + scalar/context values -> regression head -> annual PM2.5
```

The comparison with the future scratch CNN should keep the non-high-resolution
branches and fusion setup the same where possible. That way, the main difference
being tested is the high-resolution image encoder: pretrained ResNet features
versus features learned from scratch.

## Training And Evaluation

The current target is `pm25`, but the shared code can also carry `pm10`.

Targets are transformed inside the dataset:

- the station concentration is log-transformed;
- the log values are standardized using the training split mean and standard
  deviation;
- predictions are transformed back with `exp()` before RMSE, MAE, and R2 are
  reported.

Training uses `SmoothL1Loss` with a mask so missing pollutant labels can be
ignored. The optimizer is `AdamW`. In frozen ResNet mode, all trainable
parameters use `lr_head`. In layer4 mode, new pollution parameters use
`lr_head`, and trainable pretrained layer4 parameters use `lr_layer4`.

`NUM_WORKERS` in `shared/config.py` controls how many extra worker processes
PyTorch uses for each DataLoader. It currently defaults to `0` because the
university JupyterHub container has very limited shared memory (`/dev/shm` is
64 MB). Multiple workers can exceed that limit with these large multimodal
batches and cause bus errors before training starts. Using `0` loads data in the
main process. This changes runtime behavior only; it does not change the model,
folds, inputs, labels, metrics, or scientific experiment.

With `NUM_WORKERS=0`, patch files are read serially in the main process.
Repeated `.npy` reads can become slow because the same station patches are used
again every epoch. The shared Dataset therefore keeps loaded patches in ordinary
process RAM when `CACHE_PATCHES=True`. The cache is lazy: a patch is loaded from
disk the first time it is requested, then reused from RAM later. Cached arrays
are copied before they are returned, so random rotation/flip augmentation cannot
alter the stored base patch. This is only a runtime optimization and does not
change the model inputs or experiment.

`CPU_THREADS` and `CPU_INTEROP_THREADS` limit PyTorch CPU parallelism at script
startup. The current defaults are 8 intra-op threads and 4 inter-op threads.
This keeps small CPU-side tensor, augmentation, and collation work from trying
to use every CPU visible inside the JupyterHub container. The scripts print the
effective values at startup so the runtime setting can be checked.

CV training can run for up to `CV_EPOCHS` epochs. Early stopping watches the
mean validation RMSE across configured pollutants. `best_epoch` is the one-based
epoch number with the best validation RMSE. Final training does not use a held
out fold; its epoch count is the ceiling of the median CV `best_epoch` values.

Each CV epoch also prints a compact timing line for the training pass, average
training batch time, train-subset evaluation, validation evaluation with the
current TTA setting, and total epoch time. This is only logging; it is there to
help diagnose runtime bottlenecks without changing the model or experiment.
The training pass also prints a short detail line for DataLoader waiting,
CPU-to-device transfer, forward/loss calculation, backward pass, and optimizer
work so slow epochs can be traced more precisely.

Metrics:

- RMSE is root mean squared error in the original concentration units.
- MAE is mean absolute error in the original concentration units.
- R2 compares the model to predicting the validation mean. It can be negative
  when the model is worse than that simple comparison.
- The train-mean baseline predicts the arithmetic mean concentration from the
  training split. This is intentionally not the mean of log values.

Different summaries answer different questions:

- per-fold metrics show each held-out development fold separately;
- macro CV metrics average the fold metrics, giving each completed fold equal
  weight;
- pooled out-of-fold metrics combine all validation predictions first and then
  compute metrics on the pooled table.

Do not compare these summaries as if they were the same statistic.

## Augmentation And TTA

During training, the dataset applies random rotations by 0, 90, 180, or 270
degrees and a random horizontal flip to the patch-like inputs. This assumes that
the pollution signal should not depend on the arbitrary orientation of the patch
array.

TTA means test-time augmentation. When enabled, validation and test prediction
run eight views of each station patch: four rotations and a flipped version of
each rotation. The model predictions from those views are averaged.

Rotation and flip behavior is part of the current experimental setup. It may be
evaluated later as a methodological question, but it should not be changed
casually because ResNet and CNN comparisons should keep preprocessing
consistent.

## Results And Experiment Identity

New results are separated by experiment:

```text
results/resnet_frozen/
results/resnet_layer4/
results/cnn/
```

Each experiment folder can contain:

- `eea_cv_results.json`: per-fold CV metrics, baseline metrics, buffer counts,
  parameter counts, epoch metadata, and the configuration used for the run;
- `eea_cv_predictions.csv`: validation predictions with station metadata;
- `final_model.pt`: the final checkpoint plus normalization values, S5P
  training statistics, baseline concentration, buffer metadata, parameter
  counts, and CV-derived epoch information;
- `test_predictions.csv`: sealed TEST predictions with station metadata.

This metadata matters because a result is only interpretable when we know which
fold file, model variant, learning rates, TTA setting, and buffer setting
produced it.

The older `results/resnet/` directory may contain earlier frozen-ResNet outputs.
Leave those files as historical outputs unless their exact configuration is
known.

Generated checkpoints and large result files should usually stay out of Git
unless they are intentionally needed for hand-in or reproducibility.

## Logic Checks

`check_model_data.py` and `06_models/check_shared_logic.py` answer different
questions:

```text
check_model_data.py
-> Are the real input files present and usable?

06_models/check_shared_logic.py
-> Does the modeling code behave according to the intended experimental setup?
```

Run the shared logic check with:

```bash
python 06_models/check_shared_logic.py
```

It is a lightweight assertion-based smoke test. No terminal output means all
checks passed.

It currently checks:

- development-fold filtering;
- exclusion of TEST and UNASSIGNED folds where appropriate;
- the 100 km spatial leakage buffer;
- DataLoader partial-batch behavior;
- singleton-batch handling for BatchNorm safety;
- patch-cache reuse and cached-array mutation protection;
- arithmetic train-mean baseline behavior;
- prediction metadata;
- separate result paths for each experiment;
- the CV-derived final epoch rule;
- frozen ResNet trainability;
- layer4 trainability;
- pretrained BatchNorm freezing;
- BatchNorm staying in eval mode after `model.train()`;
- optimizer parameter groups;
- expected parameter counts;
- experiment metadata;
- dummy forward-pass output shape.

It does not test:

- real satellite-data completeness;
- actual model convergence;
- model quality;
- CV performance;
- sealed TEST performance.
