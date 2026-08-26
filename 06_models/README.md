# 06 Models

This folder contains the modeling part. The main task here is to predict 
annual PM2.5 concentration at monitoring stations from satellite and 
environmental data.

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
Eastern Germany are kept as a sealed final test region. 

The final development comparison is frozen:

- `cnn_deep_wide`: the selected scratch CNN (`cnn_deep --wide`);
- `resnet_frozen`: the BigEarthNet-pretrained ResNet50 with frozen backbone.

BigEarthNet is a large remote-sensing dataset. Pretraining on it gives the
ResNet useful satellite-image features before it is trained for pollution. The
scratch CNN shows what can be learned directly from the pollution dataset.

## Directory Overview

```text
06_models/
|-- 00_assign_folds.py
|-- 01_train_cv.py
|-- 02_train_final.py
|-- 03_predict_test.py
|-- run_experiment_suite.py
|-- plot_learning_curves.py
|-- summarize_cv_results.py
|-- check_shared_logic.py
|-- shared/
|-- resnet/
|-- cnn/
|-- data_size_ablation/
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
  TEST stations. For the selected `cnn_deep_wide` model, the final epoch count
  is read from the saved pre-TEST model-selection record.
- `03_predict_test.py` loads the final checkpoint and evaluates the sealed
  northern/eastern Germany TEST stations.
- `run_experiment_suite.py` is the final development-suite entry point. It runs
  CV, data-size ablation, summaries, and plots. It does not train the final
  model or evaluate TEST.
- `plot_learning_curves.py` creates epoch-wise learning-curve figures from
  saved CV histories. It does not train models.
- `summarize_cv_results.py` reads completed CV experiment result folders and
  writes comparison tables under `results/summary/`.
- `check_shared_logic.py` is a small assertion-based check for modeling logic.
  It prints nothing when all checks pass.
- `shared/` contains data loading, folds, metrics, training helpers, paths, and
  shared configuration.
- `resnet/` contains the BigEarthNet-pretrained ResNet model and ResNet-specific
  settings.
- `cnn/` contains the scratch-CNN model and its CNN-specific settings.
- `data_size_ablation/` contains the optional reduced-training-data experiment
  and plots of model performance as labelled training data decreases.
- `results/` contains generated CV results, checkpoints, and prediction CSVs.
- `archive/` keeps older Sensor.Community work that is no longer part of the
  current EEA reference-station pipeline.

## Modeling Workflow

Run commands from the repository root. On JupyterHub, first make sure the repo
can see the `data/` folder in the expected location.

```bash
python3 06_models/00_assign_folds.py
python3 check_model_data.py
python3 06_models/check_shared_logic.py

# 1. Preview the final development suite
python3 06_models/run_experiment_suite.py --all --dry-run

# 2. Run the final development suite
python3 06_models/run_experiment_suite.py --all

# 3. Rebuild summaries and plots later without retraining
python3 06_models/run_experiment_suite.py --postprocess

# 4. Final training and TEST evaluation are separate, explicit steps only
python3 06_models/02_train_final.py --experiment cnn_deep --wide
python3 06_models/03_predict_test.py --experiment cnn_deep --wide
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
- `01_train_cv.py`: trains and validates development CV folds only. Its
  `--experiment all` mode now means the final report comparison:
  `cnn_deep_wide` and `resnet_frozen`.
- `summarize_cv_results.py`: by default compares only the two final complete-CV
  experiments. Use `--all-existing` to classify diagnostic and historical
  result folders.
- `plot_learning_curves.py`: plots training behavior over epochs using
  `cv_history.csv`. New histories include both training and validation
  SmoothL1 objective loss. RMSE/MAE are plotted separately in original PM2.5
  units.
- `02_train_final.py`: after one experiment has been chosen, trains a fresh
  model for that experiment on all development folds. For `cnn_deep_wide`, it
  uses `results/cnn_deep_wide/final_model_selection.json` so the final epoch
  count stays at the recorded pre-TEST value. Fold models are not merged.
- `03_predict_test.py`: evaluates that one selected final checkpoint on the
  sealed TEST stations.

The supported experiment names are:

```text
cnn
cnn_deep
cnn_deep_wide
resnet_frozen
resnet_full
```

Use `--experiment` to choose which one to run:

```bash
python3 06_models/01_train_cv.py --experiment cnn
```

To run the final CV comparison:

```bash
python3 06_models/01_train_cv.py --experiment all
```

To run the deeper scratch-CNN ablation:

```bash
python3 06_models/01_train_cv.py --experiment cnn_deep
```

To widen either scratch CNN:

```bash
python3 06_models/01_train_cv.py --experiment cnn --wide
python3 06_models/01_train_cv.py --experiment cnn_deep --wide
```

To run only a few development folds while testing code:

```bash
python3 06_models/01_train_cv.py --experiment all --folds fold1_iberia
```

Subset runs are useful for debugging, but they are not a full CV result. The
`all` option is only for development CV. Final training and sealed TEST
prediction require one explicitly selected experiment.

Current result-folder classification:

```text
MAIN
  cnn_deep_wide - full 8-fold CV
  resnet_frozen - full 8-fold CV

SUPPLEMENTARY
  cnn - full 8-fold CV
  cnn_deep - fold1_iberia diagnostic
  cnn_large - fold1_iberia width-only diagnostic
  resnet_full - fold1_iberia diagnostic
  data-size ablation - cnn_deep_wide vs resnet_frozen

HISTORICAL / SUPERSEDED
  resnet
  resnet_layer4
```

The one-fold runs are deliberately low-cost architecture/fine-tuning
diagnostics. They should not be read as estimates of overall European
geographic generalization.

## Learning-Curve Plots

An epoch-wise learning curve shows how one training run behaves as epochs pass.
For new final-suite histories, the objective-loss plot shows `train_loss` and
`val_loss` on the same SmoothL1-loss axis. The validation-performance plot
separately shows RMSE and MAE in PM2.5 units.

```bash
python3 06_models/plot_learning_curves.py
python3 06_models/plot_learning_curves.py --experiment cnn_deep_wide
python3 06_models/plot_learning_curves.py --all-existing
```

Outputs are written under:

```text
06_models/results/<experiment>/figures/
```

The data-size learning curve is different: it shows model performance as the
fraction of labelled training stations changes. Those plots live with the
ablation experiment:

```bash
python3 06_models/data_size_ablation/plot_results.py
```

Outputs are written under:

```text
06_models/data_size_ablation/results/figures/
```

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

### `resnet_full`

`resnet_full` starts from the same pretrained ResNet50. All non-BatchNorm
pretrained backbone weights are trainable at a smaller learning rate. Pretrained
BatchNorm layers stay frozen and in evaluation mode, even after `model.train()`
is called.

This is full-backbone fine-tuning with conservative BatchNorm handling. The new
pollution parameters use `lr_head`, while trainable pretrained backbone
parameters use the smaller `lr_backbone`.

Current checked parameter counts:

```text
total      23,766,049
trainable  23,712,929
backbone   23,476,864
frozen         53,120
```

The previous `resnet_layer4` development experiment is superseded because CV
looked essentially identical to `resnet_frozen`. Its old result folder remains
historical output, but it is no longer an active experiment choice.

### `cnn`

`cnn` uses a scratch high-resolution Sentinel-2 encoder instead of the
BigEarthNet-pretrained ResNet50. It still uses the same station assignments,
100 km buffer, preprocessing, target transformation, metrics, and sealed TEST
setup as the ResNet experiments.

The default scratch encoder uses channel widths `32, 64, 128, 256`.
`cnn_deep` keeps those widths but uses three Conv-BN-ReLU layers per block
instead of two. `--wide` changes the widths to `48, 96, 192, 384`. All
downstream multimodal branches and training settings stay the same.

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

The comparison with the scratch CNN keeps the non-high-resolution branches and
fusion setup the same where possible. That way, the main difference being tested
is the high-resolution image encoder: pretrained ResNet features versus features
learned from scratch.

## Training And Evaluation

The current target is `pm25`, but the shared code can also carry `pm10`.
Station inclusion follows the configured pollutant list. A PM2.5-only run keeps
stations with valid PM2.5 labels, a PM10-only run keeps stations with valid PM10
labels, and a joint PM10+PM2.5 run keeps stations with at least one of those
configured targets.

Targets are transformed inside the dataset:

- the station concentration is log-transformed;
- the log values are standardized using the training split mean and standard
  deviation;
- predictions are transformed back with `exp()` before RMSE, MAE, and R2 are
  reported.

Training uses `SmoothL1Loss`. When more than one pollutant is configured, a mask
lets the model ignore missing labels for the pollutant that is absent at a
station. In single-pollutant runs, stations without that pollutant are filtered
out before they enter the Dataset. The optimizer is `AdamW`. In frozen ResNet
mode, all trainable parameters use `lr_head`. In full ResNet mode, new pollution
parameters use `lr_head`, and trainable pretrained backbone parameters use
`lr_backbone`. In CNN mode, all trainable CNN parameters currently share one
optimizer group using `lr` and `weight_decay`.

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
out fold. For the saved `cnn_deep_wide` final model, the epoch count is fixed in
`results/cnn_deep_wide/final_model_selection.json`.

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

Active report results are separated by experiment:

```text
results/cnn_deep_wide/   main full CV
results/resnet_frozen/   main full CV
results/cnn/             supplementary full CV
results/cnn_deep/        supplementary fold1_iberia diagnostic
results/cnn_large/       supplementary fold1_iberia width-only diagnostic
results/resnet_full/     supplementary fold1_iberia fine-tuning diagnostic
```

Superseded `resnet/` and `resnet_layer4/` outputs are archived under
`results/archive/historical/`. They are kept as history, not active report
results.

Each experiment folder can contain:

- `cv_history.csv`: one row per completed CV epoch, with the experiment, fold,
  epoch, training loss, validation objective loss for finalized reruns, train
  and validation metrics, optimizer learning rates, timing diagnostics,
  best-so-far flag, and patience counter;
- `cv_folds.csv`: one row per fold and pollutant, with station counts, buffer
  removals, best epoch, validation metrics, baseline RMSE, and parameter counts;
- `eea_cv_results.json`: per-fold CV metrics, baseline metrics, buffer counts,
  parameter counts, epoch metadata, and the configuration used for the run;
- `eea_cv_predictions.csv`: validation predictions with station metadata;
- `run_metadata.json`: experiment configuration, fold setup, 100 km buffer,
  parameter counts, model metadata, Git state, Python/PyTorch/CUDA/device
  information, and run timestamps;
- `final_model.pt`: the final checkpoint plus normalization values, S5P
  training statistics, baseline concentration, buffer metadata, parameter
  counts, and recorded pre-TEST epoch information;
- `final_model_selection.json`: records why the final `cnn_deep_wide`
  checkpoint used 24 epochs;
- `test_predictions.csv`: sealed TEST predictions with station metadata.

This metadata matters because a result is only interpretable when we know which
fold file, model variant, learning rates, TTA setting, and buffer setting
produced it.

After one or more CV runs have finished, create comparison tables with:

```bash
python3 06_models/summarize_cv_results.py
```

The default summarizer writes report-oriented outputs and does not inspect
final-model or sealed TEST outputs:

```text
results/summary/main_model_comparison.csv
results/summary/main_fold_comparison.csv
results/summary/supplementary_full_cv.csv
results/summary/iberia_diagnostics.csv
```

Use `python3 06_models/summarize_cv_results.py --all-existing` for the
classified history summary under `results/summary/all_existing/`.

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
python3 06_models/check_shared_logic.py
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
- experiment selection routes;
- epoch-history, fold-summary, and run-metadata serialization;
- summary-table behavior with complete and incomplete result folders;
- separate result paths for each experiment;
- the recorded final epoch rule;
- frozen ResNet trainability;
- full-backbone ResNet trainability;
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
