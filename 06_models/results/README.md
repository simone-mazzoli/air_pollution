# Results

This folder contains generated modeling outputs. The active root is reserved
for experiments that are part of the final report workflow.

## Active Report Results

```text
cnn_deep_wide/   MAIN: selected scratch CNN, full 8-fold CV
resnet_frozen/   MAIN: frozen pretrained ResNet, full 8-fold CV
cnn/             SUPPLEMENTARY: compact scratch CNN, full 8-fold CV
cnn_deep/        SUPPLEMENTARY: depth-only Iberia diagnostic
cnn_large/       SUPPLEMENTARY: width-only Iberia diagnostic
resnet_full/     SUPPLEMENTARY: full fine-tuning Iberia diagnostic
summary/         report tables and classified supplementary summaries
```

The one-fold diagnostics are deliberately limited to `fold1_iberia`. They are
low-cost development comparisons, not estimates of full European geographic
generalization.

## Archive

```text
archive/historical/resnet/
archive/historical/resnet_layer4/
archive/pre_val_loss_rerun/
```

`archive/historical/` preserves superseded experiment outputs. In particular,
the old `resnet/` folder contains historical TEST-related outputs; do not use
those as the current final TEST result.

`archive/pre_val_loss_rerun/` is used by the final suite to preserve old CV
histories, folds, predictions, results JSON, and metadata before replacing them
with reruns that include validation objective loss.

Archived results are scientific provenance. They should not be used as the
default final comparison.

## Summary Outputs

Run:

```bash
python3 06_models/summarize_cv_results.py
```

Report-oriented outputs:

```text
summary/main_model_comparison.csv
summary/main_fold_comparison.csv
summary/supplementary_full_cv.csv
summary/iberia_diagnostics.csv
```

Optional classified provenance summary:

```bash
python3 06_models/summarize_cv_results.py --all-existing
```

This writes:

```text
summary/all_existing/all_experiments_classified.csv
summary/all_existing/all_fold_results_classified.csv
```

## File Meanings

Each experiment directory may contain:

- `cv_history.csv`: epoch-level training history, including `train_loss` and,
  for finalized reruns, `val_loss`.
- `cv_folds.csv`: per-fold validation metrics and counts.
- `eea_cv_predictions.csv`: out-of-fold validation predictions.
- `eea_cv_results.json`: full CV results, including pooled OOF metrics.
- `run_metadata.json`: configuration and environment metadata.
- `figures/`: generated plots that can be recreated from histories/results.

The selected final checkpoint and current TEST predictions stay in:

```text
cnn_deep_wide/final_model.pt
cnn_deep_wide/test_predictions.csv
```

Do not move or overwrite those files during development-suite reruns.
