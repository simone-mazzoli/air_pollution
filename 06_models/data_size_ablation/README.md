# Data-Size Ablation

This experiment asks whether the gap between the selected scratch CNN
(`cnn_deep_wide`) and the frozen pretrained ResNet (`resnet_frozen`) changes
when less labelled training data is available.

It keeps the same geographic cross-validation setup as the main model runs:
one development fold is held out for validation, the usual 100 km buffer is
applied around that validation fold, and only then the remaining training
stations are subsampled. The sealed German TEST split is not loaded or used.

Sampling is proportional within the original source folds. For a given
validation fold and seed, each source fold gets one deterministic random order,
so the 25% subset is contained in the 50% subset, which is contained in the
full buffered training pool. The same saved station IDs are used for both
models, making the comparison paired.

Random seeds only affect which eligible training stations are selected for the
reduced-data runs. They do not change the validation fold. Full-data 100% rows
are read from the current canonical CV outputs in `06_models/results/` rather
than retrained here.

Run the configured reduced-data experiment:

```bash
python 06_models/data_size_ablation/run_ablation.py
```

Run one small job while debugging:

```bash
python 06_models/data_size_ablation/run_ablation.py --model cnn_deep_wide --fraction 0.25 --seed 1 --fold fold1_iberia
```

Rebuild summaries from completed runs:

```bash
python 06_models/data_size_ablation/run_ablation.py --summarize-only
```

Plot completed ablation outputs:

```bash
python 06_models/data_size_ablation/plot_results.py
```

This creates epoch-wise objective-loss and validation-performance curves for
any completed reduced-data run with a saved history, plus separate data-size
learning curves for pooled CV RMSE and MAE.

Outputs are written under `06_models/data_size_ablation/results/`:

- `sampled_station_ids.csv`: exact sampled training station IDs.
- `runs/`: one JSON, prediction CSV, and history CSV per completed run.
- `data_size_runs.csv`: fold-level metrics and counts.
- `data_size_summary.csv`: pooled out-of-fold metrics per model, fraction, and seed.
- `data_size_summary_by_fraction.csv`: seed mean/std summaries.
- `model_gap_summary.csv`: paired ResNet minus CNN performance gaps.
- `figures/`: epoch-wise curves, data-size RMSE/MAE curves, and RMSE/MAE gap
  plots.
