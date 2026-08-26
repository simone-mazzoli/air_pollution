import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shared import data, evaluation, folds, runtime, training
from shared.config import CACHE_PATCHES, DEVICE, SEED, result_paths, training_config
from shared.models import SUPPORTED_EXPERIMENTS, require_single_experiment, selected_model


def final_epoch_summary_from_selection(selection_path, experiment_name, expected_folds):
    if not selection_path.exists():
        raise SystemExit(
            f"ERROR: final model-selection record not found: {selection_path}. "
            "Do not derive final epochs from a later CV rerun."
        )
    record = json.loads(selection_path.read_text())
    if record.get("selected_architecture") != experiment_name:
        raise SystemExit(
            f"ERROR: {selection_path} selects {record.get('selected_architecture')!r}, "
            f"not {experiment_name!r}."
        )
    by_fold = record.get("original_best_epochs_by_fold", {})
    missing = [fold for fold in expected_folds if fold not in by_fold]
    if missing:
        raise SystemExit(f"ERROR: model-selection record is missing folds: {missing}")
    fold_best_epochs = [(fold, int(by_fold[fold])) for fold in expected_folds]
    return {
        "final_epochs": int(record["fixed_final_epochs"]),
        "best_epochs": [epoch for _, epoch in fold_best_epochs],
        "fold_best_epochs": fold_best_epochs,
        "median_best_epoch": float(record["median_best_epoch"]),
        "epoch_selection_rule": "recorded_before_test_median_ceil",
        "epoch_source": str(selection_path),
    }


def print_final_training_setup(epoch_summary, buffer_km, n_before_buffer, buffer_removed, loader_info):
    print("\nFinal training setup")
    print("--------------------")
    print("CV best epochs:")
    width = max(len(fold) for fold, _ in epoch_summary["fold_best_epochs"])
    for fold, best_epoch in epoch_summary["fold_best_epochs"]:
        print(f"  {fold:<{width}}: {best_epoch}")
    print(f"\nMedian CV best epoch: {epoch_summary['median_best_epoch']:g}")
    print(f"Epoch record: {epoch_summary['epoch_source']}")
    print("Epoch selection rule: ceil(median)")
    print(f"Final training epochs: {epoch_summary['final_epochs']}")
    print(f"\nDevelopment stations before TEST buffer: {n_before_buffer}")
    print(f"Removed by {buffer_km:g} km TEST buffer: {buffer_removed}")
    print(f"Final training stations: {loader_info['n_train_samples']}")
    print(f"Batches per epoch: {loader_info['n_train_batches']}\n")


def parse_args():
    choices = "{" + ",".join(SUPPORTED_EXPERIMENTS) + "}"
    ap = argparse.ArgumentParser(description="Train one selected experiment on all development folds.")
    ap.add_argument("--experiment", default="cnn_deep_wide", metavar=choices)
    ap.add_argument("--wide", action="store_true",
                    help="use wider scratch-CNN channels with --experiment cnn or cnn_deep")
    return ap.parse_args()


def main():
    runtime.apply_runtime_config()
    print(runtime.runtime_summary())
    args = parse_args()
    experiment_name = require_single_experiment(args.experiment, "Final training", wide=args.wide)
    data.seed_everything()
    build_model, model_config = selected_model(experiment_name, wide=args.wide)
    result = result_paths(model_config["experiment"])
    epoch_summary = final_epoch_summary_from_selection(
        result["final_model_selection"], model_config["experiment"], folds.development_fold_names())
    final_epochs = epoch_summary["final_epochs"]
    cv_best_epochs = epoch_summary["best_epochs"]
    epoch_rule = epoch_summary["epoch_selection_rule"]
    cfg = training_config(model_config, epochs=final_epochs)
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]

    df = data.load_frame(streams, cfg)
    test_df = data.load_test_frame(streams, cfg)
    buffer_removed = 0
    n_before = len(df)
    if cfg["buffer_km"] > 0:
        df = data.buffer_exclude(df, test_df, cfg["buffer_km"]).reset_index(drop=True)
        buffer_removed = n_before - len(df)
    print(f"\ndevice: {DEVICE}  |  seed: {SEED}  |  experiment: {model_config['experiment']}  |  final model on ALL {len(df)} "
          f"CV stations, {cfg['epochs']} epochs (no held-out fold)\n")

    tmean = np.array([np.nanmean(np.log(df[p].values)) for p in cfg["pollutants"]], "float64")
    tstd = np.array([np.nanstd(np.log(df[p].values)) or 1.0 for p in cfg["pollutants"]], "float64")
    s5p_stats = data.compute_s5p_stats(df, streams, cfg)
    tr, loader_info = training.train_loader(data.EEA(df, streams, tmean, tstd, s5p_stats, cfg, augment=True), cfg)
    print_final_training_setup(epoch_summary, cfg["buffer_km"], n_before, buffer_removed, loader_info)
    print(f"patch cache: {'enabled' if cfg.get('cache_patches', CACHE_PATCHES) else 'disabled'}")

    model = build_model(len(streams), cfg, len(cfg["pollutants"])).to(DEVICE)
    counts = training.parameter_counts(model)
    meta = training.model_metadata(model, cfg)
    print(f"parameters: total={counts['total']} trainable={counts['trainable']} frozen={counts['frozen']}")
    print(f"batches: n_train_samples={loader_info['n_train_samples']} "
          f"n_train_batches={loader_info['n_train_batches']} "
          f"effective_drop_last={loader_info['effective_drop_last']}")
    opt = torch.optim.AdamW(training.optimizer_parameter_groups(model, cfg))
    lossf = nn.SmoothL1Loss(reduction="none")

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        for xh, xl, xs_patch, xw, xd, xs_mean, y, m in tr:
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
                xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
                xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
            opt.zero_grad()
            loss = training.masked_loss(model(xh, xl, xs_patch, xw, xd, xs_mean), y, m, lossf)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xh)
        print(f"  [{ep:02d}] loss={tot/len(df):.3f}")

    stats = data.patch_cache_stats(cfg.get("cache_patches", CACHE_PATCHES))
    if stats["enabled"]:
        print(f"patch cache final: items={stats['items']} hits={stats['hits']} misses={stats['misses']}")

    baseline = evaluation.constant_baseline(df, df, cfg)
    bundle = {
        "model": model_config["model"],
        **meta,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": cfg,
        "streams": streams,
        "pollutants": list(cfg["pollutants"]),
        "tmean": tmean,
        "tstd": tstd,
        "s5p_stats": s5p_stats,
        "baseline_concentration": {p: baseline[p]["mean"] for p in cfg["pollutants"]},
        "buffer_km": cfg["buffer_km"],
        "buffer_removed_train_stations": buffer_removed,
        "n_train_before_buffer": n_before,
        "cv_best_epochs": cv_best_epochs,
        "final_epochs": final_epochs,
        "epoch_selection_rule": epoch_rule,
        **loader_info,
        "parameter_counts": counts,
    }
    Path(result["final_checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, result["final_checkpoint"])
    print(f"\nsaved {result['final_checkpoint']}")


if __name__ == "__main__":
    main()
