from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shared import data, evaluation, folds, training
from shared.config import CACHE_PATCHES, DEVICE, MODEL, SEED, result_paths, training_config
from shared.models import selected_model


def main():
    data.seed_everything()
    build_model, model_config = selected_model(MODEL)
    result = result_paths(model_config["experiment"])
    final_epochs, cv_best_epochs, epoch_rule = training.final_epochs_from_cv(
        result["cv_results"], folds.development_fold_names())
    cfg = training_config(model_config, epochs=final_epochs)
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]

    df = data.load_frame(streams, cfg)
    test_df = data.load_test_frame(streams, cfg)
    buffer_removed = 0
    n_before = len(df)
    if cfg["buffer_km"] > 0:
        df = data.buffer_exclude(df, test_df, cfg["buffer_km"]).reset_index(drop=True)
        buffer_removed = n_before - len(df)
        print(f"buffer {cfg['buffer_km']:g}km: dropped {buffer_removed}/{n_before} "
              "train stations near NE-Germany")
    print(f"\ndevice: {DEVICE}  |  seed: {SEED}  |  model: {MODEL}  |  final model on ALL {len(df)} "
          f"CV stations, {cfg['epochs']} epochs (no held-out fold)\n")

    tmean = np.array([np.nanmean(np.log(df[p].values)) for p in cfg["pollutants"]], "float64")
    tstd = np.array([np.nanstd(np.log(df[p].values)) or 1.0 for p in cfg["pollutants"]], "float64")
    s5p_stats = data.compute_s5p_stats(df, streams, cfg)
    tr, loader_info = training.train_loader(data.EEA(df, streams, tmean, tstd, s5p_stats, cfg, augment=True), cfg)
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
        "model": MODEL,
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
