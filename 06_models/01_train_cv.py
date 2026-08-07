import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared import data, evaluation, folds, training
from shared.config import CV_EPOCHS, CV_FOLDS, DEVICE, DISPLAY, MODEL, SEED, result_paths, training_config
from shared.models import selected_model


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train cross-validation folds using the Python config defaults."
    )
    ap.add_argument("--folds", nargs="+", default=CV_FOLDS,
                    help="optional subset of fold names to run")
    return ap.parse_args()


def main():
    args = parse_args()
    data.seed_everything()
    build_model, model_config = selected_model(MODEL)
    result = result_paths(model_config["experiment"])
    cfg = training_config(model_config, epochs=CV_EPOCHS, folds=args.folds)
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]
    df = data.load_frame(streams, cfg)
    run_folds = cfg["folds"] or folds.development_fold_names()

    print(f"\ndevice: {DEVICE}  |  seed: {SEED}  |  model: {MODEL}  |  running folds: {run_folds}")
    print(f"config: {json.dumps({k: v for k, v in cfg.items() if k != 'folds'})}\n")
    cv, scatter = {}, []
    for fold in run_folds:
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        if len(val_df) < 10:
            print(f"########## {fold}: only {len(val_df)} stations, SKIPPING ##########\n")
            continue
        buffer_removed = 0
        n_before = len(train_df)
        if cfg["buffer_km"] > 0:
            train_df = data.buffer_exclude(train_df, val_df, cfg["buffer_km"]).reset_index(drop=True)
            buffer_removed = n_before - len(train_df)
            print(f"  buffer {cfg['buffer_km']:g}km: dropped {buffer_removed}/{n_before} "
                  "train stations near val border")
        print(f"########## VAL FOLD: {fold}  (train {len(train_df)}, val {len(val_df)}) ##########")
        res, arrs = training.train_one_fold(train_df, val_df, streams, cfg, build_model)
        res["buffer_km"] = cfg["buffer_km"]
        res["buffer_removed_train_stations"] = buffer_removed
        res["n_train_before_buffer"] = n_before
        print(f"  FINAL [{fold}]:")
        print(f"    {evaluation.fmt_metrics({p: res[p] for p in cfg['pollutants']}, cfg)}\n")
        cv[fold] = res
        for p in cfg["pollutants"]:
            if p in arrs:
                pred, true = arrs[p]
                scatter.extend(evaluation.prediction_table(val_df, {p: (pred, true)}, cfg))

    print("=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    done = list(cv)
    if done:
        overall_model_rmses = []
        for p in cfg["pollutants"]:
            rmses = [cv[f][p]["rmse"] for f in done if not np.isnan(cv[f][p]["rmse"])]
            maes = [cv[f][p]["mae"] for f in done if not np.isnan(cv[f][p]["mae"])]
            r2s = [cv[f][p]["r2"] for f in done if not np.isnan(cv[f][p]["r2"])]
            bases = [cv[f][p]["baseline"] for f in done if not np.isnan(cv[f][p]["baseline"])]
            overall_model_rmses.extend(rmses)
            print(f"  {DISPLAY[p]}: mean-of-folds RMSE={np.mean(rmses):.2f}  "
                  f"(baseline RMSE={np.mean(bases):.2f})  MAE={np.mean(maes):.2f}  "
                  f"R2 mean={np.mean(r2s):+.3f} "
                  f"(positive in {sum(r > 0 for r in r2s)}/{len(r2s)})")
            for f in done:
                print(f"    {f:<16} RMSE={cv[f][p]['rmse']:.2f}  "
                      f"(baseline={cv[f][p]['baseline']:.2f})  MAE={cv[f][p]['mae']:.2f}  "
                      f"R2={cv[f][p]['r2']:+.3f}  (n={cv[f][p]['n']})")
        print(f"\n  OVERALL (mean of folds, all pollutants pooled): RMSE={np.mean(overall_model_rmses):.2f}")
        predictions = pd.concat(scatter, ignore_index=True)
        cv["pooled_out_of_fold"] = evaluation.metrics_from_prediction_table(predictions, cfg)
        print("\n  POOLED OUT-OF-FOLD")
        for p in cfg["pollutants"]:
            r = cv["pooled_out_of_fold"][p]
            print(f"    {DISPLAY[p]}: RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  "
                  f"R2={r['r2']:+.3f}  (n={r['n']})")
        result["cv_predictions"].parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(result["cv_predictions"], index=False)
        print(f"saved {result['cv_predictions']}")
    cv["_config"] = {k: v for k, v in cfg.items() if k != "folds"}
    result["cv_results"].parent.mkdir(parents=True, exist_ok=True)
    result["cv_results"].write_text(json.dumps(cv, indent=2))
    print(f"saved {result['cv_results']}")


if __name__ == "__main__":
    main()
