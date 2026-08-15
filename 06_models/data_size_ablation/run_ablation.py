import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from shared import data, evaluation, experiment, folds, runtime, training
from shared.config import CV_EPOCHS, SEED, result_paths, training_config
from shared.models import selected_model


ABLATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ABLATION_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"

MODELS = {
    "cnn_deep_wide": ("cnn_deep", True),
    "resnet_frozen": ("resnet_frozen", False),
}
FRACTIONS = [0.25, 0.50, 1.00]
SUBSAMPLING_SEEDS = [1, 2, 3]


def fraction_tag(fraction):
    return f"frac_{int(round(fraction * 100)):03d}"


def fraction_seeds(fraction):
    return [None] if fraction == 1.0 else SUBSAMPLING_SEEDS


def run_path(model, fraction, seed, fold):
    seed_dir = "full" if seed is None else f"seed_{seed:03d}"
    return RUNS_DIR / model / fraction_tag(fraction) / seed_dir / f"{fold}.json"


def prediction_path(model, fraction, seed, fold):
    return run_path(model, fraction, seed, fold).with_suffix(".predictions.csv")


def history_path(model, fraction, seed, fold):
    return run_path(model, fraction, seed, fold).with_suffix(".history.csv")


def source_fold_order(train_df, seed):
    rng = np.random.default_rng(seed)
    orders = {}
    for source_fold, sub in train_df.groupby("fold", sort=True):
        codes = sub["station_code"].astype(str).to_numpy()
        orders[source_fold] = rng.permutation(codes).tolist()
    return orders


def sample_count(n, fraction):
    if fraction == 1.0:
        return n
    return min(n, max(1, int(round(n * fraction)))) if n else 0


def sampled_codes_from_orders(orders, fraction):
    codes = []
    counts = {}
    for source_fold, ordered_codes in orders.items():
        n = sample_count(len(ordered_codes), fraction)
        selected = ordered_codes[:n]
        counts[source_fold] = len(selected)
        codes.extend(selected)
    return set(codes), counts


def build_sampling_plan(df, selected_folds, buffer_km=100.0):
    rows, plan = [], {}
    if (df["fold"] == "TEST").any():
        raise AssertionError("sealed TEST stations reached the ablation frame")
    for val_fold in selected_folds:
        val_df = df[df["fold"] == val_fold].reset_index(drop=True)
        train_before = df[df["fold"] != val_fold].reset_index(drop=True)
        train_after = data.buffer_exclude(train_before, val_df, buffer_km).reset_index(drop=True)
        excluded = set(train_before["station_code"]) - set(train_after["station_code"])
        val_codes = set(val_df["station_code"])
        if val_codes & set(train_after["station_code"]):
            raise AssertionError(f"{val_fold}: validation station appears in training")
        plan[val_fold] = {
            "val_codes": val_codes,
            "after_buffer_codes": set(train_after["station_code"]),
            "buffer_excluded_codes": excluded,
            "n_train_before_buffer": len(train_before),
            "n_buffer_dropped": len(excluded),
            "n_train_after_buffer": len(train_after),
            "source_counts_after_buffer": train_after["fold"].value_counts().sort_index().to_dict(),
            "samples": {},
        }
        for seed in SUBSAMPLING_SEEDS:
            orders = source_fold_order(train_after, seed)
            previous_codes = set()
            for fraction in sorted(FRACTIONS):
                codes, counts = sampled_codes_from_orders(orders, fraction)
                if fraction < 1.0 and previous_codes and not previous_codes <= codes:
                    raise AssertionError(f"{val_fold} seed {seed}: subsets are not nested")
                if codes & val_codes:
                    raise AssertionError(f"{val_fold} seed {seed}: validation station sampled")
                if codes & excluded:
                    raise AssertionError(f"{val_fold} seed {seed}: buffer-excluded station sampled")
                plan[val_fold]["samples"][(fraction, seed)] = {"codes": codes, "counts": counts}
                if fraction == 1.0:
                    plan[val_fold]["samples"][(fraction, None)] = {"codes": codes, "counts": counts}
                previous_codes = codes
        ref_full = plan[val_fold]["samples"][(1.0, None)]["codes"]
        if ref_full != plan[val_fold]["after_buffer_codes"]:
            raise AssertionError(f"{val_fold}: full sample differs from buffered training pool")
        for fraction in FRACTIONS:
            seeds = fraction_seeds(fraction)
            ref = None
            for seed in seeds:
                sample = plan[val_fold]["samples"][(fraction, seed)]
                if ref is None:
                    ref = sample["codes"]
                elif fraction == 1.0 and sample["codes"] != ref:
                    raise AssertionError(f"{val_fold}: full sample changed across seeds")
                for code in sorted(sample["codes"]):
                    source_fold = df.loc[df["station_code"] == code, "fold"].iloc[0]
                    rows.append({
                        "validation_fold": val_fold,
                        "fraction": fraction,
                        "seed": "" if seed is None else seed,
                        "station_code": code,
                        "source_fold": source_fold,
                    })
    return plan, pd.DataFrame(rows)


def subset_train_df(df, fold, sample):
    codes = sample["codes"]
    return df[(df["fold"] != fold) & df["station_code"].isin(codes)].reset_index(drop=True)


def row_counts(counts):
    return json.dumps(counts, sort_keys=True)


def write_run_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def completed_run(path, expected):
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved.get("run_key") != expected:
        raise SystemExit(f"Refusing to overwrite incompatible existing run: {path}")
    history = path.with_suffix(".history.csv")
    if not history.exists() or "val_loss" not in pd.read_csv(history, nrows=0).columns:
        print(f"RERUN {path}: missing new-schema val_loss history")
        return False
    return True


def model_setup(model_name):
    experiment_name, wide = MODELS[model_name]
    build_model, model_config = selected_model(experiment_name, wide=wide)
    cfg = training_config(model_config, epochs=CV_EPOCHS, folds=None)
    return build_model, cfg


def run_one(model_name, fraction, seed, fold, df, streams, sampling_plan):
    sample = sampling_plan[fold]["samples"][(fraction, seed)]
    result_file = run_path(model_name, fraction, seed, fold)
    expected = {
        "model": model_name,
        "validation_fold": fold,
        "fraction": fraction,
        "seed": seed,
        "station_codes": sorted(sample["codes"]),
    }
    if completed_run(result_file, expected):
        print(f"SKIP {model_name} {fold} {fraction_tag(fraction)} seed={seed}: already complete")
        return
    for stale in (result_file, prediction_path(model_name, fraction, seed, fold), history_path(model_name, fraction, seed, fold)):
        if stale.exists():
            stale.unlink()
    build_model, cfg = model_setup(model_name)
    cfg["ablation_model"] = model_name
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    train_df = subset_train_df(df, fold, sample)
    print(f"\n########## ABLATION {model_name} {fold} {fraction_tag(fraction)} seed={seed} "
          f"(train {len(train_df)}, val {len(val_df)}) ##########")
    res, arrs = training.train_one_fold(
        train_df, val_df, streams, cfg, build_model,
        fold=fold, history_path=history_path(model_name, fraction, seed, fold),
    )
    predictions = pd.concat(evaluation.prediction_table(val_df, arrs, cfg), ignore_index=True)
    pred_file = prediction_path(model_name, fraction, seed, fold)
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_file, index=False)
    meta = sampling_plan[fold]
    payload = {
        "run_key": expected,
        "model": model_name,
        "validation_fold": fold,
        "fraction": fraction,
        "seed": seed,
        "n_train_before_buffer": meta["n_train_before_buffer"],
        "n_buffer_dropped": meta["n_buffer_dropped"],
        "n_train_after_buffer": meta["n_train_after_buffer"],
        "n_train_sampled": len(sample["codes"]),
        "source_counts_after_buffer": meta["source_counts_after_buffer"],
        "sampled_counts_by_source_fold": sample["counts"],
        "result": res,
        "prediction_file": str(pred_file.relative_to(RESULTS_DIR)),
        "history_file": str(history_path(model_name, fraction, seed, fold).relative_to(RESULTS_DIR)),
        "config": {k: v for k, v in cfg.items() if k != "folds"},
    }
    write_run_result(result_file, payload)


def canonical_prediction_frame(model_name):
    path = result_paths(model_name)["cv_predictions"]
    if not path.exists():
        raise SystemExit(f"Missing canonical predictions for {model_name}: {path}")
    return pd.read_csv(path)


def canonical_runs(model_name):
    result = result_paths(model_name)
    if not result["cv_folds"].exists():
        raise SystemExit(f"Missing canonical fold results for {model_name}: {result['cv_folds']}")
    folds_df = pd.read_csv(result["cv_folds"])
    rows = []
    for _, r in folds_df.iterrows():
        rows.append({
            "model": model_name,
            "validation_fold": r["fold"],
            "fraction": 1.0,
            "seed": "",
            "pollutant": r["pollutant"],
            "n_train_before_buffer": r.get("n_train_before_buffer"),
            "n_buffer_dropped": r.get("n_buffer_dropped"),
            "n_train_after_buffer": r.get("n_train"),
            "n_train_sampled": r.get("n_train"),
            "sampled_counts_by_source_fold": "",
            "best_epoch": r["best_epoch"],
            "rmse": r["rmse"],
            "mae": r["mae"],
            "r2": r["r2"],
            "n_validation": r["n"],
            "total_parameters": r["total_parameters"],
            "trainable_parameters": r["trainable_parameters"],
            "frozen_parameters": r["frozen_parameters"],
            "source": "canonical_cv",
        })
    return rows


def reduced_runs(model_name):
    rows = []
    for path in sorted((RUNS_DIR / model_name).glob("frac_*/*/*.json")):
        payload = json.loads(path.read_text())
        result = payload["result"]
        counts = result["parameter_counts"]
        for pollutant, vals in ((p, result[p]) for p in result["config"]["pollutants"]):
            rows.append({
                "model": model_name,
                "validation_fold": payload["validation_fold"],
                "fraction": payload["fraction"],
                "seed": payload["seed"],
                "pollutant": pollutant,
                "n_train_before_buffer": payload["n_train_before_buffer"],
                "n_buffer_dropped": payload["n_buffer_dropped"],
                "n_train_after_buffer": payload["n_train_after_buffer"],
                "n_train_sampled": payload["n_train_sampled"],
                "sampled_counts_by_source_fold": row_counts(payload["sampled_counts_by_source_fold"]),
                "best_epoch": result["best_epoch"],
                "rmse": vals["rmse"],
                "mae": vals["mae"],
                "r2": vals["r2"],
                "n_validation": vals["n"],
                "total_parameters": counts["total"],
                "trainable_parameters": counts["trainable"],
                "frozen_parameters": counts["frozen"],
                "source": "ablation_run",
            })
    return rows


def prediction_frames(model_name):
    frames = []
    full = canonical_prediction_frame(model_name)
    full["model"] = model_name
    full["fraction"] = 1.0
    full["seed"] = ""
    frames.append(full)
    for path in sorted((RUNS_DIR / model_name).glob("frac_*/*/*.predictions.csv")):
        run_file = path.with_name(path.name.removesuffix(".predictions.csv") + ".json")
        run = json.loads(run_file.read_text())
        frame = pd.read_csv(path)
        frame["model"] = model_name
        frame["fraction"] = run["fraction"]
        frame["seed"] = run["seed"]
        frames.append(frame)
    return frames


def summarize():
    run_rows = []
    pred_frames = []
    for model_name in MODELS:
        run_rows.extend(canonical_runs(model_name))
        run_rows.extend(reduced_runs(model_name))
        pred_frames.extend(prediction_frames(model_name))
    runs = pd.DataFrame(run_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_DIR / "data_size_runs.csv", index=False)
    predictions = pd.concat(pred_frames, ignore_index=True)
    summary_rows = []
    for (model_name, fraction, seed), sub in predictions.groupby(["model", "fraction", "seed"], dropna=False):
        fold_metrics = runs[
            (runs["model"] == model_name)
            & (runs["fraction"] == fraction)
            & (runs["seed"].fillna("").astype(str) == str(seed))
        ]
        metrics = evaluation.metric_values(sub["pred"].to_numpy(), sub["true"].to_numpy())
        summary_rows.append({
            "model": model_name,
            "fraction": fraction,
            "seed": seed,
            "n_folds": int(sub["fold"].nunique()),
            "n_oof": metrics["n"],
            "pooled_oof_rmse": metrics["rmse"],
            "pooled_oof_mae": metrics["mae"],
            "pooled_oof_r2": metrics["r2"],
            "mean_fold_rmse": float(fold_metrics["rmse"].mean()) if len(fold_metrics) else np.nan,
            "mean_fold_mae": float(fold_metrics["mae"].mean()) if len(fold_metrics) else np.nan,
            "mean_fold_r2": float(fold_metrics["r2"].mean()) if len(fold_metrics) else np.nan,
            "median_best_epoch": float(fold_metrics["best_epoch"].median()) if len(fold_metrics) else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values(["model", "fraction", "seed"])
    aggregates = []
    for (model_name, fraction), sub in summary.groupby(["model", "fraction"]):
        aggregates.append({
            "model": model_name,
            "fraction": fraction,
            "n_seeds": int(len(sub)),
            "pooled_oof_rmse_mean": float(sub["pooled_oof_rmse"].mean()),
            "pooled_oof_rmse_std": float(sub["pooled_oof_rmse"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "pooled_oof_mae_mean": float(sub["pooled_oof_mae"].mean()),
            "pooled_oof_mae_std": float(sub["pooled_oof_mae"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "pooled_oof_r2_mean": float(sub["pooled_oof_r2"].mean()),
            "pooled_oof_r2_std": float(sub["pooled_oof_r2"].std(ddof=1)) if len(sub) > 1 else 0.0,
        })
    summary.to_csv(RESULTS_DIR / "data_size_summary.csv", index=False)
    pd.DataFrame(aggregates).to_csv(RESULTS_DIR / "data_size_summary_by_fraction.csv", index=False)
    make_gap_summary(summary)


def make_gap_summary(summary):
    rows = []
    cnn = summary[summary["model"] == "cnn_deep_wide"]
    resnet = summary[summary["model"] == "resnet_frozen"]
    for _, c in cnn.iterrows():
        match = resnet[(resnet["fraction"] == c["fraction"]) & (resnet["seed"].astype(str) == str(c["seed"]))]
        if len(match) != 1:
            continue
        r = match.iloc[0]
        rows.append({
            "fraction": c["fraction"],
            "seed": c["seed"],
            "cnn_rmse": c["pooled_oof_rmse"],
            "resnet_rmse": r["pooled_oof_rmse"],
            "rmse_gap_resnet_minus_cnn": r["pooled_oof_rmse"] - c["pooled_oof_rmse"],
            "cnn_mae": c["pooled_oof_mae"],
            "resnet_mae": r["pooled_oof_mae"],
            "mae_gap_resnet_minus_cnn": r["pooled_oof_mae"] - c["pooled_oof_mae"],
        })
    gap = pd.DataFrame(rows)
    if len(gap):
        means = []
        for fraction, sub in gap.groupby("fraction"):
            means.append({
                "fraction": fraction,
                "seed": "mean",
                "cnn_rmse": sub["cnn_rmse"].mean(),
                "resnet_rmse": sub["resnet_rmse"].mean(),
                "rmse_gap_resnet_minus_cnn": sub["rmse_gap_resnet_minus_cnn"].mean(),
                "cnn_mae": sub["cnn_mae"].mean(),
                "resnet_mae": sub["resnet_mae"].mean(),
                "mae_gap_resnet_minus_cnn": sub["mae_gap_resnet_minus_cnn"].mean(),
            })
        gap = pd.concat([gap, pd.DataFrame(means)], ignore_index=True)
    gap.to_csv(RESULTS_DIR / "model_gap_summary.csv", index=False)


def write_metadata(selected_folds):
    _, cfg = model_setup("cnn_deep_wide")
    payload = experiment.run_metadata(
        {
            **cfg,
            "experiment": "data_size_ablation",
            "models": list(MODELS),
            "fractions": FRACTIONS,
            "subsampling_seeds": SUBSAMPLING_SEEDS,
            "folds": selected_folds,
            "seed": SEED,
        },
        started_at=experiment.now_utc(),
    )
    experiment.write_json(RESULTS_DIR / "run_metadata.json", payload)


def parse_args():
    ap = argparse.ArgumentParser(description="Run the development-CV data-size ablation.")
    ap.add_argument("--model", choices=list(MODELS), help="run only one model")
    ap.add_argument("--fraction", type=float, choices=FRACTIONS, help="run only one reduced-data fraction")
    ap.add_argument("--seed", type=int, choices=SUBSAMPLING_SEEDS, help="run only one subsampling seed")
    ap.add_argument("--fold", choices=folds.FOLD_ORDER, help="run only one validation fold")
    ap.add_argument("--summarize-only", action="store_true", help="only rebuild summary CSVs")
    ap.add_argument("--self-check", action="store_true", help="run lightweight sampling checks and exit")
    return ap.parse_args()


def self_check():
    df = pd.DataFrame({
        "station_code": [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(8)] + ["v0", "near"],
        "fold": ["fold2_france"] * 10 + ["fold3_italy"] * 8 + ["fold1_iberia", "fold2_france"],
        "lat": [2.0] * 18 + [0.0, 0.1],
        "lon": [0.0] * 20,
        "pm25": [5.0] * 20,
    })
    plan, sampled = build_sampling_plan(df, ["fold1_iberia"])
    assert "near" not in plan["fold1_iberia"]["samples"][(0.5, 1)]["codes"]
    s25 = plan["fold1_iberia"]["samples"][(0.25, 1)]["codes"]
    s50 = plan["fold1_iberia"]["samples"][(0.50, 1)]["codes"]
    assert s25 <= s50
    counts = plan["fold1_iberia"]["samples"][(0.50, 1)]["counts"]
    assert counts["fold2_france"] == 5
    assert counts["fold3_italy"] == 4
    keys = ["validation_fold", "fraction", "seed", "station_code"]
    assert len(sampled) == len(sampled.drop_duplicates(keys))
    print("data-size ablation self-check passed")


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return
    runtime.apply_runtime_config()
    print(runtime.runtime_summary())
    if args.summarize_only:
        summarize()
        return
    models_to_run = [args.model] if args.model else list(MODELS)
    fractions_to_run = [args.fraction] if args.fraction else [f for f in FRACTIONS if f < 1.0]
    selected_folds = [args.fold] if args.fold else folds.development_fold_names()
    _, cfg = model_setup("cnn_deep_wide")
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]
    df = data.load_frame(streams, cfg)
    sampling_plan, sampled = build_sampling_plan(df, selected_folds, cfg["buffer_km"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(RESULTS_DIR / "sampled_station_ids.csv", index=False)
    write_metadata(selected_folds)
    for model_name in models_to_run:
        for fraction in fractions_to_run:
            if fraction == 1.0:
                continue
            seeds = [args.seed] if args.seed is not None else fraction_seeds(fraction)
            for seed in seeds:
                for fold in selected_folds:
                    run_one(model_name, fraction, seed, fold, df, streams, sampling_plan)
    summarize()


if __name__ == "__main__":
    main()
