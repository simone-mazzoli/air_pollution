import json

import numpy as np
import pandas as pd

from . import paths
from .config import result_paths
from .folds import FOLD_ORDER
from .models import (
    ALL_RESULT_EXPERIMENTS,
    DIAGNOSTIC_EXPERIMENTS,
    FINAL_MAIN_EXPERIMENTS,
    HISTORICAL_EXPERIMENTS,
    SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS,
    SUPPLEMENTARY_FULL_CV_EXPERIMENTS,
    SUMMARY_EXPERIMENTS,
)

IBERIA_FOLD = "fold1_iberia"


def result_dir_for(name):
    active = result_paths(name)["dir"]
    archived = paths.RESULTS / "archive" / "historical" / name
    return archived if name in HISTORICAL_EXPERIMENTS and archived.exists() else active


def result_files(name):
    result_dir = result_dir_for(name)
    return {
        "dir": result_dir,
        "cv_results": result_dir / "eea_cv_results.json",
        "cv_folds": result_dir / "cv_folds.csv",
    }


def role(name):
    if name in FINAL_MAIN_EXPERIMENTS:
        return "main"
    if name in SUPPLEMENTARY_FULL_CV_EXPERIMENTS:
        return "supplementary_full_cv"
    if name in SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS:
        return "supplementary_iberia_diagnostic"
    if name in HISTORICAL_EXPERIMENTS:
        return "historical"
    return "other"


def cv_status(name, folds_df):
    present = set(folds_df["fold"].dropna().unique()) if "fold" in folds_df else set()
    complete = present == set(FOLD_ORDER)
    if name in FINAL_MAIN_EXPERIMENTS and complete:
        return "main_complete_cv"
    if name in HISTORICAL_EXPERIMENTS:
        return "historical_complete_cv" if complete else "historical"
    if name in DIAGNOSTIC_EXPERIMENTS:
        return "development_complete_cv" if complete else "partial_diagnostic"
    return "complete_cv" if complete else f"partial_{len(present)}_of_8"


def summary_rows(experiments, *, require_complete=True):
    rows, fold_frames, missing = [], [], []
    for name in experiments:
        result = result_files(name)
        if not (result["cv_results"].exists() and result["cv_folds"].exists()):
            missing.append(name)
            continue
        cv = json.loads(result["cv_results"].read_text())
        folds = pd.read_csv(result["cv_folds"])
        status = cv_status(name, folds)
        if require_complete and status != "main_complete_cv":
            missing.append(f"{name} ({status})")
            continue
        fold_frames.append(folds.assign(role=role(name), status=status))
        pooled = cv.get("pooled_out_of_fold", {})
        for pollutant in sorted(folds["pollutant"].unique()):
            sub = folds[folds["pollutant"] == pollutant]
            pooled_metrics = pooled.get(pollutant, {})
            rows.append({
                "experiment": name,
                "role": role(name),
                "status": status,
                "pollutant": pollutant,
                "pooled_cv_rmse": pooled_metrics.get("rmse"),
                "pooled_cv_mae": pooled_metrics.get("mae"),
                "pooled_cv_r2": pooled_metrics.get("r2"),
                "mean_fold_rmse": float(sub["rmse"].mean()),
                "mean_fold_mae": float(sub["mae"].mean()),
                "mean_fold_r2": float(sub["r2"].mean()),
                "n_oof": pooled_metrics.get("n", int(sub["n"].sum())),
                "median_best_epoch": float(np.median(sub["best_epoch"])),
                "total_parameters": int(sub["total_parameters"].iloc[0]),
                "trainable_parameters": int(sub["trainable_parameters"].iloc[0]),
            })
    comparison = pd.DataFrame(rows)
    fold_comparison = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    return missing, comparison, fold_comparison


def summarize_results(experiments=SUMMARY_EXPERIMENTS, out_dir=None, *, require_complete=True):
    out_dir = paths.RESULTS / "summary" if out_dir is None else out_dir
    missing, comparison, fold_comparison = summary_rows(experiments, require_complete=require_complete)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(comparison):
        comparison.to_csv(out_dir / "main_model_comparison.csv", index=False)
    if len(fold_comparison):
        fold_comparison.to_csv(out_dir / "main_fold_comparison.csv", index=False)
    available = sorted(set(comparison["experiment"])) if len(comparison) else []
    return {"available": available, "missing": missing,
            "comparison": comparison, "fold_comparison": fold_comparison}


def summarize_all_existing(out_dir=None):
    out_dir = paths.RESULTS / "summary" / "all_existing" if out_dir is None else out_dir
    missing, comparison, fold_comparison = summary_rows(ALL_RESULT_EXPERIMENTS, require_complete=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(comparison):
        comparison.to_csv(out_dir / "all_experiments_classified.csv", index=False)
    if len(fold_comparison):
        fold_comparison.to_csv(out_dir / "all_fold_results_classified.csv", index=False)
    return {"available": sorted(set(comparison["experiment"])) if len(comparison) else [],
            "missing": missing, "comparison": comparison, "fold_comparison": fold_comparison}


def summarize_report_tables(out_dir=None):
    out_dir = paths.RESULTS / "summary" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    main = summarize_results(out_dir=out_dir)

    _, supplementary, _ = summary_rows(
        (*SUPPLEMENTARY_FULL_CV_EXPERIMENTS, *FINAL_MAIN_EXPERIMENTS),
        require_complete=False,
    )
    if len(supplementary):
        supplementary = supplementary[
            supplementary["status"].isin(["main_complete_cv", "development_complete_cv"])
        ]
        supplementary.to_csv(out_dir / "supplementary_full_cv.csv", index=False)

    diagnostic_rows = []
    diagnostic_experiments = (
        "cnn", "cnn_deep", "cnn_large", "cnn_deep_wide", "resnet_frozen", "resnet_full"
    )
    for name in diagnostic_experiments:
        result = result_files(name)
        if not result["cv_folds"].exists():
            continue
        folds = pd.read_csv(result["cv_folds"])
        sub = folds[folds["fold"] == IBERIA_FOLD]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            diagnostic_rows.append({
                "experiment": name,
                "role": role(name),
                "validation_fold": IBERIA_FOLD,
                "pollutant": row["pollutant"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "best_epoch": row["best_epoch"],
                "n_validation": row["n"],
                "total_parameters": row["total_parameters"],
                "trainable_parameters": row["trainable_parameters"],
                "frozen_parameters": row["frozen_parameters"],
            })
    diagnostics = pd.DataFrame(diagnostic_rows)
    if len(diagnostics):
        diagnostics.to_csv(out_dir / "iberia_diagnostics.csv", index=False)
    return {**main, "supplementary_full_cv": supplementary, "iberia_diagnostics": diagnostics}
