import json

import numpy as np
import pandas as pd

from . import paths
from .config import result_paths
from .models import SUMMARY_EXPERIMENTS


def summarize_results(experiments=SUMMARY_EXPERIMENTS, out_dir=None):
    out_dir = paths.RESULTS / "summary" if out_dir is None else out_dir
    rows, fold_frames, missing = [], [], []
    for name in experiments:
        result = result_paths(name)
        if not (result["cv_results"].exists() and result["cv_folds"].exists()):
            missing.append(name)
            continue
        cv = json.loads(result["cv_results"].read_text())
        folds = pd.read_csv(result["cv_folds"])
        fold_frames.append(folds)
        pooled = cv.get("pooled_out_of_fold", {})
        for pollutant in sorted(folds["pollutant"].unique()):
            sub = folds[folds["pollutant"] == pollutant]
            pooled_metrics = pooled.get(pollutant, {})
            rows.append({
                "experiment": name,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(comparison):
        comparison.to_csv(out_dir / "experiment_comparison.csv", index=False)
    if len(fold_comparison):
        fold_comparison.to_csv(out_dir / "fold_comparison.csv", index=False)
    available = sorted({r["experiment"] for r in rows})
    return {"available": available, "missing": missing,
            "comparison": comparison, "fold_comparison": fold_comparison}
