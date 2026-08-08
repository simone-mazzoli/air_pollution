import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch

from . import folds
from .config import BUFFER_KM, DEVICE, SEED


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def reset_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def append_csv_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row)
    if exists:
        with path.open(newline="") as f:
            fieldnames = next(csv.reader(f))
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def optimizer_lrs(opt, cfg):
    out = {}
    groups = opt.param_groups
    for i, group in enumerate(groups):
        out[f"lr_group_{i}"] = group["lr"]
    if groups:
        out["lr" if cfg.get("experiment") == "cnn" else "lr_head"] = groups[0]["lr"]
    if cfg.get("experiment") == "resnet_full" and len(groups) > 1:
        out["lr_backbone"] = groups[1]["lr"]
    return out


def epoch_history_row(cfg, fold, epoch, train_loss, train_metrics, val_metrics,
                      opt, timings, best_so_far, epochs_since_improvement):
    row = {
        "experiment": cfg["experiment"],
        "fold": fold,
        "epoch": epoch,
        "train_loss": train_loss,
        "best_so_far": best_so_far,
        "epochs_since_improvement": epochs_since_improvement,
        **optimizer_lrs(opt, cfg),
        **timings,
    }
    for p in cfg["pollutants"]:
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            vals = metrics[p]
            row[f"{prefix}_{p}_rmse"] = vals["rmse"]
            row[f"{prefix}_{p}_mae"] = vals["mae"]
            row[f"{prefix}_{p}_r2"] = vals["r2"]
            row[f"{prefix}_{p}_n"] = vals["n"]
    return row


def fold_summary_rows(cfg, fold, result):
    counts = result["parameter_counts"]
    rows = []
    for p in cfg["pollutants"]:
        vals = result[p]
        rows.append({
            "experiment": cfg["experiment"],
            "fold": fold,
            "pollutant": p,
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_buffer_dropped": result.get("buffer_removed_train_stations"),
            "n_train_before_buffer": result.get("n_train_before_buffer"),
            "best_epoch": result["best_epoch"],
            "rmse": vals["rmse"],
            "mae": vals["mae"],
            "r2": vals["r2"],
            "n": vals["n"],
            "baseline_rmse": vals["baseline"],
            "total_parameters": counts["total"],
            "trainable_parameters": counts["trainable"],
            "frozen_parameters": counts["frozen"],
        })
    return rows


def git_value(args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_metadata(cfg, model=None, parameter_counts=None, model_metadata=None, *,
                 started_at=None, completed_at=None):
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = None
    if model is not None and hasattr(model, "parameter_metadata"):
        model_metadata = model.parameter_metadata()
    counts = dict(parameter_counts) if parameter_counts else None
    if counts and model_metadata:
        for key, value in model_metadata.items():
            if key.endswith("_parameters"):
                counts[key] = value
    metadata = {
        "experiment": cfg.get("experiment"),
        "model": cfg.get("model"),
        "backbone_mode": cfg.get("backbone_mode"),
        "high_encoder": cfg.get("high_encoder"),
        "pollutants": cfg.get("pollutants"),
        "seed": SEED,
        "config": dict(cfg),
        "fold_setup": folds.FOLDS,
        "test_laender": sorted(folds.DE_TEST_LAENDER),
        "buffer_km": cfg.get("buffer_km", BUFFER_KM),
        "parameter_counts": counts,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": git_value(["status", "--porcelain"]) not in (None, ""),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": DEVICE,
        "gpu": gpu,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    if model_metadata is not None:
        metadata["model_metadata"] = model_metadata
    return metadata
