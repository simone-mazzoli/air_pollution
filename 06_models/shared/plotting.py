import os
from pathlib import Path

import pandas as pd

_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_FOLDS = [
    "fold1_iberia",
    "fold2_france",
    "fold3_italy",
    "fold4_alpine",
    "fold5_north",
    "fold6_balkan_e",
    "fold7_balkan_s",
    "fold8_poland",
]

FOLD_LABELS = {
    "fold1_iberia": "Iberia",
    "fold2_france": "France",
    "fold3_italy": "Italy",
    "fold4_alpine": "Alpine",
    "fold5_north": "North",
    "fold6_balkan_e": "Balkan E",
    "fold7_balkan_s": "Balkan S",
    "fold8_poland": "Poland",
}


def load_history(path):
    df = pd.read_csv(path)
    required = {"fold", "epoch", "train_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    rmse_cols = [c for c in df.columns if c.startswith("val_") and c.endswith("_rmse")]
    if not rmse_cols:
        raise ValueError(f"{path} has no validation RMSE column")
    return df


def validation_rmse_column(history):
    cols = [c for c in history.columns if c.startswith("val_") and c.endswith("_rmse")]
    return "val_pm25_rmse" if "val_pm25_rmse" in cols else cols[0]


def validation_mae_column(history):
    cols = [c for c in history.columns if c.startswith("val_") and c.endswith("_mae")]
    return "val_pm25_mae" if "val_pm25_mae" in cols else (cols[0] if cols else None)


def best_epoch_from_history(history):
    if "best_so_far" not in history.columns:
        return None
    marked = history[history["best_so_far"].astype(str).str.lower().isin(["true", "1"])]
    return None if marked.empty else int(marked["epoch"].max())


def best_epochs_from_fold_results(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"fold", "best_epoch"}.issubset(df.columns):
        return {}
    return {row["fold"]: int(row["best_epoch"]) for _, row in df.drop_duplicates("fold").iterrows()}


def cv_status(history):
    have = set(history["fold"].dropna().unique())
    return "complete 8-fold CV" if have == set(EXPECTED_FOLDS) else f"partial/diagnostic ({len(have)}/8 folds)"


def _mark_best_epoch(ax, history, best_epoch, y_col=None, label=True):
    if best_epoch is None:
        return
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
    if y_col and best_epoch in set(history["epoch"]):
        ax.scatter(
            [best_epoch],
            [history.loc[history["epoch"] == best_epoch, y_col].iloc[-1]],
            color="black",
            s=24,
            zorder=3,
            label=f"RMSE-selected best epoch {best_epoch}" if label else None,
        )


def save_objective_loss_curve(history, out_path, title, best_epoch=None):
    best_epoch = best_epoch if best_epoch is not None else best_epoch_from_history(history)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(history["epoch"], history["train_loss"], color="#1f77b4", linewidth=1.8, label="train loss")
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], color="#9467bd", linewidth=1.8, label="validation loss")
    _mark_best_epoch(ax, history, best_epoch)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("SmoothL1 objective loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_performance_curve(history, out_path, title, best_epoch=None):
    rmse_col = validation_rmse_column(history)
    mae_col = validation_mae_column(history)
    best_epoch = best_epoch if best_epoch is not None else best_epoch_from_history(history)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(history["epoch"], history[rmse_col], color="#d62728", linewidth=1.8, label="validation RMSE")
    if mae_col:
        ax.plot(history["epoch"], history[mae_col], color="#ff7f0e", linewidth=1.4, alpha=0.9, label="validation MAE")
    _mark_best_epoch(ax, history, best_epoch, rmse_col)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation error [ug/m3]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_run_curve(history, out_path, title, best_epoch=None):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    best_epoch = best_epoch if best_epoch is not None else best_epoch_from_history(history)
    rmse_col = validation_rmse_column(history)
    mae_col = validation_mae_column(history)
    axes[0].plot(history["epoch"], history["train_loss"], color="#1f77b4", linewidth=1.8, label="train loss")
    if "val_loss" in history.columns:
        axes[0].plot(history["epoch"], history["val_loss"], color="#9467bd", linewidth=1.8, label="validation loss")
    _mark_best_epoch(axes[0], history, best_epoch, label=False)
    axes[0].set_ylabel("SmoothL1 objective loss")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history[rmse_col], color="#d62728", linewidth=1.8, label="validation RMSE")
    if mae_col:
        axes[1].plot(history["epoch"], history[mae_col], color="#ff7f0e", linewidth=1.2, alpha=0.8, label="validation MAE")
    _mark_best_epoch(axes[1], history, best_epoch, rmse_col)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation error [ug/m3]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_summary_grid(history, out_path, experiment_name, status, best_epochs=None):
    rmse_col = validation_rmse_column(history)
    best_epochs = best_epochs or {}
    folds = [f for f in EXPECTED_FOLDS if f in set(history["fold"])]
    folds += sorted(set(history["fold"]) - set(folds))
    if not folds:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncols, nrows = 4, (len(folds) + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.0 * nrows), squeeze=False)
    for ax, fold in zip(axes.ravel(), folds):
        sub = history[history["fold"] == fold]
        ax.plot(sub["epoch"], sub[rmse_col], color="#d62728", linewidth=1.5)
        best_epoch = best_epochs.get(fold, best_epoch_from_history(sub))
        if best_epoch is not None:
            ax.axvline(best_epoch, color="black", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_title(FOLD_LABELS.get(fold, fold))
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE [ug/m3]")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(folds):]:
        ax.axis("off")
    fig.suptitle(f"{experiment_name}: validation RMSE by fold ({status})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
