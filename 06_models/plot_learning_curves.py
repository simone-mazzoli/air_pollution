import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from shared import plotting
from shared import paths
from shared.models import REPORT_EXPERIMENTS, SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS


DEFAULT_EXPERIMENTS = list(REPORT_EXPERIMENTS)


def experiment_dirs(selected, all_existing=False):
    if selected:
        return [paths.RESULTS / selected]
    if all_existing:
        return sorted(p.parent for p in paths.RESULTS.glob("*/cv_history.csv"))
    return [paths.RESULTS / name for name in DEFAULT_EXPERIMENTS]


def plot_experiment(result_dir):
    name = result_dir.name
    history_path = result_dir / "cv_history.csv"

    if not history_path.exists():
        print(f"SKIP {name}: missing {history_path}")
        return False

    try:
        history = plotting.load_history(history_path)
    except ValueError as exc:
        print(f"SKIP {name}: {exc}")
        return False

    status = plotting.cv_status(history)
    best_epochs = plotting.best_epochs_from_fold_results(
        result_dir / "cv_folds.csv"
    )
    out_dir = result_dir / "figures" / "learning_curves"

    # One objective-loss scale for all folds of this model.
    loss_columns = ["train_loss"]
    if "val_loss" in history.columns:
        loss_columns.append("val_loss")
    loss_ylim = plotting.common_ylim(history, loss_columns)

    # One RMSE/MAE scale for all folds of this model.
    rmse_col = plotting.validation_rmse_column(history)
    mae_col = plotting.validation_mae_column(history)
    performance_columns = [rmse_col]
    if mae_col:
        performance_columns.append(mae_col)
    performance_ylim = plotting.common_ylim(history, performance_columns)

    fold_order = plotting.ordered_folds(history)

    for fold in fold_order:
        sub = history[history["fold"] == fold].sort_values("epoch")
        label = plotting.FOLD_LABELS.get(fold, fold)

        scope = (
            "supplementary diagnostic - fold1_iberia"
            if name in SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS
            else status
        )

        suffix = (
            ""
            if "val_loss" in sub.columns
            else " (training loss only; val_loss not logged)"
        )

        plotting.save_objective_loss_curve(
            sub,
            out_dir / f"{fold}_objective_loss.png",
            f"{name} - {label}: objective loss{suffix}",
            best_epochs.get(fold),
            ylim=loss_ylim,
        )

        plotting.save_performance_curve(
            sub,
            out_dir / f"{fold}_validation_performance.png",
            f"{name} - {label}: validation RMSE/MAE ({scope})",
            best_epochs.get(fold),
            ylim=performance_ylim,
        )

    objective_summary = (
        result_dir / "figures" / "learning_curves_summary_objective_loss.png"
    )
    performance_summary = (
        result_dir
        / "figures"
        / "learning_curves_summary_validation_performance.png"
    )
    legacy_summary = result_dir / "figures" / "learning_curves_summary.png"

    if status == "complete 8-fold CV":
        plotting.save_objective_summary_grid(
            history,
            objective_summary,
            name,
            status,
            best_epochs,
            ylim=loss_ylim,
        )
        plotting.save_performance_summary_grid(
            history,
            performance_summary,
            name,
            status,
            best_epochs,
            ylim=performance_ylim,
        )
        if legacy_summary.exists():
            legacy_summary.unlink()
    else:
        for stale in [objective_summary, performance_summary, legacy_summary]:
            if stale.exists():
                stale.unlink()

    print(
        f"wrote {name} learning curves ({status}); "
        "shared y-scales applied across folds and both summary grids exported"
    )
    return True


def final_training_note():
    checkpoint = paths.RESULTS / "cnn_deep_wide" / "final_model.pt"
    history = paths.RESULTS / "cnn_deep_wide" / "final_history.csv"

    if checkpoint.exists() and not history.exists():
        print(
            "NOTE final cnn_deep_wide: final_model.pt exists, but no "
            "per-epoch final-training history is saved; no validation curve "
            "is invented."
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot epoch-wise learning curves from saved CV histories."
    )
    ap.add_argument(
        "--experiment",
        help="plot one experiment result directory",
    )
    ap.add_argument(
        "--all-existing",
        action="store_true",
        help="also plot historical and partial diagnostic result folders",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    dirs = experiment_dirs(args.experiment, args.all_existing)

    any_plotted = False
    for result_dir in dirs:
        any_plotted = plot_experiment(result_dir) or any_plotted

    final_training_note()

    if not any_plotted:
        raise SystemExit("No valid cv_history.csv files found.")


if __name__ == "__main__":
    main()
