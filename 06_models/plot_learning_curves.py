import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from shared import plotting
from shared import paths
from shared.models import FINAL_MAIN_EXPERIMENTS


DEFAULT_EXPERIMENTS = list(FINAL_MAIN_EXPERIMENTS)


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
    best_epochs = plotting.best_epochs_from_fold_results(result_dir / "cv_folds.csv")
    out_dir = result_dir / "figures" / "learning_curves"
    for fold in sorted(history["fold"].unique(), key=lambda f: plotting.EXPECTED_FOLDS.index(f) if f in plotting.EXPECTED_FOLDS else 99):
        sub = history[history["fold"] == fold].sort_values("epoch")
        label = plotting.FOLD_LABELS.get(fold, fold)
        suffix = "" if "val_loss" in sub.columns else " (training loss only; val_loss not logged)"
        plotting.save_objective_loss_curve(
            sub,
            out_dir / f"{fold}_objective_loss.png",
            f"{name} - {label}: objective loss{suffix}",
            best_epochs.get(fold),
        )
        plotting.save_performance_curve(
            sub,
            out_dir / f"{fold}_validation_performance.png",
            f"{name} - {label}: validation RMSE/MAE ({status})",
            best_epochs.get(fold),
        )
    plotting.save_summary_grid(
        history,
        result_dir / "figures" / "learning_curves_summary.png",
        name,
        status,
        best_epochs,
    )
    print(f"wrote {name} learning curves ({status})")
    return True


def final_training_note():
    checkpoint = paths.RESULTS / "cnn_deep_wide" / "final_model.pt"
    history = paths.RESULTS / "cnn_deep_wide" / "final_history.csv"
    if checkpoint.exists() and not history.exists():
        print("NOTE final cnn_deep_wide: final_model.pt exists, but no per-epoch final-training history is saved; no validation curve is invented.")


def parse_args():
    ap = argparse.ArgumentParser(description="Plot epoch-wise learning curves from saved CV histories.")
    ap.add_argument("--experiment", help="plot one experiment result directory")
    ap.add_argument("--all-existing", action="store_true",
                    help="also plot historical and partial diagnostic result folders")
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
