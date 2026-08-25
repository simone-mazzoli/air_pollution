import argparse
import sys
from pathlib import Path

import pandas as pd

ABLATION_DIR = Path(__file__).resolve().parent
MODELS_DIR = ABLATION_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from shared import plotting


RESULTS_DIR = ABLATION_DIR / "results"


def run_info(history_path):
    run_path = history_path.with_name(history_path.name.removesuffix(".history.csv") + ".json")
    if not run_path.exists():
        return None
    import json

    return json.loads(run_path.read_text())


def plot_epoch_curves():
    out_dir = RESULTS_DIR / "figures" / "learning_curves"
    count = 0
    for history_path in sorted((RESULTS_DIR / "runs").glob("*/*/*/*.history.csv")):
        info = run_info(history_path)
        if info is None:
            print(f"SKIP {history_path}: missing run JSON")
            continue
        history = plotting.load_history(history_path)
        model = info["model"]
        fraction = int(round(float(info["fraction"]) * 100))
        seed = info["seed"]
        fold = info["validation_fold"]
        title = f"{model} frac {fraction}% seed {seed} - {plotting.FOLD_LABELS.get(fold, fold)}"
        prefix = f"{model}_frac{fraction:03d}_seed{seed}_{fold}"
        history = history.sort_values("epoch")
        plotting.save_objective_loss_curve(
            history,
            out_dir / f"{prefix}_objective_loss.png",
            f"{title}: objective loss",
        )
        plotting.save_performance_curve(
            history,
            out_dir / f"{prefix}_validation_performance.png",
            f"{title}: validation RMSE/MAE",
        )
        count += 1
    print(f"wrote {count} ablation epoch learning curves")


def plot_data_size_curve():
    summary_path = RESULTS_DIR / "data_size_summary_by_fraction.csv"
    gap_path = RESULTS_DIR / "model_gap_summary.csv"
    if not summary_path.exists():
        print(f"SKIP data-size curve: missing {summary_path}")
        return
    import matplotlib.pyplot as plt
    import seaborn as sns
    import report_plot_style

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    for metric, label in [("rmse", "RMSE"), ("mae", "MAE")]:
        fig, ax = plt.subplots(figsize=report_plot_style.FIGSIZE_SINGLE)
        for model, sub in summary.groupby("model"):
            sub = sub.sort_values("fraction")
            x = sub["fraction"] * 100
            y = sub[f"pooled_oof_{metric}_mean"]
            yerr = sub.get(f"pooled_oof_{metric}_std")
            sns.lineplot(
                x=x,
                y=y,
                marker="o",
                linewidth=1.8,
                color=report_plot_style.model_color(model),
                label=report_plot_style.model_label(model),
                ax=ax,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=report_plot_style.model_color(model),
                elinewidth=1.0,
                capsize=3,
                alpha=0.9,
            )
        ax.set_xlabel("training fraction [%]")
        ax.set_ylabel(f"pooled OOF {label} [µg/m³]")
        report_plot_style.clean_axis(ax)
        ax.legend(frameon=False, title=None)
        fig.tight_layout()
        plotting.savefig(fig, out_dir / f"data_size_{metric}_learning_curve.png")
        plt.close(fig)

    if gap_path.exists():
        gap = pd.read_csv(gap_path)
        gap = gap[gap["seed"].astype(str) == "mean"].sort_values("fraction")
        if len(gap):
            for metric, label in [("rmse", "RMSE"), ("mae", "MAE")]:
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                ax.axhline(0, color="black", linewidth=1.0)
                ax.plot(
                    gap["fraction"] * 100,
                    gap[f"{metric}_gap_resnet_minus_cnn"],
                    marker="o",
                    linewidth=1.8,
                )
                ax.set_xlabel("training fraction [%]")
                ax.set_ylabel(f"{label} gap [µg/m³]")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                plotting.savefig(fig, out_dir / f"data_size_{metric}_gap.png")
                plt.close(fig)
    print("wrote ablation data-size figures")


def parse_args():
    ap = argparse.ArgumentParser(description="Plot data-size ablation results.")
    ap.add_argument("--epoch-only", action="store_true", help="only plot epoch-wise curves for completed runs")
    ap.add_argument("--data-size-only", action="store_true", help="only plot RMSE-vs-training-fraction figures")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.data_size_only:
        plot_epoch_curves()
    if not args.epoch_only:
        plot_data_size_curve()


if __name__ == "__main__":
    main()
