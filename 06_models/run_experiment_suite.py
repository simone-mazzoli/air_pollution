import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
REPO = MODELS_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from shared import paths
from shared.config import CV_EPOCHS, CV_PATIENCE, BUFFER_KM, SEED, result_paths
from shared.folds import FOLD_ORDER
from shared.models import (
    FINAL_MAIN_EXPERIMENTS,
    REPORT_EXPERIMENTS,
    SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS,
    SUPPLEMENTARY_FULL_CV_EXPERIMENTS,
)


ARCHIVE_DIR = paths.RESULTS / "archive" / "pre_val_loss_rerun"
CV_ARTIFACTS = ["cv_history.csv", "cv_folds.csv", "eea_cv_predictions.csv", "eea_cv_results.json", "run_metadata.json"]
REQUIRED_DATA_PATHS = [
    paths.LABELS,
    paths.STATION_FOLD,
    paths.HIGH,
    paths.LOW,
    paths.SAT / "no2_tropomi",
    paths.SAT / "co_tropomi",
    paths.AERW,
    paths.DEMD,
]


def run(cmd, dry_run=False):
    text = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"DRY-RUN {text}")
        return
    print(text)
    subprocess.run(cmd, cwd=REPO, check=True)


def validate_setup(dry_run=False):
    print("final experiments:", ", ".join(FINAL_MAIN_EXPERIMENTS))
    print("supplementary full CV:", ", ".join(SUPPLEMENTARY_FULL_CV_EXPERIMENTS))
    print("supplementary Iberia diagnostics:", ", ".join(SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS))
    print(f"config: epochs={CV_EPOCHS} patience={CV_PATIENCE} buffer_km={BUFFER_KM} seed={SEED}")
    if dry_run:
        print("report suite plan:")
        print("  cnn_deep_wide  8 folds")
        print("  resnet_frozen  8 folds")
        print("  cnn            8 folds")
        print("  cnn_deep       fold1_iberia only")
        print("  cnn_large      fold1_iberia only")
        print("  resnet_full    fold1_iberia only")
        print("  data-size ablation: cnn_deep_wide vs resnet_frozen")
        print("  summaries and plots")
        print("excluded: resnet, resnet_layer4, final training, TEST prediction")
    if CV_PATIENCE != 25:
        raise SystemExit(f"ERROR: expected CV_PATIENCE=25, got {CV_PATIENCE}")
    if dry_run:
        print("DRY-RUN would check required data paths and shared logic")
        return
    missing = [p for p in REQUIRED_DATA_PATHS if not p.exists()]
    if missing:
        raise SystemExit("ERROR: missing required data paths:\n" + "\n".join(str(p) for p in missing))
    run([sys.executable, "06_models/check_shared_logic.py"])
    run([sys.executable, "06_models/data_size_ablation/run_ablation.py", "--self-check"])


def archive_previous_cv(dry_run=False):
    for exp in REPORT_EXPERIMENTS:
        result = result_paths(exp)
        dest = ARCHIVE_DIR / exp
        marker = dest / "ARCHIVED_ONCE.json"
        if marker.exists():
            print(f"archive exists for {exp}: {dest}")
            continue
        existing = [result["dir"] / name for name in CV_ARTIFACTS if (result["dir"] / name).exists()]
        if not existing:
            print(f"no previous CV artifacts to archive for {exp}")
            continue
        if dry_run:
            print(f"DRY-RUN would archive {exp} CV artifacts to {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for src in existing:
            shutil.copy2(src, dest / src.name)
        marker.write_text(json.dumps({"archived_artifacts": [p.name for p in existing]}, indent=2))
        print(f"archived previous {exp} CV artifacts to {dest}")


def run_main_cv(dry_run=False, force=False, archive=True):
    if archive:
        archive_previous_cv(dry_run=dry_run)
    for exp in FINAL_MAIN_EXPERIMENTS:
        cmd = [sys.executable, "06_models/01_train_cv.py", "--experiment", exp, "--resume"]
        if force:
            cmd.append("--force")
        run(cmd, dry_run=dry_run)


def run_supplementary(dry_run=False, force=False, archive=True):
    if archive:
        archive_previous_cv(dry_run=dry_run)
    for exp in SUPPLEMENTARY_FULL_CV_EXPERIMENTS:
        cmd = [sys.executable, "06_models/01_train_cv.py", "--experiment", exp, "--resume"]
        if force:
            cmd.append("--force")
        run(cmd, dry_run=dry_run)
    for exp in SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS:
        cmd = [
            sys.executable, "06_models/01_train_cv.py",
            "--experiment", exp,
            "--folds", "fold1_iberia",
            "--resume",
        ]
        if force:
            cmd.append("--force")
        run(cmd, dry_run=dry_run)


def run_ablation(dry_run=False):
    run([sys.executable, "06_models/data_size_ablation/run_ablation.py"], dry_run=dry_run)


def pooled_metrics(path):
    cv = json.loads(path.read_text())
    return cv.get("pooled_out_of_fold", {}).get("pm25", {})


def best_epochs(path):
    df = pd.read_csv(path)
    return dict(zip(df["fold"], df["best_epoch"]))


def write_provenance_comparison():
    rows = []
    for exp in REPORT_EXPERIMENTS:
        old_dir = ARCHIVE_DIR / exp
        new = result_paths(exp)
        if not ((old_dir / "eea_cv_results.json").exists() and new["cv_results"].exists()):
            continue
        old_metrics = pooled_metrics(old_dir / "eea_cv_results.json")
        new_metrics = pooled_metrics(new["cv_results"])
        old_epochs = best_epochs(old_dir / "cv_folds.csv") if (old_dir / "cv_folds.csv").exists() else {}
        new_epochs = best_epochs(new["cv_folds"]) if new["cv_folds"].exists() else {}
        rows.append({
            "experiment": exp,
            "old_pooled_rmse": old_metrics.get("rmse"),
            "new_pooled_rmse": new_metrics.get("rmse"),
            "old_pooled_mae": old_metrics.get("mae"),
            "new_pooled_mae": new_metrics.get("mae"),
            "old_pooled_r2": old_metrics.get("r2"),
            "new_pooled_r2": new_metrics.get("r2"),
            "old_fold_best_epochs": json.dumps(old_epochs, sort_keys=True),
            "new_fold_best_epochs": json.dumps(new_epochs, sort_keys=True),
        })
    if rows:
        out = paths.RESULTS / "summary" / "pre_val_loss_rerun_comparison.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"saved {out}")
    else:
        print("no archived/new CV pair found for provenance comparison")


def postprocess(dry_run=False):
    run([sys.executable, "06_models/summarize_cv_results.py"], dry_run=dry_run)
    run([sys.executable, "06_models/data_size_ablation/run_ablation.py", "--summarize-only"], dry_run=dry_run)
    run([sys.executable, "06_models/plot_learning_curves.py"], dry_run=dry_run)
    run([sys.executable, "06_models/data_size_ablation/plot_results.py"], dry_run=dry_run)
    if dry_run:
        print("DRY-RUN would write old-vs-new CV provenance comparison if archived runs exist")
    else:
        write_provenance_comparison()
    print("excluded from suite: resnet, resnet_layer4, final training, TEST prediction")


def parse_args():
    ap = argparse.ArgumentParser(description="Run the final development experiment suite.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="validate, run report CV/diagnostics, ablation, summaries, and plots")
    group.add_argument("--main-cv", action="store_true", help="run only final main CV experiments")
    group.add_argument("--supplementary", action="store_true", help="run only report supplementary CV/diagnostic experiments")
    group.add_argument("--ablation", action="store_true", help="run only the data-size ablation")
    group.add_argument("--postprocess", action="store_true", help="rebuild summaries and plots without training")
    ap.add_argument("--dry-run", action="store_true", help="print planned actions without training or writing")
    ap.add_argument("--force", action="store_true", help="force rerun of compatible main-CV folds")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.all:
        validate_setup(dry_run=args.dry_run)
        archive_previous_cv(dry_run=args.dry_run)
        run_main_cv(dry_run=args.dry_run, force=args.force, archive=False)
        run_supplementary(dry_run=args.dry_run, force=args.force, archive=False)
        run_ablation(dry_run=args.dry_run)
        postprocess(dry_run=args.dry_run)
    elif args.main_cv:
        validate_setup(dry_run=args.dry_run)
        run_main_cv(dry_run=args.dry_run, force=args.force)
    elif args.ablation:
        validate_setup(dry_run=args.dry_run)
        run_ablation(dry_run=args.dry_run)
    elif args.supplementary:
        validate_setup(dry_run=args.dry_run)
        run_supplementary(dry_run=args.dry_run, force=args.force)
    elif args.postprocess:
        postprocess(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
