#!/usr/bin/env python3
"""Post-hoc diagnostics for the frozen cnn_deep_wide sealed-TEST predictions."""

from __future__ import annotations

import math
import os
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "06_models" / "results" / "cnn_deep_wide" / "test_predictions.csv"
CHECKPOINT = ROOT / "06_models" / "results" / "cnn_deep_wide" / "final_model.pt"
OUT = ROOT / "07_prediction_analysis" / "outputs"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
POLLUTANT = "pm25"
UNITS = "µg/m³"


def first_existing(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def prediction_columns(df):
    cols = set(df.columns)
    return {
        "observed": first_existing(cols, [f"true_{POLLUTANT}", f"observed_{POLLUTANT}", POLLUTANT, "true", "observed"]),
        "predicted": first_existing(cols, [f"pred_{POLLUTANT}", f"predicted_{POLLUTANT}", "pred", "predicted"]),
        "lat": first_existing(cols, ["lat", "latitude"]),
        "lon": first_existing(cols, ["lon", "longitude"]),
        "station_id": first_existing(cols, ["station_code", "station_id", "location", "id"]),
        "state": first_existing(cols, ["land", "state", "region", "bundesland"]),
        "station_type": first_existing(cols, ["station_type", "type", "station_category", "site_type"]),
    }


def require_columns(found):
    missing = [name for name in ("observed", "predicted") if found[name] is None]
    if missing:
        raise SystemExit(f"ERROR: prediction file is missing required columns: {missing}")


def metric_values(pred, obs):
    err = pred - obs
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": int(len(obs)),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"),
    }


def load_training_mean_baseline():
    if not CHECKPOINT.exists():
        return None
    try:
        import torch

        bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"warning: could not read training-mean baseline from {CHECKPOINT}: {exc}")
        return None
    baseline = bundle.get("baseline_concentration", {})
    value = baseline.get(POLLUTANT)
    return float(value) if value is not None else None


def finite_analysis_frame(df, found):
    out = df.copy()
    out["observed"] = pd.to_numeric(out[found["observed"]], errors="coerce")
    out["predicted"] = pd.to_numeric(out[found["predicted"]], errors="coerce")
    out = out.dropna(subset=["observed", "predicted"]).copy()
    out["residual"] = out["predicted"] - out["observed"]
    out["absolute_error"] = out["residual"].abs()
    return out


def save_scatter(df, metrics):
    path = FIGURES / "test_observed_vs_predicted.png"
    lo = float(min(df["observed"].min(), df["predicted"].min()))
    hi = float(max(df["observed"].max(), df["predicted"].max()))
    pad = max((hi - lo) * 0.06, 0.25)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    ax.scatter(df["observed"], df["predicted"], s=34, alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed annual PM2.5 [{UNITS}]")
    ax.set_ylabel(f"Predicted annual PM2.5 [{UNITS}]")
    ax.text(
        0.04,
        0.96,
        f"n={metrics['n']}\nRMSE={metrics['rmse']:.2f}\nMAE={metrics['mae']:.2f}\nR²={metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_distributions(df):
    path = FIGURES / "test_prediction_distributions.png"
    values = pd.concat([df["observed"], df["predicted"]])
    bins = np.linspace(values.min(), values.max(), 18)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(df["observed"], bins=bins, alpha=0.55, label="Observed", color="#2f6f9f")
    ax.hist(df["predicted"], bins=bins, alpha=0.55, label="Predicted", color="#d9822b")
    ax.set_xlabel(f"Annual PM2.5 [{UNITS}]")
    ax.set_ylabel("Number of TEST stations")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_residual_distribution(df):
    path = FIGURES / "test_residual_distribution.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(df["residual"], bins=20, color="#6d5f9f", alpha=0.75)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel(f"Residual: prediction - observation [{UNITS}]")
    ax.set_ylabel("Number of TEST stations")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_station_map(df, value_col, title, path, *, cmap, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6.4, 6.8), dpi=160)
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df[value_col],
        s=38,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.22)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label(f"Station-level annual PM2.5 [{UNITS}]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_maps(df, found):
    if found["lat"] is None or found["lon"] is None:
        return []
    geo = df.copy()
    geo["lat"] = pd.to_numeric(geo[found["lat"]], errors="coerce")
    geo["lon"] = pd.to_numeric(geo[found["lon"]], errors="coerce")
    geo = geo.dropna(subset=["lat", "lon"])
    if geo.empty:
        return []

    conc_min = float(min(geo["observed"].min(), geo["predicted"].min()))
    conc_max = float(max(geo["observed"].max(), geo["predicted"].max()))
    residual_abs = float(np.nanmax(np.abs(geo["residual"])))
    ae_max = float(geo["absolute_error"].max())
    return [
        save_station_map(
            geo,
            "observed",
            "Observed TEST station annual PM2.5",
            FIGURES / "test_observed_map.png",
            cmap="viridis",
            vmin=conc_min,
            vmax=conc_max,
        ),
        save_station_map(
            geo,
            "predicted",
            "Predicted TEST station annual PM2.5",
            FIGURES / "test_predicted_map.png",
            cmap="viridis",
            vmin=conc_min,
            vmax=conc_max,
        ),
        save_station_map(
            geo,
            "residual",
            "TEST residual: prediction - observation",
            FIGURES / "test_residual_map.png",
            cmap="coolwarm",
            vmin=-residual_abs,
            vmax=residual_abs,
        ),
        save_station_map(
            geo,
            "absolute_error",
            "TEST absolute error",
            FIGURES / "test_absolute_error_map.png",
            cmap="magma",
            vmin=0,
            vmax=ae_max,
        ),
    ]


def summary_row(df, metrics, training_mean):
    obs = df["observed"]
    pred = df["predicted"]
    if training_mean is None:
        baseline_mae = baseline_rmse = baseline_improve_abs = baseline_improve_pct = float("nan")
    else:
        b_err = training_mean - obs
        baseline_mae = float(np.mean(np.abs(b_err)))
        baseline_rmse = float(np.sqrt(np.mean(b_err ** 2)))
        baseline_improve_abs = baseline_rmse - metrics["rmse"]
        baseline_improve_pct = 100 * baseline_improve_abs / baseline_rmse if baseline_rmse else float("nan")
    test_mean = float(obs.mean())
    tm_err = test_mean - obs
    return {
        "n": metrics["n"],
        "observed_mean": float(obs.mean()),
        "observed_std": float(obs.std(ddof=1)),
        "observed_min": float(obs.min()),
        "observed_max": float(obs.max()),
        "predicted_mean": float(pred.mean()),
        "predicted_std": float(pred.std(ddof=1)),
        "predicted_min": float(pred.min()),
        "predicted_max": float(pred.max()),
        "bias": metrics["bias"],
        "median_residual": float(df["residual"].median()),
        "residual_std": float(df["residual"].std(ddof=1)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "training_mean_baseline_value": training_mean,
        "training_mean_baseline_mae": baseline_mae,
        "training_mean_baseline_rmse": baseline_rmse,
        "training_mean_baseline_rmse_improvement": baseline_improve_abs,
        "training_mean_baseline_rmse_improvement_pct": baseline_improve_pct,
        "test_mean_reference_value": test_mean,
        "test_mean_reference_mae": float(np.mean(np.abs(tm_err))),
        "test_mean_reference_rmse": float(np.sqrt(np.mean(tm_err ** 2))),
    }


def save_summary(df, metrics, training_mean):
    row = summary_row(df, metrics, training_mean)
    path = TABLES / "test_summary.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path, row


def save_by_state(df, found):
    if found["state"] is None:
        return None
    rows = []
    for state, sub in df.groupby(found["state"], dropna=True):
        if pd.isna(state) or str(state).strip() == "":
            continue
        m = metric_values(sub["predicted"].to_numpy(), sub["observed"].to_numpy())
        rows.append({
            "state": state,
            "n": m["n"],
            "observed_mean": float(sub["observed"].mean()),
            "predicted_mean": float(sub["predicted"].mean()),
            "bias": m["bias"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"] if m["n"] >= 2 else float("nan"),
        })
    if not rows:
        return None
    path = TABLES / "test_metrics_by_state.csv"
    pd.DataFrame(rows).sort_values(["state"]).to_csv(path, index=False)
    return path


def save_largest_errors(df, found, n=20):
    keep = []
    for key in ("station_id", "lat", "lon", "state", "station_type"):
        if found[key] is not None and found[key] not in keep:
            keep.append(found[key])
    renamed = df.sort_values("absolute_error", ascending=False).head(n).copy()
    renamed = renamed[keep + ["observed", "predicted", "residual", "absolute_error"]]
    rename = {
        "observed": f"observed_{POLLUTANT}",
        "predicted": f"predicted_{POLLUTANT}",
        "residual": "residual_prediction_minus_observation",
    }
    path = TABLES / "largest_test_errors.csv"
    renamed.rename(columns=rename).to_csv(path, index=False)
    return path


def print_column_report(found, df):
    print("Prediction columns found")
    print("------------------------")
    for label in ("observed", "predicted", "lat", "lon", "station_id", "state", "station_type"):
        print(f"{label:>12}: {found[label] or 'not available'}")
    print(f"\nall columns: {', '.join(df.columns)}\n")


def run_analysis(predictions=PREDICTIONS):
    if not predictions.exists():
        raise SystemExit(f"ERROR: prediction file not found: {predictions}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(predictions)
    found = prediction_columns(raw)
    print_column_report(found, raw)
    require_columns(found)
    df = finite_analysis_frame(raw, found)
    if df.empty:
        raise SystemExit("ERROR: no rows have both observed and predicted PM2.5 values")

    metrics = metric_values(df["predicted"].to_numpy(), df["observed"].to_numpy())
    training_mean = load_training_mean_baseline()

    figure_paths = [
        save_scatter(df, metrics),
        save_distributions(df),
        save_residual_distribution(df),
        *save_maps(df, found),
    ]
    summary_path, summary = save_summary(df, metrics, training_mean)
    table_paths = [summary_path, save_largest_errors(df, found), save_by_state(df, found)]
    table_paths = [p for p in table_paths if p is not None]

    print("Summary")
    print("-------")
    print(f"n={metrics['n']}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}")
    print(f"observed mean/std/range: {summary['observed_mean']:.2f} / {summary['observed_std']:.2f} / "
          f"{summary['observed_min']:.2f}-{summary['observed_max']:.2f} {UNITS}")
    print(f"predicted mean/std/range: {summary['predicted_mean']:.2f} / {summary['predicted_std']:.2f} / "
          f"{summary['predicted_min']:.2f}-{summary['predicted_max']:.2f} {UNITS}")
    print(f"bias={summary['bias']:.2f}  median residual={summary['median_residual']:.2f}  "
          f"residual std={summary['residual_std']:.2f} {UNITS}")
    if training_mean is None or math.isnan(summary["training_mean_baseline_rmse"]):
        print("training-mean baseline: not available; final_model.pt with baseline_concentration was not read")
    else:
        print(f"training-mean baseline: RMSE={summary['training_mean_baseline_rmse']:.2f} "
              f"MAE={summary['training_mean_baseline_mae']:.2f}")
        print(f"model improvement vs training-mean baseline: "
              f"{summary['training_mean_baseline_rmse_improvement']:.2f} RMSE "
              f"({summary['training_mean_baseline_rmse_improvement_pct']:.1f}%)")
    print(f"TEST-mean reference: RMSE={summary['test_mean_reference_rmse']:.2f} "
          f"MAE={summary['test_mean_reference_mae']:.2f} (R² reference, not deployable)")
    print("\nGenerated figures")
    for path in figure_paths:
        print(f"  {path.relative_to(ROOT)}")
    print("\nGenerated tables")
    for path in table_paths:
        print(f"  {path.relative_to(ROOT)}")
    return found, summary, figure_paths, table_paths


def self_check():
    df = pd.DataFrame({
        "station_code": ["a", "b", "c"],
        "land": ["X", "X", "Y"],
        "lat": [52.0, 53.0, 54.0],
        "lon": [12.0, 13.0, 14.0],
        "true_pm25": [5.0, 7.0, 9.0],
        "pred_pm25": [6.0, 6.0, 10.0],
    })
    found = prediction_columns(df)
    assert found["observed"] == "true_pm25"
    assert found["predicted"] == "pred_pm25"
    assert found["state"] == "land"
    clean = finite_analysis_frame(df, found)
    m = metric_values(clean["predicted"].to_numpy(), clean["observed"].to_numpy())
    assert m["n"] == 3
    assert round(m["mae"], 6) == 1.0
    row = summary_row(clean, m, training_mean=7.0)
    assert row["training_mean_baseline_rmse"] > 0
    assert row["test_mean_reference_rmse"] > 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze frozen cnn_deep_wide sealed-TEST predictions.")
    parser.add_argument("--self-test", action="store_true", help="run lightweight internal checks and exit")
    args = parser.parse_args()
    if args.self_test:
        self_check()
    else:
        run_analysis()
