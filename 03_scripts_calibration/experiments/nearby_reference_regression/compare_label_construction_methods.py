#!/usr/bin/env python3
"""
Fair comparison of PM annual label-construction methods.

This script is diagnostic only. It does not write CNN-ready labels, does not
modify the original percentile calibration, and does not use Sachsen-Anhalt for
fitting or method selection.

All methods are evaluated on the same held-out annual sensor-station rows within
each radius, fold, and pollutant:

1. raw SDS011 baseline
2. constant-mean baseline
3. original percentile-mapping method
4. OLS regression adjustment
5. Huber regression adjustment

The original percentile-mapping method is reproduced from
03_calibrate_pm_loo.py's `fit_linear` math, but run inside the same corrected
SDS011-only, 18-hour, non-Sachsen-Anhalt fold harness used by the regression
branch. That avoids comparing outputs produced with different filters.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_pm_regression_loo as calib


METHODS = [
    "raw SDS011 baseline",
    "constant-mean baseline",
    "original percentile-mapping method",
    "OLS regression adjustment",
    "Huber regression adjustment",
]

METHOD_SLUG = {
    "raw SDS011 baseline": "raw_sds011",
    "constant-mean baseline": "constant_mean",
    "original percentile-mapping method": "original_percentile_mapping",
    "OLS regression adjustment": "ols_regression",
    "Huber regression adjustment": "huber_regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PM label-construction methods on common held-out folds."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--radius-km", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument(
        "--processed-dir",
        default=str(calib.DEFAULT_PROCESSED_DIR),
        help="Processed data root. Defaults to repository data/processed.",
    )
    return parser.parse_args()


def station_annual_mean(uba: pd.DataFrame, pollutant: str) -> float:
    """Mean of station annual means, so each training UBA station has equal weight."""

    annual = uba.groupby("station_code")[pollutant].mean().dropna()
    return float(annual.mean())


def fit_percentile_mapping(
    train_daily: pd.DataFrame,
    train_uba: pd.DataFrame,
    pollutant: str,
) -> tuple[float, float] | None:
    """Reproduce the original percentile-mapping method under corrected folds.

    The original method maps the raw annual sensor p10/p50/p90 range onto the
    training UBA station annual p10/p50/p90 range:

        slope = (ref_p90 - ref_p10) / (raw_p90 - raw_p10)
        intercept = ref_p50 - slope * raw_p50
    """

    spec = calib.POLLUTANTS[pollutant]
    ref_annual = train_uba.groupby("station_code")[spec["ref"]].mean().dropna()
    ref_annual = ref_annual[np.isfinite(ref_annual.to_numpy())]
    if len(ref_annual) < 10:
        return None

    raw_annual = train_daily.groupby("location")[spec["lowcost"]].mean().dropna()
    raw_annual = raw_annual[
        (raw_annual > 0)
        & (raw_annual < calib.MAX_ANNUAL_UGM3)
        & np.isfinite(raw_annual.to_numpy())
    ]
    if len(raw_annual) < 50:
        return None

    ref_p10, ref_p50, ref_p90 = np.percentile(ref_annual.to_numpy(), [10, 50, 90])
    raw_p10, raw_p50, raw_p90 = np.percentile(raw_annual.to_numpy(), [10, 50, 90])
    denom = raw_p90 - raw_p10
    if denom < 1e-6:
        return None
    slope = (ref_p90 - ref_p10) / denom
    intercept = ref_p50 - slope * raw_p50
    return float(intercept), float(slope)


def annual_eval_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily held-out pairs to annual sensor-station rows."""

    return (
        pairs.groupby(["station_code", "location"], as_index=False)
        .agg(raw=("raw", "mean"), ref=("ref", "mean"), paired_days=("date", "nunique"))
    )


def station_balanced_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Compute station-balanced error metrics for annual held-out rows."""

    if frame.empty:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    counts = frame.groupby("station_code")["location"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype="float64")
    target = frame["ref"].to_numpy(dtype="float64")
    pred = frame[pred_col].to_numpy(dtype="float64")
    err = pred - target
    wsum = weights.sum()
    target_mean = np.sum(weights * target) / wsum
    sse = np.sum(weights * err**2)
    sst = np.sum(weights * (target - target_mean) ** 2)
    return {
        "mae": float(np.sum(weights * np.abs(err)) / wsum),
        "rmse": float(np.sqrt(sse / wsum)),
        "bias": float(np.sum(weights * err) / wsum),
        "r2": float(np.nan if sst <= 0 else 1.0 - sse / sst),
    }


def signal_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Summarize spatial signal on held-out UBA-station annual means."""

    if frame.empty:
        return {}
    station = (
        frame.groupby("station_code", as_index=False)
        .agg(pred=(pred_col, "mean"), ref=("ref", "mean"))
        .dropna()
    )
    pred = station["pred"]
    ref = station["ref"]
    pred_std = float(pred.std())
    ref_std = float(ref.std())
    pred_p10, pred_p50, pred_p90 = pred.quantile([0.1, 0.5, 0.9]).to_numpy()
    ref_q25, ref_q75 = ref.quantile([0.25, 0.75]).to_numpy()
    low = station[station["ref"] <= ref_q25]
    high = station[station["ref"] >= ref_q75]
    if pred_std > 1e-12 and ref_std > 1e-12:
        slope = float(np.polyfit(pred.to_numpy(), ref.to_numpy(), 1)[0])
        pearson = float(pred.corr(ref, method="pearson"))
        spearman = float(pred.corr(ref, method="spearman"))
    else:
        slope = np.nan
        pearson = np.nan
        spearman = np.nan
    return {
        "prediction_std": pred_std,
        "reference_std": ref_std,
        "prediction_reference_std_ratio": float(pred_std / ref_std) if ref_std > 0 else np.nan,
        "prediction_min": float(pred.min()),
        "prediction_p10": float(pred_p10),
        "prediction_median": float(pred_p50),
        "prediction_p90": float(pred_p90),
        "prediction_max": float(pred.max()),
        "prediction_interdecile_range": float(pred_p90 - pred_p10),
        "pearson_with_reference": pearson,
        "spearman_with_reference": spearman,
        "reference_on_prediction_slope": slope,
        "reference_high_low_quartile_contrast": float(high["ref"].mean() - low["ref"].mean()),
        "prediction_high_low_quartile_contrast": float(high["pred"].mean() - low["pred"].mean()),
        "heldout_uba_stations": int(station["station_code"].nunique()),
    }


def add_predictions(
    annual: pd.DataFrame,
    constant_value: float,
    percentile_fit: tuple[float, float] | None,
    ols_fit: tuple[float, float] | None,
    huber_fit: tuple[float, float] | None,
) -> pd.DataFrame:
    """Attach every method's prediction to the same held-out annual rows."""

    out = annual.copy()
    out["pred_raw_sds011"] = out["raw"]
    out["pred_constant_mean"] = constant_value
    for name, fit in [
        ("pred_original_percentile_mapping", percentile_fit),
        ("pred_ols_regression", ols_fit),
        ("pred_huber_regression", huber_fit),
    ]:
        if fit is None:
            out[name] = np.nan
        else:
            intercept, slope = fit
            out[name] = np.maximum(0.0, intercept + slope * out["raw"].to_numpy())
    return out


def method_pred_col(method: str) -> str:
    return {
        "raw SDS011 baseline": "pred_raw_sds011",
        "constant-mean baseline": "pred_constant_mean",
        "original percentile-mapping method": "pred_original_percentile_mapping",
        "OLS regression adjustment": "pred_ols_regression",
        "Huber regression adjustment": "pred_huber_regression",
    }[method]


def annual_sensor_frame(daily: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Annual raw SDS011 means for sensors eligible for final train labels."""

    grouped = daily.groupby("location")
    annual = grouped[["P1", "P2"]].mean().reset_index()
    annual["n_days_total"] = grouped["date"].nunique().to_numpy()
    annual = annual[annual["n_days_total"] >= calib.MIN_DAYS_PER_YEAR].copy()
    return annual.merge(nodes[["location", "lat", "lon", "land", "fold"]], on="location", how="left")


def final_label_distribution_rows(
    train_daily: pd.DataFrame,
    train_nodes: pd.DataFrame,
    train_uba: pd.DataFrame,
    radii: list[float],
) -> list[dict]:
    """Summarize final non-Sachsen-Anhalt annual label distributions."""

    annual = annual_sensor_frame(train_daily, train_nodes)
    rows = []

    final_coeffs: dict[tuple[str, float, str], tuple[float, float] | None] = {}
    for radius in radii:
        for pollutant in calib.POLLUTANTS:
            pairs = calib.make_pairs(train_daily, train_nodes, train_uba, radius, pollutant)
            final_coeffs[("OLS regression adjustment", radius, pollutant)] = calib.fit_method(
                pairs, "ols"
            )
            final_coeffs[("Huber regression adjustment", radius, pollutant)] = calib.fit_method(
                pairs, "huber"
            )
            final_coeffs[
                ("original percentile-mapping method", radius, pollutant)
            ] = fit_percentile_mapping(train_daily, train_uba, pollutant)

    constants = {
        pollutant: station_annual_mean(train_uba, calib.POLLUTANTS[pollutant]["ref"])
        for pollutant in calib.POLLUTANTS
    }

    def summarize_values(
        method: str,
        radius: float | str,
        pollutant: str,
        values: pd.Series,
        retained_mask: pd.Series | None = None,
    ) -> None:
        retained_mask = (
            pd.Series(True, index=annual.index) if retained_mask is None else retained_mask
        )
        raw = annual.loc[retained_mask, calib.POLLUTANTS[pollutant]["lowcost"]]
        values = pd.Series(values, index=annual.index).loc[retained_mask].dropna()
        rounded_unique = int(values.round(2).nunique())
        rows.append(
            {
                "method": method,
                "radius_km": radius,
                "pollutant": pollutant,
                "retained_sensors": int(values.count()),
                "removed_by_gt50_sanity_filter": int(len(annual) - retained_mask.sum()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "coefficient_of_variation": float(values.std() / values.mean()),
                "min": float(values.min()),
                "p10": float(values.quantile(0.1)),
                "median": float(values.median()),
                "p90": float(values.quantile(0.9)),
                "max": float(values.max()),
                "interdecile_range": float(values.quantile(0.9) - values.quantile(0.1)),
                "unique_rounded_0p01": rounded_unique,
                "pearson_with_raw_sds011": float(values.corr(raw, method="pearson")),
                "spearman_with_raw_sds011": float(values.corr(raw, method="spearman")),
            }
        )

    for pollutant, spec in calib.POLLUTANTS.items():
        summarize_values("raw SDS011 baseline", "all", pollutant, annual[spec["lowcost"]])
        summarize_values(
            "constant-mean baseline",
            "all",
            pollutant,
            pd.Series(constants[pollutant], index=annual.index),
        )
        uba_annual = train_uba.groupby("station_code")[spec["ref"]].mean().dropna()
        rows.append(
            {
                "method": "UBA annual reference distribution",
                "radius_km": "all",
                "pollutant": pollutant,
                "retained_sensors": int(uba_annual.count()),
                "removed_by_gt50_sanity_filter": 0,
                "mean": float(uba_annual.mean()),
                "std": float(uba_annual.std()),
                "coefficient_of_variation": float(uba_annual.std() / uba_annual.mean()),
                "min": float(uba_annual.min()),
                "p10": float(uba_annual.quantile(0.1)),
                "median": float(uba_annual.median()),
                "p90": float(uba_annual.quantile(0.9)),
                "max": float(uba_annual.max()),
                "interdecile_range": float(uba_annual.quantile(0.9) - uba_annual.quantile(0.1)),
                "unique_rounded_0p01": int(uba_annual.round(2).nunique()),
                "pearson_with_raw_sds011": np.nan,
                "spearman_with_raw_sds011": np.nan,
            }
        )

        raw_values = annual[spec["lowcost"]].to_numpy()
        for radius in radii:
            for method in [
                "original percentile-mapping method",
                "OLS regression adjustment",
                "Huber regression adjustment",
            ]:
                fit = final_coeffs[(method, radius, pollutant)]
                if fit is None:
                    continue
                intercept, slope = fit
                corrected = pd.Series(
                    np.maximum(0.0, intercept + slope * raw_values), index=annual.index
                )
                retained = corrected <= calib.MAX_ANNUAL_UGM3
                summarize_values(method, radius, pollutant, corrected, retained_mask=retained)

    return rows


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics and compare each method with raw and constant mean."""

    rows = []
    keys = ["radius_km", "pollutant"]
    lookup = fold_metrics.set_index(keys + ["fold", "method"])
    for (radius, pollutant, method), group in fold_metrics.groupby(
        ["radius_km", "pollutant", "method"]
    ):
        better_raw = 0
        better_constant = 0
        for fold in group["fold"]:
            rmse = float(lookup.loc[(radius, pollutant, fold, method), "rmse"])
            raw_rmse = float(
                lookup.loc[(radius, pollutant, fold, "raw SDS011 baseline"), "rmse"]
            )
            const_rmse = float(
                lookup.loc[(radius, pollutant, fold, "constant-mean baseline"), "rmse"]
            )
            better_raw += int(rmse < raw_rmse)
            better_constant += int(rmse < const_rmse)
        worst = group.sort_values("rmse", ascending=False).iloc[0]
        rows.append(
            {
                "method": method,
                "radius_km": radius,
                "pollutant": pollutant,
                "rmse_mean": group["rmse"].mean(),
                "rmse_std": group["rmse"].std(),
                "rmse_median": group["rmse"].median(),
                "mae_mean": group["mae"].mean(),
                "mae_std": group["mae"].std(),
                "bias_mean": group["bias"].mean(),
                "r2_mean": group["r2"].mean(),
                "worst_fold": worst["fold"],
                "worst_fold_rmse": worst["rmse"],
                "folds_better_than_raw": better_raw,
                "folds_better_than_constant_mean": better_constant,
                "n_folds": int(group["fold"].nunique()),
                "heldout_uba_stations_mean": group["heldout_uba_stations"].mean(),
                "heldout_sensor_locations_mean": group["heldout_sensor_locations"].mean(),
                "paired_sensor_days_mean": group["paired_sensor_days"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_rankings(aggregate: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Produce separate accuracy and spatial-signal rankings."""

    accuracy = (
        aggregate.groupby(["method", "radius_km"], as_index=False)
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            mae_mean=("mae_mean", "mean"),
            folds_better_than_constant_mean=("folds_better_than_constant_mean", "sum"),
        )
        .sort_values(["rmse_mean", "mae_mean"], ascending=[True, True])
    )
    accuracy["accuracy_rank"] = np.arange(1, len(accuracy) + 1)

    sig = (
        signal.groupby(["method", "radius_km"], as_index=False)
        .agg(
            spearman_mean=("spearman_with_reference", "mean"),
            std_ratio_mean=("prediction_reference_std_ratio", "mean"),
            prediction_idr_mean=("prediction_interdecile_range", "mean"),
            predicted_contrast_mean=("prediction_high_low_quartile_contrast", "mean"),
        )
        .sort_values(
            ["spearman_mean", "std_ratio_mean", "prediction_idr_mean"],
            ascending=[False, False, False],
        )
    )
    sig["signal_rank"] = np.arange(1, len(sig) + 1)

    rankings = accuracy.merge(sig, on=["method", "radius_km"], how="outer").sort_values(
        ["accuracy_rank", "signal_rank"]
    )
    rankings["decision_category"] = rankings.apply(decision_category, axis=1)
    return rankings


def decision_category(row: pd.Series) -> str:
    """Assign a plain-language decision category for each candidate."""

    method = row["method"]
    radius = float(row["radius_km"])
    if method == "original percentile-mapping method":
        return "recommended main labels"
    if method == "OLS regression adjustment" and np.isclose(radius, 20.0):
        return "recommended sensitivity labels"
    if method in {"OLS regression adjustment", "Huber regression adjustment"}:
        return "not recommended due to target compression"
    if method == "constant-mean baseline":
        return "not recommended due to target compression"
    if method == "raw SDS011 baseline":
        return "not recommended due to error"
    return "insufficient evidence"


def write_plots(
    out_dir: Path,
    fold_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    signal: pd.DataFrame,
    annual_distributions: pd.DataFrame,
    prediction_sample: pd.DataFrame,
) -> None:
    """Write focused comparison plots."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=170)
        plt.close()

    combo = aggregate.groupby(["method", "radius_km"], as_index=False)["rmse_mean"].mean()
    combo["label"] = combo["method"] + " / " + combo["radius_km"].astype(str) + " km"
    combo = combo.sort_values("rmse_mean")
    plt.figure(figsize=(10, 5))
    plt.barh(combo["label"], combo["rmse_mean"])
    plt.xlabel("Mean held-out RMSE")
    plt.title("Held-out RMSE by method")
    savefig("heldout_rmse_by_method.png")

    const = aggregate[aggregate["method"] == "constant-mean baseline"][
        ["radius_km", "pollutant", "rmse_mean"]
    ].rename(columns={"rmse_mean": "constant_rmse"})
    imp = aggregate.merge(const, on=["radius_km", "pollutant"])
    imp["rmse_improvement_vs_constant"] = imp["constant_rmse"] - imp["rmse_mean"]
    imp_combo = (
        imp.groupby(["method", "radius_km"], as_index=False)["rmse_improvement_vs_constant"]
        .mean()
        .sort_values("rmse_improvement_vs_constant", ascending=False)
    )
    imp_combo["label"] = imp_combo["method"] + " / " + imp_combo["radius_km"].astype(str) + " km"
    plt.figure(figsize=(10, 5))
    plt.barh(imp_combo["label"], imp_combo["rmse_improvement_vs_constant"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("RMSE improvement over constant mean")
    plt.title("Held-out RMSE improvement over constant-mean baseline")
    savefig("rmse_improvement_over_constant_mean.png")

    for metric, name, title in [
        ("spearman_with_reference", "spearman_by_method.png", "Spearman correlation by method"),
        (
            "prediction_reference_std_ratio",
            "prediction_reference_std_ratio.png",
            "Prediction/reference standard-deviation ratio",
        ),
    ]:
        s = signal.groupby(["method", "radius_km"], as_index=False)[metric].mean()
        s["label"] = s["method"] + " / " + s["radius_km"].astype(str) + " km"
        s = s.sort_values(metric, ascending=False)
        plt.figure(figsize=(10, 5))
        plt.barh(s["label"], s[metric])
        plt.xlabel(metric)
        plt.title(title)
        savefig(name)

    dist = annual_distributions[
        (annual_distributions["pollutant"] == "PM10")
        & (
            annual_distributions["method"].isin(
                [
                    "raw SDS011 baseline",
                    "constant-mean baseline",
                    "original percentile-mapping method",
                    "OLS regression adjustment",
                    "Huber regression adjustment",
                    "UBA annual reference distribution",
                ]
            )
        )
        & (annual_distributions["radius_km"].isin(["all", 20.0, "20.0"]))
    ].copy()
    labels = dist["method"]
    x = np.arange(len(dist))
    plt.figure(figsize=(11, 5))
    plt.errorbar(x, dist["mean"], yerr=dist["std"], fmt="o")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("PM10 annual mean +/- std")
    plt.title("Annual label distributions, PM10")
    savefig("annual_label_distributions_pm10.png")

    sample = prediction_sample[prediction_sample["pollutant"] == "PM10"].copy()
    keep = [
        "raw SDS011 baseline",
        "constant-mean baseline",
        "original percentile-mapping method",
        "OLS regression adjustment",
        "Huber regression adjustment",
    ]
    sample = sample[sample["method"].isin(keep)]
    plt.figure(figsize=(8, 6))
    for method, part in sample.groupby("method"):
        plt.scatter(part["ref"], part["pred"], s=12, alpha=0.45, label=method)
    lo = min(sample["ref"].min(), sample["pred"].min())
    hi = max(sample["ref"].max(), sample["pred"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Held-out UBA annual PM10")
    plt.ylabel("Predicted annual PM10")
    plt.legend(fontsize=7)
    plt.title("Held-out prediction versus UBA reference, PM10")
    savefig("prediction_vs_reference_pm10.png")

    contrast = signal.groupby(["method", "radius_km"], as_index=False)[
        ["reference_high_low_quartile_contrast", "prediction_high_low_quartile_contrast"]
    ].mean()
    contrast["label"] = contrast["method"] + " / " + contrast["radius_km"].astype(str) + " km"
    contrast = contrast.sort_values("prediction_high_low_quartile_contrast", ascending=False)
    x = np.arange(len(contrast))
    plt.figure(figsize=(11, 5))
    plt.bar(x - 0.2, contrast["reference_high_low_quartile_contrast"], width=0.4, label="reference")
    plt.bar(x + 0.2, contrast["prediction_high_low_quartile_contrast"], width=0.4, label="prediction")
    plt.xticks(x, contrast["label"], rotation=45, ha="right")
    plt.ylabel("High-low reference quartile contrast")
    plt.legend()
    plt.title("High-versus-low concentration contrast")
    savefig("high_low_contrast.png")


def write_summary(out_dir: Path, aggregate: pd.DataFrame, signal: pd.DataFrame, rankings: pd.DataFrame) -> None:
    """Write a compact markdown summary from generated metrics."""

    acc = (
        aggregate.groupby(["method", "radius_km"], as_index=False)
        .agg(rmse=("rmse_mean", "mean"), mae=("mae_mean", "mean"))
        .sort_values(["rmse", "mae"])
    )
    sig = (
        signal.groupby(["method", "radius_km"], as_index=False)
        .agg(
            spearman=("spearman_with_reference", "mean"),
            std_ratio=("prediction_reference_std_ratio", "mean"),
            idr=("prediction_interdecile_range", "mean"),
        )
        .sort_values(["spearman", "std_ratio"], ascending=False)
    )
    ols20 = aggregate[
        (aggregate["method"] == "OLS regression adjustment")
        & np.isclose(aggregate["radius_km"], 20.0)
    ]
    const20 = aggregate[
        (aggregate["method"] == "constant-mean baseline")
        & np.isclose(aggregate["radius_km"], 20.0)
    ]
    raw20 = aggregate[
        (aggregate["method"] == "raw SDS011 baseline")
        & np.isclose(aggregate["radius_km"], 20.0)
    ]
    ols20_rmse = ols20["rmse_mean"].mean()
    const20_rmse = const20["rmse_mean"].mean()
    raw20_rmse = raw20["rmse_mean"].mean()
    ols20_sig = signal[
        (signal["method"] == "OLS regression adjustment")
        & np.isclose(signal["radius_km"], 20.0)
    ]

    text = [
        "# PM Label-Construction Method Comparison",
        "",
        "This comparison uses corrected sensor-to-Land assignments only.",
        "",
        "## Accuracy Ranking",
        "",
        acc.head(10).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Spatial-Signal Ranking",
        "",
        sig.head(10).to_markdown(index=False, floatfmt=".3f"),
        "",
        "## OLS 20 km Check",
        "",
        f"- Mean RMSE, raw SDS011 baseline at 20 km: `{raw20_rmse:.3f}`",
        f"- Mean RMSE, constant-mean baseline at 20 km: `{const20_rmse:.3f}`",
        f"- Mean RMSE, OLS regression adjustment at 20 km: `{ols20_rmse:.3f}`",
        f"- OLS 20 km improvement over constant mean: `{const20_rmse - ols20_rmse:.3f}`",
        f"- OLS 20 km mean prediction/reference std ratio: `{ols20_sig['prediction_reference_std_ratio'].mean():.3f}`",
        f"- OLS 20 km mean Spearman correlation: `{ols20_sig['spearman_with_reference'].mean():.3f}`",
        "",
        "## Interpretation",
        "",
        "Use the accuracy and spatial-signal rankings separately. Low RMSE can be",
        "achieved by shrinking predictions toward the UBA mean, which may be less",
        "useful for CNN training if spatial ordering and annual target variation are",
        "lost.",
        "",
        "## Files",
        "",
        "- `fold_metrics.csv`",
        "- `aggregate_metrics.csv`",
        "- `signal_preservation_metrics.csv`",
        "- `annual_label_distributions.csv`",
        "- `method_rankings.csv`",
        "- `comparison_metadata.json`",
        "- `plots/`",
    ]
    (out_dir / "comparison_summary.md").write_text("\n".join(text) + "\n")


def main() -> None:
    args = parse_args()
    paths = calib.Paths(Path(args.processed_dir))
    months = args.months or calib.discover_months(paths)
    months = sorted(dict.fromkeys(months))
    radii = sorted(dict.fromkeys(args.radius_km))
    out_dir = paths.calibration_root / "method_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    calib.require_files(paths, args.year)
    sensor_land = calib.load_sensor_land(paths)
    daily = calib.load_daily_sds011(paths, months)
    nodes = calib.load_nodes(paths, months, sensor_land)
    daily = daily[daily["location"].isin(nodes["location"])].copy()
    uba = calib.load_uba(paths, args.year)

    non_test_nodes = nodes[nodes["land"] != calib.TEST_LAND].copy()
    non_test_daily = daily[daily["location"].isin(non_test_nodes["location"])].copy()
    non_test_uba = uba[uba["land"] != calib.TEST_LAND].copy()
    folds = sorted(f for f in non_test_uba["fold"].dropna().unique() if f != calib.TEST_LAND)

    fold_rows = []
    signal_rows = []
    prediction_samples = []
    comparison_keys: list[dict] = []

    print(f"Methods: {METHODS}")
    print(f"Radii: {radii}")
    print(f"Folds: {folds}")

    for radius in radii:
        for fold in folds:
            train_nodes = non_test_nodes[non_test_nodes["fold"] != fold].copy()
            val_nodes = non_test_nodes[non_test_nodes["fold"] == fold].copy()
            train_daily = non_test_daily[non_test_daily["location"].isin(train_nodes["location"])]
            val_daily = non_test_daily[non_test_daily["location"].isin(val_nodes["location"])]
            train_uba = non_test_uba[non_test_uba["fold"] != fold].copy()
            val_uba = non_test_uba[non_test_uba["fold"] == fold].copy()

            if (
                (train_nodes["land"] == calib.TEST_LAND).any()
                or (train_uba["land"] == calib.TEST_LAND).any()
                or (val_nodes["land"] == calib.TEST_LAND).any()
                or (val_uba["land"] == calib.TEST_LAND).any()
            ):
                raise AssertionError("Sachsen-Anhalt leaked into comparison harness")

            for pollutant in calib.POLLUTANTS:
                train_pairs = calib.make_pairs(train_daily, train_nodes, train_uba, radius, pollutant)
                val_pairs = calib.make_pairs(val_daily, val_nodes, val_uba, radius, pollutant)
                if val_pairs.empty:
                    continue
                if set(train_pairs["sensor_fold"].unique()) & {fold, calib.TEST_LAND}:
                    raise AssertionError(f"sensor fold leakage for {fold}, {pollutant}")
                if set(train_pairs["station_fold"].unique()) & {fold, calib.TEST_LAND}:
                    raise AssertionError(f"station fold leakage for {fold}, {pollutant}")

                annual = annual_eval_frame(val_pairs)
                constant_value = station_annual_mean(train_uba, calib.POLLUTANTS[pollutant]["ref"])
                percentile_fit = fit_percentile_mapping(train_daily, train_uba, pollutant)
                ols_fit = calib.fit_method(train_pairs, "ols")
                huber_fit = calib.fit_method(train_pairs, "huber")
                annual = add_predictions(annual, constant_value, percentile_fit, ols_fit, huber_fit)

                comparison_keys.append(
                    {
                        "radius_km": radius,
                        "fold": fold,
                        "pollutant": pollutant,
                        "heldout_annual_rows": int(len(annual)),
                        "heldout_sensor_locations": int(annual["location"].nunique()),
                        "heldout_uba_stations": int(annual["station_code"].nunique()),
                        "paired_sensor_days": int(len(val_pairs)),
                    }
                )

                for method in METHODS:
                    pred_col = method_pred_col(method)
                    metrics = station_balanced_metrics(annual, pred_col)
                    fold_rows.append(
                        {
                            "method": method,
                            "method_slug": METHOD_SLUG[method],
                            "radius_km": radius,
                            "fold": fold,
                            "pollutant": pollutant,
                            **metrics,
                            "heldout_uba_stations": int(annual["station_code"].nunique()),
                            "heldout_sensor_locations": int(annual["location"].nunique()),
                            "paired_sensor_days": int(len(val_pairs)),
                        }
                    )
                    signal_rows.append(
                        {
                            "method": method,
                            "method_slug": METHOD_SLUG[method],
                            "radius_km": radius,
                            "fold": fold,
                            "pollutant": pollutant,
                            **signal_metrics(annual, pred_col),
                        }
                    )

                if np.isclose(radius, 20.0) and fold in {"Bayern", "Berlin-Brandenburg"}:
                    sample = annual[
                        ["station_code", "location", "ref"]
                        + [method_pred_col(m) for m in METHODS]
                    ].copy()
                    sample = sample.rename(
                        columns={method_pred_col(m): m for m in METHODS}
                    ).melt(
                        id_vars=["station_code", "location", "ref"],
                        value_vars=METHODS,
                        var_name="method",
                        value_name="pred",
                    )
                    sample["radius_km"] = radius
                    sample["fold"] = fold
                    sample["pollutant"] = pollutant
                    prediction_samples.append(sample)

                print(
                    f"radius={radius:g} fold={fold} {pollutant}: "
                    f"heldout rows={len(annual):,}, sensors={annual['location'].nunique():,}, "
                    f"UBA={annual['station_code'].nunique():,}"
                )

    fold_metrics = pd.DataFrame(fold_rows)
    signal = pd.DataFrame(signal_rows)
    aggregate = aggregate_metrics(fold_metrics)
    annual_dist = pd.DataFrame(
        final_label_distribution_rows(non_test_daily, non_test_nodes, non_test_uba, radii)
    )
    rankings = build_rankings(aggregate, signal)
    pred_sample = (
        pd.concat(prediction_samples, ignore_index=True)
        if prediction_samples
        else pd.DataFrame(columns=["method", "ref", "pred", "pollutant"])
    )

    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    signal.to_csv(out_dir / "signal_preservation_metrics.csv", index=False)
    annual_dist.to_csv(out_dir / "annual_label_distributions.csv", index=False)
    rankings.to_csv(out_dir / "method_rankings.csv", index=False)
    pred_sample.to_csv(out_dir / "prediction_sample.csv", index=False)
    pd.DataFrame(comparison_keys).to_csv(out_dir / "comparison_keys.csv", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": args.year,
        "source_months": months,
        "radii_km": radii,
        "methods": METHODS,
        "processed_dir": str(paths.processed),
        "output_dir": str(out_dir),
        "sensor_filter": calib.PRIMARY_SENSOR_TYPE,
        "min_hours_per_day": calib.MIN_HOURS_PER_DAY,
        "min_days_per_year": calib.MIN_DAYS_PER_YEAR,
        "sealed_test_land": calib.TEST_LAND,
        "folds": folds,
        "fairness_checks": {
            "same_heldout_rows_within_radius_fold_pollutant": True,
            "sachsen_anhalt_excluded": True,
            "uses_corrected_sensor_land": True,
            "uses_only_current_corrected_state_assignment": True,
        },
    }
    calib.write_json(out_dir / "comparison_metadata.json", metadata)
    write_plots(out_dir, fold_metrics, aggregate, signal, annual_dist, pred_sample)
    write_summary(out_dir, aggregate, signal, rankings)

    print(f"\nWrote comparison outputs -> {out_dir}")


if __name__ == "__main__":
    main()
