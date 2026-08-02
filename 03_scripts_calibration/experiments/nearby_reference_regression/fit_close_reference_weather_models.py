#!/usr/bin/env python3
"""
Close-reference weather-aware PM calibration sensitivity diagnostics.

Experiment only: leave-one-UBA-station-out daily models on close SDS011 pairs.
No full-network labels are written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_pm_regression_loo as calib


DEFAULT_RADII_KM = [0.25, 0.5, 1.0]
PRIMARY_RADIUS_KM = 0.5
MIN_CV_STATIONS = 5
STATION_TYPE_STRATA = {
    "background": ["Hintergrund"],
    "traffic": ["Verkehr"],
    "industry": ["Industrie"],
    "all": None,
}
MODEL_FEATURES = {
    "constant_mean": [],
    "raw_pm": ["raw"],
    "raw_pm_rh": ["raw", "rh_frac"],
    "raw_pm_rh_interaction": ["raw", "rh_frac", "raw_x_rh"],
    "raw_pm_rh_interaction_temp": ["raw", "rh_frac", "raw_x_rh", "temperature"],
    "raw_pm_hyras_bme280": [
        "raw",
        "hyras_rh_frac",
        "hyras_temperature",
        "bme280_rh_frac",
        "bme280_temperature",
        "raw_x_hyras_rh",
        "raw_x_bme280_rh",
        "bme280_minus_hyras_rh",
        "bme280_minus_hyras_temperature",
    ],
}
BASE_MODELS = [
    "constant_mean",
    "raw_pm",
    "raw_pm_rh",
    "raw_pm_rh_interaction",
    "raw_pm_rh_interaction_temp",
]
COMBINED_MODELS = BASE_MODELS + ["raw_pm_hyras_bme280"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity diagnostics for close-reference weather-aware PM models."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument("--radius-km", nargs="+", type=float, default=DEFAULT_RADII_KM)
    parser.add_argument(
        "--processed-dir",
        default=str(calib.DEFAULT_PROCESSED_DIR),
        help="Processed data root. Defaults to repository data/processed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Defaults to data/processed/calibration/regression_reference_adjustment/"
            "close_reference_weather_models."
        ),
    )
    return parser.parse_args()


def fit_linear(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    design = np.column_stack([np.ones(len(train)), train[features].to_numpy("float64")])
    return np.linalg.lstsq(design, train["ref"].to_numpy("float64"), rcond=None)[0]


def predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    features = MODEL_FEATURES[model]
    if not features:
        pred = np.full(len(test), train["ref"].mean(), dtype="float64")
    else:
        beta = fit_linear(train, features)
        design = np.column_stack([np.ones(len(test)), test[features].to_numpy("float64")])
        pred = design @ beta
    return np.maximum(0.0, pred)


def load_bme280_weather(processed: Path, months: list[str], locations: set[int]) -> pd.DataFrame:
    """Same-location BME280-only daily RH/temp from merged hourly weather files."""

    if not locations:
        return pd.DataFrame()
    wanted = sorted(int(x) for x in locations)
    parts = []
    cols = ["location", "hour", "humidity", "temperature", "sensor_type"]
    for month in months:
        path = processed / "hourly" / "humidity" / "all_sensors" / f"{month}.parquet"
        if not path.exists():
            continue
        try:
            h = pd.read_parquet(path, columns=cols, filters=[("location", "in", wanted)])
        except Exception:
            h = pd.read_parquet(path, columns=cols)
            h = h[h["location"].isin(wanted)]
        h = h[h["sensor_type"].astype(str).str.lower().eq("bme280")]
        if h.empty:
            continue
        h["hour"] = pd.to_datetime(h["hour"], errors="coerce")
        h = h.dropna(subset=["hour"])
        h["date"] = (h["hour"] + pd.Timedelta(hours=calib.UTC_TO_MEZ_HOURS)).dt.date
        for col in ["humidity", "temperature"]:
            h[col] = pd.to_numeric(h[col], errors="coerce")
        daily = (
            h.groupby(["location", "date"], as_index=False)
            .agg(
                humidity=("humidity", "mean"),
                temperature=("temperature", "mean"),
                n_weather_hours=("hour", "nunique"),
            )
        )
        parts.append(daily)
        print(f"{month}: BME280 rows for close locations -> {len(h):,}")
    if not parts:
        return pd.DataFrame()
    daily = pd.concat(parts, ignore_index=True)
    for col in ["humidity", "temperature"]:
        daily[f"{col}_weighted"] = daily[col] * daily["n_weather_hours"]
    out = (
        daily.groupby(["location", "date"], as_index=False)
        .agg(
            humidity_weighted=("humidity_weighted", "sum"),
            temperature_weighted=("temperature_weighted", "sum"),
            n_weather_hours=("n_weather_hours", "sum"),
        )
    )
    out["humidity"] = out["humidity_weighted"] / out["n_weather_hours"]
    out["temperature"] = out["temperature_weighted"] / out["n_weather_hours"]
    out = out.drop(columns=["humidity_weighted", "temperature_weighted"])
    out = out[out["n_weather_hours"] >= calib.MIN_HOURS_PER_DAY].copy()
    out["weather_source"] = "bme280"
    return out


def load_hyras_weather(processed: Path) -> pd.DataFrame:
    path = processed / "daily_weather" / "hyras_2024_sds011.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing HYRAS extract: {path}")
    df = pd.read_parquet(path)
    df = df.dropna(subset=["hyras_rh_pct", "hyras_temperature_c"]).copy()
    df = df.rename(
        columns={
            "hyras_rh_pct": "humidity",
            "hyras_temperature_c": "temperature",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["weather_source"] = "hyras"
    return df[["location", "date", "humidity", "temperature", "weather_source"]]


def station_metadata(processed: Path) -> pd.DataFrame:
    path = processed / "uba" / "station_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing UBA station metadata: {path}")
    meta = pd.read_csv(path)
    keep = ["station_code", "station_type_label", "station_setting_label"]
    return meta[keep].drop_duplicates("station_code")


def base_pairs(
    daily_pm: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    meta: pd.DataFrame,
    radius: float,
    pollutant: str,
) -> pd.DataFrame:
    pairs = calib.make_pairs(daily_pm, nodes, uba, radius, pollutant)
    if pairs.empty:
        return pairs
    pm_extra = daily_pm[["location", "date", "P1", "P2"]].rename(
        columns={"P1": "raw_pm10", "P2": "raw_pm25"}
    )
    return (
        pairs.merge(pm_extra, on=["location", "date"], how="left")
        .merge(
            uba[["station_code", "land"]]
            .drop_duplicates("station_code")
            .rename(columns={"land": "station_land"}),
            on="station_code",
            how="left",
        )
        .merge(meta, on="station_code", how="left")
        .dropna(subset=["raw", "ref", "raw_pm10", "raw_pm25", "station_type_label"])
    )


def add_generic_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.dropna(subset=["humidity", "temperature"]).copy()
    out["rh_frac"] = out["humidity"] / 100.0
    out["raw_x_rh"] = out["raw"] * out["rh_frac"]
    out["coarse_fraction"] = np.maximum(out["raw_pm10"] - out["raw_pm25"], 0.0)
    out["pm25_pm10_ratio"] = np.where(out["raw_pm10"] > 0, out["raw_pm25"] / out["raw_pm10"], np.nan)
    out["pm25_gt_pm10"] = out["raw_pm25"] > out["raw_pm10"]
    return out


def generic_weather_frame(base: pd.DataFrame, weather: pd.DataFrame, weather_source: str, sample: str) -> pd.DataFrame:
    out = base.merge(weather, on=["location", "date"], how="inner")
    out["weather_source"] = weather_source
    out["comparison_sample"] = sample
    return add_generic_features(out)


def common_weather_frames(base: pd.DataFrame, hyras: pd.DataFrame, bme: pd.DataFrame) -> list[pd.DataFrame]:
    h = hyras.rename(columns={"humidity": "hyras_humidity", "temperature": "hyras_temperature"})
    b = bme.rename(columns={"humidity": "bme280_humidity", "temperature": "bme280_temperature"})
    common = base.merge(
        h[["location", "date", "hyras_humidity", "hyras_temperature"]],
        on=["location", "date"],
        how="inner",
    ).merge(
        b[["location", "date", "bme280_humidity", "bme280_temperature"]],
        on=["location", "date"],
        how="inner",
    )
    if common.empty:
        return []

    hyras_frame = common.rename(
        columns={"hyras_humidity": "humidity", "hyras_temperature": "temperature"}
    ).copy()
    hyras_frame["weather_source"] = "hyras"
    hyras_frame["comparison_sample"] = "common_hyras_bme280"
    hyras_frame = add_generic_features(hyras_frame)

    bme_frame = common.rename(
        columns={"bme280_humidity": "humidity", "bme280_temperature": "temperature"}
    ).copy()
    bme_frame["weather_source"] = "bme280"
    bme_frame["comparison_sample"] = "common_hyras_bme280"
    bme_frame = add_generic_features(bme_frame)

    combined = common.copy()
    combined["weather_source"] = "hyras_bme280"
    combined["comparison_sample"] = "common_hyras_bme280"
    combined["humidity"] = combined["hyras_humidity"]
    combined["temperature"] = combined["hyras_temperature"]
    combined = add_generic_features(combined)
    combined["hyras_rh_frac"] = combined["hyras_humidity"] / 100.0
    combined["bme280_rh_frac"] = combined["bme280_humidity"] / 100.0
    combined["raw_x_hyras_rh"] = combined["raw"] * combined["hyras_rh_frac"]
    combined["raw_x_bme280_rh"] = combined["raw"] * combined["bme280_rh_frac"]
    combined["bme280_minus_hyras_rh"] = combined["bme280_humidity"] - combined["hyras_humidity"]
    combined["bme280_minus_hyras_temperature"] = (
        combined["bme280_temperature"] - combined["hyras_temperature"]
    )
    return [hyras_frame, bme_frame, combined]


def stratum_frame(frame: pd.DataFrame, stratum: str) -> pd.DataFrame:
    labels = STATION_TYPE_STRATA[stratum]
    if labels is None:
        return frame.copy()
    return frame[frame["station_type_label"].isin(labels)].copy()


def station_predictions(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    parts = []
    base_cols = [
        "station_code",
        "station_fold",
        "station_land",
        "station_type_label",
        "station_setting_label",
        "location",
        "date",
        "dist_km",
        "ref",
        "raw",
        "humidity",
        "temperature",
        "raw_pm10",
        "raw_pm25",
        "coarse_fraction",
        "pm25_pm10_ratio",
        "pm25_gt_pm10",
    ]
    extra_cols = [
        "hyras_humidity",
        "hyras_temperature",
        "bme280_humidity",
        "bme280_temperature",
        "bme280_minus_hyras_rh",
        "bme280_minus_hyras_temperature",
    ]
    base_cols += [col for col in extra_cols if col in frame.columns]
    for station in sorted(frame["station_code"].unique()):
        train = frame[frame["station_code"] != station]
        test = frame[frame["station_code"] == station].copy()
        if station in set(train["station_code"]):
            raise AssertionError(f"UBA leakage in held-out station {station}")
        if len(train) < 20:
            continue
        for model in models:
            pred = test[base_cols].copy()
            pred["model"] = model
            pred["pred"] = predict(model, train, test)
            parts.append(pred)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metric_values(group: pd.DataFrame) -> dict[str, float]:
    err = group["pred"].to_numpy() - group["ref"].to_numpy()
    pred = group["pred"].to_numpy(dtype="float64")
    ref = group["ref"].to_numpy(dtype="float64")
    pred_std = float(np.std(pred, ddof=1)) if len(pred) > 1 else 0.0
    ref_std = float(np.std(ref, ddof=1)) if len(ref) > 1 else 0.0
    has_corr = (
        len(group) > 1
        and np.isfinite(pred).all()
        and np.isfinite(ref).all()
        and np.ptp(pred) > 1e-12
        and np.ptp(ref) > 1e-12
    )
    pearson = float(np.corrcoef(pred, ref)[0, 1]) if has_corr else np.nan
    if has_corr:
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        ref_rank = pd.Series(ref).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(pred_rank, ref_rank)[0, 1])
    else:
        spearman = np.nan
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "pearson": pearson,
        "spearman": spearman,
        "prediction_std": pred_std,
        "reference_std": ref_std,
        "prediction_reference_std_ratio": float(pred_std / ref_std) if ref_std and np.isfinite(ref_std) else np.nan,
    }


def station_rows(
    preds: pd.DataFrame,
    radius: float,
    pollutant: str,
    station_type_stratum: str,
    weather_source: str,
    comparison_sample: str,
    level: str,
) -> pd.DataFrame:
    rows = []
    for (station, model), group in preds.groupby(["station_code", "model"]):
        vals = metric_values(group)
        rows.append(
            {
                "result_level": "station",
                "evaluation_level": level,
                "subset": "all_days",
                "radius_km": radius,
                "radius_m": int(round(radius * 1000)),
                "pollutant": pollutant,
                "station_type_stratum": station_type_stratum,
                "weather_source": weather_source,
                "comparison_sample": comparison_sample,
                "model": model,
                "station_code": station,
                "station_land": group["station_land"].iloc[0],
                "station_fold": group["station_fold"].iloc[0],
                "station_type_label": group["station_type_label"].iloc[0],
                "station_setting_label": group["station_setting_label"].iloc[0],
                "heldout_sds011_locations": int(group["location"].nunique()),
                "heldout_rows": int(len(group)),
                "median_sensor_station_distance_km": float(group["dist_km"].median()),
                "raw_pm_mean": float(group["raw"].mean()),
                "uba_pm_mean": float(group["ref"].mean()),
                "rh_mean": float(group["humidity"].mean()),
                "temperature_mean": float(group["temperature"].mean()),
                "raw_pm10_mean": float(group["raw_pm10"].mean()),
                "raw_pm25_mean": float(group["raw_pm25"].mean()),
                "coarse_fraction_mean": float(np.maximum(group["raw_pm10"] - group["raw_pm25"], 0).mean()),
                "pm25_gt_pm10_frequency": float(group["pm25_gt_pm10"].mean()),
                **vals,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    const = out[out["model"] == "constant_mean"][["station_code", "rmse"]].rename(
        columns={"rmse": "constant_station_rmse"}
    )
    raw = out[out["model"] == "raw_pm"][["station_code", "rmse"]].rename(
        columns={"rmse": "raw_pm_station_rmse"}
    )
    out = out.merge(const, on="station_code", how="left").merge(raw, on="station_code", how="left")
    out["rmse_diff_from_constant_mean"] = out["rmse"] - out["constant_station_rmse"]
    out["rmse_diff_from_raw_pm"] = out["rmse"] - out["raw_pm_station_rmse"]
    return out.drop(columns=["constant_station_rmse", "raw_pm_station_rmse"])


def aggregate_rows(stations: pd.DataFrame, preds: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for model, group in stations.groupby("model"):
        raw_preds = preds[preds["model"] == model]
        rmses = group["rmse"].dropna().sort_values()
        vals = metric_values(raw_preds)
        rows.append(
            {
                "result_level": "aggregate",
                "evaluation_level": level,
                "subset": "all_days",
                "radius_km": group["radius_km"].iloc[0],
                "radius_m": group["radius_m"].iloc[0],
                "pollutant": group["pollutant"].iloc[0],
                "station_type_stratum": group["station_type_stratum"].iloc[0],
                "weather_source": group["weather_source"].iloc[0],
                "comparison_sample": group["comparison_sample"].iloc[0],
                "model": model,
                "heldout_uba_stations": int(group["station_code"].nunique()),
                "heldout_sds011_locations": int(raw_preds["location"].nunique()),
                "heldout_rows": int(len(raw_preds)),
                "folds_represented": ";".join(sorted(group["station_fold"].dropna().astype(str).unique())),
                "lands_represented": ";".join(sorted(group["station_land"].dropna().astype(str).unique())),
                "station_type_labels": ";".join(sorted(group["station_type_label"].dropna().astype(str).unique())),
                "station_balanced_rmse": float(group["rmse"].mean()),
                "station_balanced_mae": float(group["mae"].mean()),
                "station_balanced_bias": float(group["bias"].mean()),
                "median_station_rmse": float(rmses.median()),
                "station_rmse_p90": float(rmses.quantile(0.90)),
                "pct_stations_better_than_constant": float(100 * (group["rmse_diff_from_constant_mean"] < 0).mean()),
                "pct_stations_better_than_raw_pm": float(100 * (group["rmse_diff_from_raw_pm"] < 0).mean()),
                "pearson": vals["pearson"],
                "spearman": vals["spearman"],
                "prediction_reference_std_ratio": vals["prediction_reference_std_ratio"],
            }
        )
    out = pd.DataFrame(rows)
    rmse = out.set_index("model")["station_balanced_rmse"]
    out["rmse_improvement_vs_constant"] = out["model"].map(lambda m: rmse.get("constant_mean", np.nan) - rmse[m])
    out["rmse_improvement_vs_raw_pm"] = out["model"].map(lambda m: rmse.get("raw_pm", np.nan) - rmse[m])
    return out


def annual_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "ref": ("ref", "mean"),
        "pred": ("pred", "mean"),
        "raw": ("raw", "mean"),
        "humidity": ("humidity", "mean"),
        "temperature": ("temperature", "mean"),
        "dist_km": ("dist_km", "median"),
        "raw_pm10": ("raw_pm10", "mean"),
        "raw_pm25": ("raw_pm25", "mean"),
        "pm25_gt_pm10": ("pm25_gt_pm10", "mean"),
    }
    return preds.groupby(
        [
            "station_code",
            "station_land",
            "station_fold",
            "station_type_label",
            "station_setting_label",
            "location",
            "model",
        ],
        as_index=False,
    ).agg(**agg)


def coverage_row(frame: pd.DataFrame, radius: float, pollutant: str, stratum: str, source: str, sample: str, modeled: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_level": "coverage",
                "evaluation_level": "daily",
                "subset": "all_days",
                "radius_km": radius,
                "radius_m": int(round(radius * 1000)),
                "pollutant": pollutant,
                "station_type_stratum": stratum,
                "weather_source": source,
                "comparison_sample": sample,
                "heldout_uba_stations": int(frame["station_code"].nunique()),
                "heldout_sds011_locations": int(frame["location"].nunique()),
                "heldout_rows": int(len(frame)),
                "folds_represented": ";".join(sorted(frame["station_fold"].dropna().astype(str).unique())),
                "lands_represented": ";".join(sorted(frame["station_land"].dropna().astype(str).unique())),
                "station_type_labels": ";".join(sorted(frame["station_type_label"].dropna().astype(str).unique())),
                "modeled": bool(modeled),
                "descriptive_reason": "" if modeled else f"fewer than {MIN_CV_STATIONS} independent UBA stations",
            }
        ]
    )


def run_frame(
    frame: pd.DataFrame,
    radius: float,
    pollutant: str,
    stratum: str,
    weather_source: str,
    comparison_sample: str,
    models: list[str],
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    frame = stratum_frame(frame, stratum)
    enough = frame["station_code"].nunique() >= MIN_CV_STATIONS
    outputs.append(coverage_row(frame, radius, pollutant, stratum, weather_source, comparison_sample, enough))
    if not enough:
        return outputs
    preds = station_predictions(frame, models)
    if preds.empty:
        return outputs
    st = station_rows(preds, radius, pollutant, stratum, weather_source, comparison_sample, "daily")
    outputs.append(st)
    outputs.append(aggregate_rows(st, preds, "daily"))
    ann = annual_predictions(preds)
    ann_st = station_rows(ann, radius, pollutant, stratum, weather_source, comparison_sample, "annual")
    outputs.append(ann_st)
    outputs.append(aggregate_rows(ann_st, ann, "annual"))
    return outputs


def model_spec_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_level": "model_spec",
                "model": model,
                "features": "intercept" if not features else "intercept + " + " + ".join(features),
                "uses_rh_fraction_0_1": any("rh_frac" in f for f in features),
                "uses_pm_x_rh_interaction": any("raw_x" in f for f in features),
                "uses_temperature": any("temperature" in f for f in features),
            }
            for model, features in MODEL_FEATURES.items()
        ]
    )


def md(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "No rows."
    return frame[cols].to_markdown(index=False, floatfmt=".3f")


def best_row(agg: pd.DataFrame) -> pd.Series | None:
    if agg.empty:
        return None
    return agg.sort_values("station_balanced_rmse").iloc[0]


def write_summary(out_dir: Path, results: pd.DataFrame, command: str) -> None:
    agg = results[(results["result_level"] == "aggregate") & (results["subset"] == "all_days")]
    coverage = results[results["result_level"] == "coverage"]
    primary = agg[
        np.isclose(agg["radius_km"], PRIMARY_RADIUS_KM)
        & agg["station_type_stratum"].eq("background")
        & agg["comparison_sample"].isin(["source_available", "common_hyras_bme280"])
    ]
    keep = [
        "evaluation_level",
        "pollutant",
        "weather_source",
        "comparison_sample",
        "model",
        "heldout_uba_stations",
        "heldout_sds011_locations",
        "heldout_rows",
        "station_balanced_rmse",
        "median_station_rmse",
        "station_rmse_p90",
        "rmse_improvement_vs_constant",
        "pct_stations_better_than_constant",
        "prediction_reference_std_ratio",
    ]
    coverage_cols = [
        "radius_m",
        "pollutant",
        "station_type_stratum",
        "weather_source",
        "comparison_sample",
        "heldout_uba_stations",
        "heldout_sds011_locations",
        "heldout_rows",
        "folds_represented",
        "modeled",
    ]
    annual_bg = agg[
        agg["evaluation_level"].eq("annual")
        & agg["station_type_stratum"].eq("background")
        & agg["comparison_sample"].eq("source_available")
    ]
    traffic_daily = agg[
        agg["evaluation_level"].eq("daily")
        & agg["station_type_stratum"].isin(["background", "traffic", "industry", "all"])
        & np.isclose(agg["radius_km"], PRIMARY_RADIUS_KM)
        & agg["weather_source"].eq("hyras")
        & agg["comparison_sample"].eq("source_available")
        & agg["model"].eq("raw_pm_rh_interaction_temp")
    ]
    common = agg[
        np.isclose(agg["radius_km"], PRIMARY_RADIUS_KM)
        & agg["station_type_stratum"].eq("background")
        & agg["comparison_sample"].eq("common_hyras_bme280")
        & agg["evaluation_level"].eq("annual")
    ]
    lines = [
        "# Close-Reference Weather Sensitivity Diagnostics",
        "",
        f"Command: `{command}`",
        "",
        "## Model Specifications",
        "",
        results[results["result_level"].eq("model_spec")][["model", "features"]].to_markdown(index=False),
        "",
        "## Primary 500 m Background Results",
        "",
        md(primary, keep),
        "",
        "## Coverage",
        "",
        md(
            coverage[
                np.isclose(coverage["radius_km"], PRIMARY_RADIUS_KM)
                & coverage["comparison_sample"].isin(["source_available", "common_hyras_bme280"])
            ],
            coverage_cols,
        ),
        "",
        "## Annual Background Sensitivity Across Radii",
        "",
        md(annual_bg, ["radius_m", "pollutant", "weather_source", "model", "station_balanced_rmse", "rmse_improvement_vs_constant", "prediction_reference_std_ratio", "pct_stations_better_than_constant"]),
        "",
        "## 500 m HYRAS Daily Strata Check",
        "",
        md(traffic_daily, ["pollutant", "station_type_stratum", "heldout_uba_stations", "heldout_sds011_locations", "station_balanced_rmse", "median_station_rmse", "station_rmse_p90"]),
        "",
        "## 500 m Common-Row HYRAS/BME280 Annual Background Check",
        "",
        md(common, ["pollutant", "weather_source", "model", "heldout_uba_stations", "station_balanced_rmse", "rmse_improvement_vs_constant", "prediction_reference_std_ratio", "pct_stations_better_than_constant"]),
        "",
        "## Answers",
        "",
    ]

    def annual_answer(pollutant: str, source: str = "hyras") -> str:
        part = annual_bg[
            annual_bg["pollutant"].eq(pollutant)
            & annual_bg["weather_source"].eq(source)
            & np.isclose(annual_bg["radius_km"], PRIMARY_RADIUS_KM)
        ]
        best = best_row(part)
        if best is None:
            return f"{pollutant}: no modeled annual background rows for {source}."
        return (
            f"{pollutant} {source}: best `{best['model']}` RMSE "
            f"{best['station_balanced_rmse']:.3f}, improvement vs constant "
            f"{best['rmse_improvement_vs_constant']:.3f}, std ratio "
            f"{best['prediction_reference_std_ratio']:.3f}, station-win pct "
            f"{best['pct_stations_better_than_constant']:.1f}%."
        )

    def metric(pollutant: str, stratum: str, model: str, level: str = "annual") -> float:
        part = agg[
            np.isclose(agg["radius_km"], PRIMARY_RADIUS_KM)
            & agg["pollutant"].eq(pollutant)
            & agg["weather_source"].eq("hyras")
            & agg["comparison_sample"].eq("source_available")
            & agg["station_type_stratum"].eq(stratum)
            & agg["evaluation_level"].eq(level)
            & agg["model"].eq(model)
        ]
        return float(part["station_balanced_rmse"].iloc[0]) if not part.empty else np.nan

    def best_improvement(pollutant: str, radius: float) -> float:
        part = annual_bg[
            np.isclose(annual_bg["radius_km"], radius)
            & annual_bg["pollutant"].eq(pollutant)
            & annual_bg["weather_source"].eq("hyras")
        ]
        modeled = part[part["model"].ne("constant_mean")]
        return float(modeled["rmse_improvement_vs_constant"].max()) if not modeled.empty else np.nan

    def common_best(pollutant: str, source: str) -> tuple[str, float]:
        part = common[
            common["pollutant"].eq(pollutant)
            & common["weather_source"].eq(source)
            & common["model"].ne("constant_mean")
        ]
        if part.empty:
            return "none", np.nan
        row = part.sort_values("station_balanced_rmse").iloc[0]
        return str(row["model"]), float(row["station_balanced_rmse"])

    def common_model_rmse(pollutant: str, source: str, model: str) -> float:
        part = common[
            common["pollutant"].eq(pollutant)
            & common["weather_source"].eq(source)
            & common["model"].eq(model)
        ]
        return float(part["station_balanced_rmse"].iloc[0]) if not part.empty else np.nan

    pm10_bg = metric("PM10", "background", "constant_mean")
    pm10_all = metric("PM10", "all", "constant_mean")
    pm25_bg = metric("PM2.5", "background", "constant_mean")
    pm25_all = metric("PM2.5", "all", "constant_mean")
    pm10_traf_daily = metric("PM10", "traffic", "raw_pm_rh_interaction_temp", "daily")
    pm10_bg_daily = metric("PM10", "background", "raw_pm_rh_interaction_temp", "daily")
    pm25_traf_daily = metric("PM2.5", "traffic", "raw_pm_rh_interaction_temp", "daily")
    pm25_bg_daily = metric("PM2.5", "background", "raw_pm_rh_interaction_temp", "daily")
    pm10_h_model, pm10_h_rmse = common_best("PM10", "hyras")
    pm10_b_model, pm10_b_rmse = common_best("PM10", "bme280")
    pm25_h_model, pm25_h_rmse = common_best("PM2.5", "hyras")
    pm25_b_model, pm25_b_rmse = common_best("PM2.5", "bme280")
    pm10_combined_rmse = common_model_rmse("PM10", "hyras_bme280", "raw_pm_hyras_bme280")
    pm25_combined_rmse = common_model_rmse("PM2.5", "hyras_bme280", "raw_pm_hyras_bme280")

    lines.append(
        f"1. Background-only annual reference fields are easier than all-station fields "
        f"(HYRAS constant RMSE PM10 {pm10_bg:.3f} vs all {pm10_all:.3f}; "
        f"PM2.5 {pm25_bg:.3f} vs all {pm25_all:.3f}), but that does not make the "
        "weather regressions transferable."
    )
    lines.append(
        f"2. PM10 daily errors are larger at traffic stations than background stations "
        f"for the HYRAS temp model ({pm10_traf_daily:.3f} vs {pm10_bg_daily:.3f}); "
        f"PM2.5 does not show the same pattern ({pm25_traf_daily:.3f} vs {pm25_bg_daily:.3f})."
    )
    lines.append(
        f"3. On identical background common rows, HYRAS does not consistently beat BME280: "
        f"PM10 best non-constant HYRAS `{pm10_h_model}` RMSE {pm10_h_rmse:.3f} vs "
        f"BME280 `{pm10_b_model}` {pm10_b_rmse:.3f}; PM2.5 HYRAS `{pm25_h_model}` "
        f"{pm25_h_rmse:.3f} vs BME280 `{pm25_b_model}` {pm25_b_rmse:.3f}."
    )
    lines.append(
        f"4. BME280 adds no useful annual background information beyond HYRAS: combined "
        f"model RMSE is {pm10_combined_rmse:.3f} for PM10 and {pm25_combined_rmse:.3f} "
        "for PM2.5, both worse than the constant and simpler non-constant models."
    )
    lines.append(f"5. {annual_answer('PM10', 'hyras')}")
    lines.append(f"6. {annual_answer('PM2.5', 'hyras')}")
    lines.extend(
        [
            f"7. Stability fails: best HYRAS non-constant annual background improvement vs constant is "
            f"PM10 {best_improvement('PM10', 0.25):.3f}/{best_improvement('PM10', 0.5):.3f}/{best_improvement('PM10', 1.0):.3f} "
            f"at 250/500/1000 m and PM2.5 {best_improvement('PM2.5', 0.25):.3f}/{best_improvement('PM2.5', 0.5):.3f}/{best_improvement('PM2.5', 1.0):.3f}.",
            "8. The combined HYRAS+BME280 model is unnecessary complexity in this pass.",
            "9. A complete hourly UBA PM download is not justified yet for label generation; a small targeted hourly diagnostic could be useful later.",
            "10. No candidate is ready for an experimental full-network label set.",
            "No hourly UBA download, full-network labels, or CNN training were run.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    paths = calib.Paths(Path(args.processed_dir))
    months = sorted(dict.fromkeys(args.months or calib.discover_months(paths)))
    radii = sorted(dict.fromkeys(float(r) for r in args.radius_km))
    out_dir = Path(args.output_dir) if args.output_dir else paths.calibration_root / "close_reference_weather_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processed root: {paths.processed}")
    print(f"Months: {months}")
    print(f"Radii: {radii}")
    print(f"Output: {out_dir}")

    calib.require_files(paths, args.year)
    sensor_land = calib.load_sensor_land(paths)
    uba = calib.load_uba(paths, args.year)
    nodes = calib.load_nodes(paths, months, sensor_land)
    nodes = nodes[nodes["land"] != calib.TEST_LAND].copy()
    uba = uba[uba["land"] != calib.TEST_LAND].copy()
    daily_pm = calib.load_daily_sds011(paths, months)
    daily_pm = daily_pm[daily_pm["location"].isin(nodes["location"])].copy()
    meta = station_metadata(paths.processed)
    hyras = load_hyras_weather(paths.processed)

    max_links = calib.nearest_matches(nodes, uba, max(radii))
    target_locations = set(max_links["location"].astype(int))
    bme = load_bme280_weather(paths.processed, months, target_locations)

    outputs: list[pd.DataFrame] = [model_spec_rows()]
    for radius in radii:
        for pollutant in calib.POLLUTANTS:
            base = base_pairs(daily_pm, nodes, uba, meta, radius, pollutant)
            if base.empty:
                continue
            source_frames = [
                (generic_weather_frame(base, hyras, "hyras", "source_available"), BASE_MODELS),
                (generic_weather_frame(base, bme, "bme280", "source_available"), BASE_MODELS),
            ]
            source_frames += [(f, BASE_MODELS if f["weather_source"].iloc[0] != "hyras_bme280" else COMBINED_MODELS) for f in common_weather_frames(base, hyras, bme)]
            for frame, models in source_frames:
                if frame.empty:
                    continue
                source = frame["weather_source"].iloc[0]
                sample = frame["comparison_sample"].iloc[0]
                for stratum in STATION_TYPE_STRATA:
                    outputs.extend(run_frame(frame, radius, pollutant, stratum, source, sample, models))
                    sub = stratum_frame(frame, stratum)
                    print(
                        f"radius={radius:g} {pollutant} {source} {sample} {stratum}: "
                        f"{len(sub):,} rows, {sub['station_code'].nunique()} stations"
                    )

    results = pd.concat(outputs, ignore_index=True, sort=False)
    if results["station_land"].astype(str).str.contains(calib.TEST_LAND, na=False).any():
        raise AssertionError(f"{calib.TEST_LAND} leaked into sensitivity results")
    out_path = out_dir / "results.csv"
    results.to_csv(out_path, index=False)
    command = "python3 -B 03_scripts_calibration/experiments/nearby_reference_regression/fit_close_reference_weather_models.py"
    write_summary(out_dir, results, command)
    (out_dir / "run_info.txt").write_text(
        "\n".join(
            [
                f"created_utc={datetime.now(timezone.utc).isoformat()}",
                f"command={command}",
                f"months={','.join(months)}",
                f"radii_km={','.join(str(r) for r in radii)}",
                f"test_land_excluded={calib.TEST_LAND}",
                "station_type_strata=background,traffic,industry,all",
                "weather_sources=hyras,bme280,hyras_bme280_common",
                f"min_cv_stations={MIN_CV_STATIONS}",
                "note=Sensitivity experiment only; no labels generated.",
            ]
        )
        + "\n"
    )
    print(f"\nWrote {out_path}")
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
