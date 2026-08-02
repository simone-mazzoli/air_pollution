#!/usr/bin/env python3
"""Build the canonical Sensor.Community calibration summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_scripts_calibration" / "sensor_community_calibration_summary.csv"
METHOD_AGG = ROOT / "data/processed/calibration/regression_reference_adjustment/method_comparison/aggregate_metrics.csv"
SIGNAL = ROOT / "data/processed/calibration/regression_reference_adjustment/method_comparison/signal_preservation_metrics.csv"
WEATHER = ROOT / "data/processed/calibration/regression_reference_adjustment/close_reference_weather_models/results.csv"
CLUSTER = ROOT / "03_scripts_calibration/experiments/clustered_sensors/cluster_test_results.csv"

FIELDS = [
    "approach",
    "pollutant",
    "temporal_level",
    "matching_or_aggregation",
    "evaluation_target",
    "headline_metric",
    "headline_value",
    "comparison_baseline",
    "beats_baseline",
    "spatial_signal_result",
    "main_limitation",
    "final_role",
    "source_file",
]

METHOD_ROLE = {
    "raw SDS011 baseline": (
        "Raw SDS011 annual values have weak rank signal but very large error and malfunction tails.",
        "not reference-grade; comparison only",
    ),
    "constant-mean baseline": (
        "Lowest annual RMSE because it removes spatial variation.",
        "diagnostic baseline only",
    ),
    "original percentile-mapping method": (
        "Plausible national annual distribution, but not paired local calibration.",
        "current proxy label candidate only; not reference-grade",
    ),
    "OLS regression adjustment": (
        "Greatly reduces raw error but collapses most annual spatial variation.",
        "archived sensitivity labels; not main calibration",
    ),
    "Huber regression adjustment": (
        "Greatly reduces raw error but collapses most annual spatial variation.",
        "kept for report comparison",
    ),
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing expected summary input: {path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def find_one(rows: list[dict[str, str]], **match: object) -> dict[str, str]:
    found = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in match.items())
    ]
    if len(found) != 1:
        raise ValueError(f"expected one row for {match}, found {len(found)}")
    return found[0]


def signal_text(rows: list[dict[str, str]], method: str, pollutant: str) -> str:
    found = [
        row
        for row in rows
        if row["method"] == method and row["pollutant"] == pollutant and row["radius_km"] == "20.0"
    ]
    if not found:
        raise ValueError(f"expected signal rows for {method} {pollutant} at 20 km")

    def mean(column: str) -> str:
        vals = [float(row[column]) for row in found if row[column] != ""]
        return "" if not vals else f"{sum(vals) / len(vals):.3f}"

    spearman = mean("spearman_with_reference")
    ratio = mean("prediction_reference_std_ratio")
    if spearman == "":
        spearman = "undefined"
    return f"Spearman {spearman}; prediction/reference std ratio {ratio}"


def method_rows() -> list[dict[str, str]]:
    agg = read_rows(METHOD_AGG)
    signal = read_rows(SIGNAL)
    out = []
    source = f"{rel(METHOD_AGG)}; {rel(SIGNAL)}"
    methods = [
        "raw SDS011 baseline",
        "constant-mean baseline",
        "original percentile-mapping method",
        "OLS regression adjustment",
        "Huber regression adjustment",
    ]
    for method in methods:
        for pollutant in ("PM10", "PM2.5"):
            row = find_one(agg, method=method, pollutant=pollutant, radius_km="20.0")
            limitation, role = METHOD_ROLE[method]
            beats = ""
            if method != "constant-mean baseline":
                beats = "yes" if int(row["folds_better_than_constant_mean"]) == int(row["n_folds"]) else "no"
            out.append(
                {
                    "approach": (
                        "national percentile/range mapping"
                        if method == "original percentile-mapping method"
                        else method
                    ),
                    "pollutant": pollutant,
                    "temporal_level": "annual",
                    "matching_or_aggregation": "nearest SDS011-UBA pairs within 20 km; non-Sachsen-Anhalt LOO folds",
                    "evaluation_target": "held-out UBA annual station-balanced RMSE",
                    "headline_metric": "mean RMSE",
                    "headline_value": fnum(row["rmse_mean"]),
                    "comparison_baseline": "constant-mean baseline",
                    "beats_baseline": beats,
                    "spatial_signal_result": signal_text(signal, method, pollutant),
                    "main_limitation": limitation,
                    "final_role": role,
                    "source_file": source,
                }
            )
    return out


def weather_rows() -> list[dict[str, str]]:
    rows = read_rows(WEATHER)
    out = []
    daily_choices = {
        "PM10": "raw_pm_rh_interaction_temp",
        "PM2.5": "raw_pm_rh_interaction_temp",
    }
    annual_choices = {
        "PM10": "raw_pm_rh",
        "PM2.5": "raw_pm",
    }
    for pollutant, model in daily_choices.items():
        row = find_one(
            rows,
            result_level="aggregate",
            evaluation_level="daily",
            pollutant=pollutant,
            station_type_stratum="background",
            weather_source="hyras",
            comparison_sample="source_available",
            radius_km="0.25",
            model=model,
        )
        out.append(
            {
                "approach": "daily weather-aware regression",
                "pollutant": pollutant,
                "temporal_level": "daily",
                "matching_or_aggregation": "background UBA stations, HYRAS weather, SDS011 within 250 m",
                "evaluation_target": "held-out UBA daily station-balanced RMSE",
                "headline_metric": "RMSE improvement vs constant",
                "headline_value": fnum(row["rmse_improvement_vs_constant"]),
                "comparison_baseline": "constant-mean baseline",
                "beats_baseline": "yes" if float(row["rmse_improvement_vs_constant"]) > 0 else "no",
                "spatial_signal_result": f"prediction/reference std ratio {fnum(row['prediction_reference_std_ratio'])}",
                "main_limitation": "Some daily improvements in small close-reference subsets did not translate into convincing annual calibration.",
                "final_role": "diagnostic only; no annual Sensor.Community labels accepted",
                "source_file": rel(WEATHER),
            }
        )
    for pollutant, model in annual_choices.items():
        row = find_one(
            rows,
            result_level="aggregate",
            evaluation_level="annual",
            pollutant=pollutant,
            station_type_stratum="background",
            weather_source="hyras",
            comparison_sample="source_available",
            radius_km="0.5",
            model=model,
        )
        out.append(
            {
                "approach": "background-station-only sensitivity",
                "pollutant": pollutant,
                "temporal_level": "annual",
                "matching_or_aggregation": "background UBA stations, HYRAS weather, SDS011 within 500 m",
                "evaluation_target": "held-out UBA annual station-balanced RMSE",
                "headline_metric": "RMSE improvement vs constant",
                "headline_value": fnum(row["rmse_improvement_vs_constant"]),
                "comparison_baseline": "constant-mean baseline",
                "beats_baseline": "yes" if float(row["rmse_improvement_vs_constant"]) > 0 else "no",
                "spatial_signal_result": f"prediction/reference std ratio {fnum(row['prediction_reference_std_ratio'])}",
                "main_limitation": "Restricting to background stations did not rescue annual reference-grade calibration.",
                "final_role": "archived sensitivity only",
                "source_file": rel(WEATHER),
            }
        )
    return out


def cluster_rows() -> list[dict[str, str]]:
    rows = read_rows(CLUSTER)
    out = []
    for agg in ("mean", "median"):
        best = min((row for row in rows if row["agg"] == agg), key=lambda row: float(row["rmse"]))
        out.append(
            {
                "approach": f"clustered-sensor {agg}",
                "pollutant": "PM10",
                "temporal_level": "annual",
                "matching_or_aggregation": f"{best['k']} sensors per cluster, {agg} aggregation",
                "evaluation_target": "co-located UBA cluster diagnostic",
                "headline_metric": "RMSE",
                "headline_value": fnum(best["rmse"]),
                "comparison_baseline": "constant-mean-like skill score",
                "beats_baseline": "yes" if float(best["skill"]) > 0 else "no",
                "spatial_signal_result": f"correlation {fnum(best['corr'])}; skill {fnum(best['skill'])}",
                "main_limitation": "Aggregation reduces random device noise but current cluster diagnostics still do not beat the baseline.",
                "final_role": "diagnostic only; not a reference-station substitute",
                "source_file": rel(CLUSTER),
            }
        )
    return out


def main() -> None:
    rows = method_rows() + weather_rows() + cluster_rows()
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
