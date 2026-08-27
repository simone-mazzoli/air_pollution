#!/usr/bin/env python3
"""Build preliminary PM2.5 figures from saved model stations and local inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "06_models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from shared import paths


CV_PREDICTIONS = ROOT / "06_models" / "results" / "cnn_deep_wide" / "eea_cv_predictions.csv"
DAILY_LABELS = paths.LABELS
METADATA = paths.PROC / "eea" / "airbase_raw" / "metadata.csv"
OUT_DATA = ROOT / "analysis_outputs" / "preliminary_analysis"
OUT_FIG = ROOT / "Air_pollution_report" / "Figures" / "generated"
plt = None
report_plot_style = None


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path.relative_to(ROOT)}")


def check_required_inputs() -> None:
    required = [CV_PREDICTIONS, DAILY_LABELS, METADATA, paths.HIGH]
    missing = [path.relative_to(ROOT) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required input(s):\n{joined}")


def init_plotting() -> None:
    global plt, report_plot_style
    import matplotlib.pyplot as pyplot
    import report_plot_style as style

    plt = pyplot
    report_plot_style = style


def first_existing(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise SystemExit(f"Missing one of these columns: {names}")


def load_development_stations() -> pd.DataFrame:
    require_file(CV_PREDICTIONS)
    cv = pd.read_csv(CV_PREDICTIONS, dtype={"station_code": str})
    if "pollutant" in cv.columns:
        cv = cv.loc[cv["pollutant"].eq("pm25")].copy()
    cv = cv.rename(columns={"true": "annual_pm25"})
    cols = ["station_code", "country", "land", "lat", "lon", "fold", "annual_pm25"]
    missing = [c for c in cols if c not in cv.columns]
    if missing:
        raise SystemExit(f"Missing columns in {CV_PREDICTIONS.relative_to(ROOT)}: {missing}")
    return cv[cols].drop_duplicates("station_code")


def load_valid_pm25_days() -> pd.DataFrame:
    require_file(DAILY_LABELS)
    daily = pd.read_csv(DAILY_LABELS, dtype={"station_code": str})
    code = first_existing(daily, ["station_code", "station", "StationCode"])
    pm25 = first_existing(daily, ["PM2.5", "pm25", "PM25"])
    return (
        daily.rename(columns={code: "station_code"})
        .groupby("station_code")[pm25]
        .apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
        .reset_index(name="valid_pm25_days")
    )


def load_station_areas() -> pd.DataFrame:
    require_file(METADATA)
    meta = pd.read_csv(METADATA, dtype=str)
    code = first_existing(
        meta,
        [
            "Air Quality Station EoI Code",
            "station_code",
            "StationCode",
            "AirQualityStationEoICode",
        ],
    )
    area = first_existing(meta, ["Air Quality Station Area"])
    return (
        meta.rename(columns={code: "station_code", area: "station_area"})
        [["station_code", "station_area"]]
        .dropna()
        .drop_duplicates("station_code")
    )


def load_labels() -> pd.DataFrame:
    labels = (
        load_development_stations()
        .merge(load_valid_pm25_days(), on="station_code", how="left")
        .merge(load_station_areas(), on="station_code", how="left")
    )
    missing = labels.loc[labels["valid_pm25_days"].isna() | labels["station_area"].isna()]
    if len(missing):
        codes = ", ".join(missing["station_code"].head(10))
        raise SystemExit(f"Missing PM2.5 day counts or station areas for {len(missing)} stations: {codes}")
    labels["annual_pm25"] = pd.to_numeric(labels["annual_pm25"], errors="coerce")
    labels["valid_pm25_days"] = pd.to_numeric(labels["valid_pm25_days"], errors="coerce")
    labels["station_area"] = labels["station_area"].astype(str).str.strip().str.lower()
    return labels.dropna(subset=["annual_pm25"])


def collapse_area(value: str) -> str | None:
    text = str(value).strip().lower()
    if text == "urban":
        return "Urban"
    if text == "suburban":
        return "Suburban"
    if text.startswith("rural"):
        return "Rural"
    return None


def summarize(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        frame.groupby(group_col)["annual_pm25"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )


def write_urban_rural(labels: pd.DataFrame) -> None:
    detailed = summarize(labels, "station_area").rename(columns={"station_area": "category"})
    detailed.to_csv(OUT_DATA / "urban_rural_pm25_summary_detailed.csv", index=False)

    labels = labels.assign(category=labels["station_area"].map(collapse_area)).dropna(subset=["category"])
    order = ["Urban", "Suburban", "Rural"]
    summary = summarize(labels, "category").set_index("category").reindex(order).reset_index()
    summary.to_csv(OUT_DATA / "urban_rural_pm25_summary_collapsed.csv", index=False)

    report_plot_style.apply()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    data = [labels.loc[labels["category"].eq(group), "annual_pm25"] for group in order]
    ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(123)
    for idx, values in enumerate(data, start=1):
        ax.scatter(rng.normal(idx, 0.035, len(values)), values, s=5, alpha=0.25, linewidth=0)
    ax.set_ylabel("Annual PM2.5 [ug/m3]")
    ax.set_xlabel("")
    fig.tight_layout()
    report_plot_style.savefig(fig, OUT_FIG / "urban_rural_pm25_distribution.png")
    plt.close(fig)


def load_raw_patch(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unexpected patch shape for {path}: {arr.shape}")
    if not np.isfinite(arr).any():
        raise ValueError(f"No finite values in {path}")
    return arr


def rgb_image(array: np.ndarray, low: float, high: float) -> np.ndarray:
    rgb = array[:, :, [2, 1, 0]].astype("float32")
    return np.clip((rgb - low) / (high - low), 0, 1)


def write_patch_grid(selected: pd.DataFrame, patch_dir: Path, out_name: str) -> tuple[float, float]:
    patches = [load_raw_patch(patch_dir / f"{code}.npy") for code in selected["station_code"]]
    rgb_values = np.concatenate([p[:, :, [2, 1, 0]].reshape(-1, 3) for p in patches], axis=0)
    low, high = np.nanpercentile(rgb_values[np.isfinite(rgb_values)], [2, 98])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7))
    for ax, (_, row), patch in zip(axes.ravel(), selected.iterrows(), patches):
        ax.imshow(rgb_image(patch, low, high))
        ax.set_title(f"{row.station_code}: {row.annual_pm25:.3f} ug/m3", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    report_plot_style.savefig(fig, OUT_FIG / out_name)
    plt.close(fig)
    return float(low), float(high)


def write_high_low(labels: pd.DataFrame) -> None:
    dev = labels.loc[labels["country"].astype(str).str.upper().isin(["DE", "DEU", "GERMANY"])].copy()
    dev = dev.loc[dev["valid_pm25_days"] >= 330].copy()

    usable = []
    for _, row in dev.sort_values("annual_pm25").iterrows():
        path = paths.HIGH / f"{row.station_code}.npy"
        try:
            arr = load_raw_patch(path)
        except (FileNotFoundError, ValueError):
            continue
        usable.append(
            {
                **row.to_dict(),
                "high_res_array_path": str(path.relative_to(ROOT)),
                "high_res_shape": "x".join(map(str, arr.shape)),
                "finite_fraction": float(np.isfinite(arr).mean()),
            }
        )

    candidates = pd.DataFrame(usable).sort_values("annual_pm25")
    if candidates.empty:
        raise SystemExit("No eligible German development stations with usable high-resolution patches.")

    selected = pd.concat(
        [
            candidates.head(3).assign(example_group="low"),
            candidates.tail(3).sort_values("annual_pm25", ascending=False).assign(example_group="high"),
        ],
        ignore_index=True,
    )
    selected.to_csv(OUT_DATA / "high_low_pm25_selected_stations.csv", index=False)
    selected[
        [
            "example_group",
            "station_code",
            "land",
            "station_area",
            "annual_pm25",
            "valid_pm25_days",
            "lat",
            "lon",
            "high_res_array_path",
            "high_res_shape",
            "finite_fraction",
        ]
    ].to_csv(OUT_DATA / "high_low_pm25_patch_quality_checks.csv", index=False)

    scaling = []
    low, high = write_patch_grid(selected, paths.HIGH, "high_low_pm25_patch_examples.png")
    scaling.append(
        {
            "figure": "high_res",
            "lower_percentile": 2,
            "upper_percentile": 98,
            "lower_value": low,
            "upper_value": high,
            "rgb_indices": "2,1,0",
        }
    )
    if paths.LOW.exists():
        low, high = write_patch_grid(selected, paths.LOW, "high_low_pm25_patch_examples_lowres.png")
        scaling.append(
            {
                "figure": "low_res",
                "lower_percentile": 2,
                "upper_percentile": 98,
                "lower_value": low,
                "upper_value": high,
                "rgb_indices": "2,1,0",
            }
        )
    pd.DataFrame(scaling).to_csv(OUT_DATA / "high_low_pm25_image_scaling.csv", index=False)


def main() -> None:
    check_required_inputs()
    init_plotting()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    write_urban_rural(labels)
    write_high_low(labels)
    print(f"wrote {OUT_DATA.relative_to(ROOT)}")
    print(f"wrote preliminary figures to {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
