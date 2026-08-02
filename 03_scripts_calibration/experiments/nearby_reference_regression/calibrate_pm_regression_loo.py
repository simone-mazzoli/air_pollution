#!/usr/bin/env python3
"""
The script prepares corrected annual PM labels for later CNN training and
evaluation. It is intentionally isolated from the
existing percentile/range-mapping calibration so the two calibration branches
can be compared without sharing output files.

For each pollutant it fits one affine relationship on nearby daily sensor and
UBA reference pairs:

    corrected PM = intercept + slope * raw SDS011 PM

OLS and Huber use the same paired rows; they differ only in how the intercept
and slope are estimated. The fitted equations are global for the selected
training set, not per-sensor. These are nearby sensor-station matches within a
configured radius, not genuinely co-located instruments.

Sachsen-Anhalt is treated as the sealed final CNN test Land. Inner validation
folds never fit on Sachsen-Anhalt sensors/stations, and they also exclude the
current validation fold's sensors and UBA stations from that fold's fit. After
method/radius selection, final PM10 and PM2.5 equations are fit on all
non-Sachsen-Anhalt data and the exact same coefficients are applied to both
non-Sachsen-Anhalt CNN training labels and Sachsen-Anhalt test labels.

A known limitation: the 2024 global affine fits have small slopes, so corrected
annual labels are substantially less variable than raw SDS011 annual means. The
script reports this behavior in diagnostics but does not add an unvalidated
variance-restoration step.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Match the existing calibration split exactly so regression and percentile
# mapping can be compared without changing the evaluation geography.
TEST_LAND = "Sachsen-Anhalt"
UTC_TO_MEZ_HOURS = 1
MIN_HOURS_PER_DAY = 18
MIN_DAYS_PER_YEAR = 182
MAX_ANNUAL_UGM3 = 50.0
PRIMARY_SENSOR_TYPE = "sds011"
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

# Low-cost column names come from Sensor.Community; reference names come from
# the UBA daily export after the gathering step pivots it to wide form.
POLLUTANTS = {
    "PM10": {"lowcost": "P1", "ref": "PM10"},
    "PM2.5": {"lowcost": "P2", "ref": "PM2.5"},
}

LAND_TO_FOLD = {
    "Berlin": "Berlin-Brandenburg",
    "Brandenburg": "Berlin-Brandenburg",
    "Bremen": "Bremen-Niedersachsen",
    "Niedersachsen": "Bremen-Niedersachsen",
    "Hamburg": "Hamburg-Schleswig-Holstein",
    "Schleswig-Holstein": "Hamburg-Schleswig-Holstein",
    "Saarland": "Saarland-Rheinland-Pfalz",
    "Rheinland-Pfalz": "Saarland-Rheinland-Pfalz",
    "Baden-Wuerttemberg": "Baden-Wuerttemberg",
    "Bayern": "Bayern",
    "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen",
    "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt",
    "Thueringen": "Thueringen",
}


@dataclass(frozen=True)
class Paths:
    """Centralized paths for this isolated alternative calibration tree."""

    processed: Path

    @property
    def pm_hourly_dir(self) -> Path:
        """Merged hourly PM files produced by the existing cleaning stage."""

        return self.processed / "hourly" / "pm" / "all_pm_sensors"

    @property
    def pm_nodes_dir(self) -> Path:
        """Per-sensor-type node coordinate files from PM cleaning."""

        return self.processed / "hourly" / "pm" / "nodes"

    @property
    def uba_daily_template(self) -> Path:
        """UBA daily PM reference file template for a given year."""

        return self.processed / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"

    @property
    def station_land(self) -> Path:
        """UBA station-to-Land lookup used to assign reference folds."""

        return self.processed / "uba" / "station_land.csv"

    @property
    def sensor_land(self) -> Path:
        """Low-cost sensor-to-Land lookup used to assign sensor folds."""

        return self.processed / "sensor_land.csv"

    @property
    def calibration_root(self) -> Path:
        """Output root for regression coefficients and validation summaries."""

        return self.processed / "calibration" / "regression_reference_adjustment"

    @property
    def corrected_root(self) -> Path:
        """Output root for regression-adjusted annual label CSVs."""

        return self.processed / "corrected" / "regression_reference_adjustment"


def radius_label(radius_km: float) -> str:
    """Return a filesystem-safe radius token such as ``10`` or ``7p5``."""

    return f"{radius_km:g}".replace(".", "p")


def fold_slug(fold: str) -> str:
    """Return a fold name that is safe to use as a path segment."""

    return fold.replace(" ", "_")


def discover_months(paths: Paths) -> list[str]:
    """List real merged hourly PM month files, ignoring sidecars and junk files."""

    if not paths.pm_hourly_dir.exists():
        return []
    return sorted(
        p.stem
        for p in paths.pm_hourly_dir.glob("*.parquet")
        if MONTH_RE.match(p.stem)
    )


def require_files(paths: Paths, year: int) -> None:
    """Fail early with a readable message when required lookup inputs are absent."""

    required = [
        paths.sensor_land,
        paths.station_land,
        Path(str(paths.uba_daily_template).format(year=year)),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Required calibration input(s) are missing:\n  "
            + "\n  ".join(missing)
            + "\nGenerate/copy these files before running the regression calibration."
        )


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in kilometers for scalar or array inputs."""

    r = 6371.0
    p = np.radians
    dp = p(lat2) - p(lat1)
    dl = p(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_sensor_land(paths: Paths) -> pd.DataFrame:
    """Load low-cost sensor Land assignments and attach the existing fold labels."""

    df = pd.read_csv(paths.sensor_land)
    required = {"location", "land"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{paths.sensor_land} is missing column(s): {sorted(missing)}")
    df = df[["location", "land"]].drop_duplicates("location").copy()
    df["fold"] = df["land"].map(LAND_TO_FOLD)
    bad = df[df["fold"].isna()]["land"].dropna().unique()
    if len(bad):
        raise ValueError(f"Unmapped Land values in sensor_land.csv: {sorted(bad)}")
    return df


def load_uba(paths: Paths, year: int) -> pd.DataFrame:
    """Load UBA daily reference measurements with station Land/fold metadata."""

    uba_path = Path(str(paths.uba_daily_template).format(year=year))
    df = pd.read_csv(uba_path)
    required = {"station_code", "lat", "lon", "Datum", "PM10", "PM2.5"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{uba_path} is missing column(s): {sorted(missing)}")

    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for col in ("PM10", "PM2.5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    station_land = pd.read_csv(paths.station_land)
    required_land = {"station_code", "land"}
    missing_land = required_land - set(station_land.columns)
    if missing_land:
        raise KeyError(
            f"{paths.station_land} is missing column(s): {sorted(missing_land)}"
        )
    station_land = station_land[["station_code", "land"]].drop_duplicates("station_code")

    df = df.merge(station_land, on="station_code", how="left")
    df["fold"] = df["land"].map(LAND_TO_FOLD)
    return df.dropna(subset=["fold"])[
        ["station_code", "lat", "lon", "date", "PM10", "PM2.5", "land", "fold"]
    ]


def load_nodes(paths: Paths, months: list[str], sensor_land: pd.DataFrame) -> pd.DataFrame:
    """Load SDS011 node coordinates and keep sensors with known Land assignments.

    Coordinates can jitter month to month, so the median lat/lon across requested
    months is used, matching the spirit of the upstream cleaning script.
    """

    parts = []
    for month in months:
        path = paths.pm_nodes_dir / f"{PRIMARY_SENSOR_TYPE}_{month}.parquet"
        if not path.exists():
            print(f"WARNING: missing SDS011 node file for {month}: {path}")
            continue
        part = pd.read_parquet(path, columns=["location", "lat", "lon"])
        part["month"] = month
        parts.append(part)
    if not parts:
        raise FileNotFoundError(
            f"No {PRIMARY_SENSOR_TYPE} node files found for requested months."
        )
    nodes = pd.concat(parts, ignore_index=True)
    nodes = (
        nodes.groupby("location", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"))
    )
    nodes = nodes.merge(sensor_land, on="location", how="inner")
    return nodes


def load_daily_sds011(paths: Paths, months: list[str]) -> pd.DataFrame:
    """Load merged hourly PM data, filter to SDS011, and rebuild daily means.

    The upstream cleaner has already converted raw readings into valid hourly
    means. This function counts those hourly rows, applies the requested
    SDS011-only policy before any aggregation, shifts UTC hours to fixed MEZ
    dates, and keeps sensor-days with at least ``MIN_HOURS_PER_DAY`` hours.
    """

    daily_parts = []
    total_rows_before = 0
    total_rows_after = 0
    total_locations_before: set = set()
    total_locations_after: set = set()
    excluded_counts = {}
    loaded_months = []

    columns = ["location", "hour", "P1", "P2", "sensor_type"]
    for month in months:
        path = paths.pm_hourly_dir / f"{month}.parquet"
        if not path.exists():
            print(f"WARNING: missing merged hourly PM file for {month}: {path}")
            continue

        h = pd.read_parquet(path, columns=columns)
        loaded_months.append(month)
        before_rows = len(h)
        before_locations = h["location"].nunique()
        total_rows_before += before_rows
        total_locations_before.update(h["location"].dropna().unique())

        # Normalize labels before filtering so " SDS011 " and "sds011" are
        # treated identically, without changing the upstream merged files.
        sensor_type = h["sensor_type"].astype(str).str.strip().str.lower()
        type_counts = sensor_type.value_counts(dropna=False)
        for sensor_type_name, count in type_counts.items():
            if sensor_type_name != PRIMARY_SENSOR_TYPE:
                excluded_counts[sensor_type_name] = (
                    excluded_counts.get(sensor_type_name, 0) + int(count)
                )

        h = h[sensor_type == PRIMARY_SENSOR_TYPE].copy()
        after_rows = len(h)
        after_locations = h["location"].nunique()
        total_rows_after += after_rows
        total_locations_after.update(h["location"].dropna().unique())

        h["hour"] = pd.to_datetime(h["hour"], errors="coerce")
        h = h.dropna(subset=["hour"])
        h["date"] = (h["hour"] + pd.Timedelta(hours=UTC_TO_MEZ_HOURS)).dt.date
        for col in ("P1", "P2"):
            h[col] = pd.to_numeric(h[col], errors="coerce")

        daily = (
            h.groupby(["location", "date"], as_index=False)
            .agg(P1=("P1", "mean"), P2=("P2", "mean"), n_hours=("hour", "nunique"))
        )
        daily = daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()
        daily_parts.append(daily)

        print(
            f"{month}: {before_rows:,} hourly rows / {before_locations:,} locations "
            f"-> {after_rows:,} SDS011 rows / {after_locations:,} locations"
        )

    if not daily_parts:
        raise FileNotFoundError("No merged hourly PM data loaded for requested months.")

    print("\nSDS011 filtering summary:")
    print(f"  months loaded: {loaded_months}")
    print(f"  rows before filter: {total_rows_before:,}")
    print(f"  rows after filter:  {total_rows_after:,}")
    print(f"  locations before filter: {len(total_locations_before):,}")
    print(f"  locations after filter:  {len(total_locations_after):,}")
    print(f"  excluded sensor-type counts: {excluded_counts or '{}'}")
    print(
        "  hourly rows are treated as valid hourly means from the upstream cleaning "
        "script; daily means are reconstructed here with an 18-hour threshold."
    )

    # The same MEZ date can receive rows from adjacent UTC months at boundaries.
    # Recombine those fragments as hour-weighted daily means, not simple means.
    daily = pd.concat(daily_parts, ignore_index=True)
    daily["P1_hours"] = daily["P1"] * daily["n_hours"]
    daily["P2_hours"] = daily["P2"] * daily["n_hours"]
    daily = (
        daily.groupby(["location", "date"], as_index=False)
        .agg(
            P1_hours=("P1_hours", "sum"),
            P2_hours=("P2_hours", "sum"),
            n_hours=("n_hours", "sum"),
        )
    )
    daily["P1"] = daily["P1_hours"] / daily["n_hours"]
    daily["P2"] = daily["P2_hours"] / daily["n_hours"]
    daily = daily.drop(columns=["P1_hours", "P2_hours"])
    return daily


def nearest_matches(nodes: pd.DataFrame, stations: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """Match each low-cost sensor to its nearest UBA station within the radius."""

    st = stations.drop_duplicates("station_code")[
        ["station_code", "lat", "lon", "fold"]
    ].reset_index(drop=True)
    if nodes.empty or st.empty:
        return pd.DataFrame(
            columns=["location", "station_code", "dist_km", "station_fold"]
        )
    d = haversine_km(
        nodes["lat"].to_numpy()[:, None],
        nodes["lon"].to_numpy()[:, None],
        st["lat"].to_numpy()[None, :],
        st["lon"].to_numpy()[None, :],
    )
    best = d.argmin(axis=1)
    dist = d[np.arange(len(nodes)), best]
    out = pd.DataFrame(
        {
            "location": nodes["location"].to_numpy(),
            "station_code": st["station_code"].to_numpy()[best],
            "dist_km": dist,
            "station_fold": st["fold"].to_numpy()[best],
        }
    )
    return out[out["dist_km"] <= radius_km].reset_index(drop=True)


def make_pairs(
    daily: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    radius_km: float,
    pollutant: str,
) -> pd.DataFrame:
    """Build paired daily low-cost/reference rows for one pollutant and radius.

    The caller controls which sensor folds and station folds are eligible by
    passing already-filtered ``daily``, ``nodes``, and ``uba`` frames.
    """

    spec = POLLUTANTS[pollutant]
    links = nearest_matches(nodes, uba, radius_km)
    ref = uba[["station_code", "date", spec["ref"]]].rename(
        columns={spec["ref"]: "ref"}
    )
    out = (
        daily[["location", "date", spec["lowcost"]]]
        .rename(columns={spec["lowcost"]: "raw"})
        .merge(
            nodes[["location", "fold"]].rename(columns={"fold": "sensor_fold"}),
            on="location",
        )
        .merge(links, on="location")
        .merge(ref, on=["station_code", "date"])
        .dropna(subset=["raw", "ref"])
    )
    return out[(out["raw"] > 0) & np.isfinite(out["raw"]) & np.isfinite(out["ref"])].copy()


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ordinary least squares and return ``(intercept, slope)``."""

    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def fit_huber(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 1.35,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Fit a two-parameter Huber regression with iteratively reweighted LS.

    This avoids adding a scikit-learn dependency while still down-weighting
    unusually large residuals. The tuning constant is applied in robust-scale
    units using the median absolute deviation.
    """

    design = np.column_stack([np.ones(len(x)), x])
    beta = np.array(fit_ols(x, y), dtype="float64")
    for _ in range(max_iter):
        resid = y - design @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(scale) or scale < 1e-12:
            break
        cutoff = delta * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(resid), 1e-12))
        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        new_beta = np.linalg.lstsq(weighted_design, y * sqrt_weights, rcond=None)[0]
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    return float(beta[0]), float(beta[1])


def fit_method(pairs: pd.DataFrame, method: str) -> tuple[float, float] | None:
    """Fit one requested regression method on paired daily rows."""

    x = pairs["raw"].to_numpy(dtype="float64")
    y = pairs["ref"].to_numpy(dtype="float64")
    if len(x) < 2 or np.nanstd(x) < 1e-12:
        return None
    if method == "ols":
        return fit_ols(x, y)
    if method == "huber":
        return fit_huber(x, y)
    raise ValueError(f"unknown method: {method}")


def weighted_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    """Compute station-balanced annual metrics for one prediction column.

    Each station contributes equal total weight, regardless of how many nearby
    low-cost sensors are matched to it.
    """

    if frame.empty:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    counts = frame.groupby("station_code")["location"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype="float64")
    target = frame["ref"].to_numpy(dtype="float64")
    pred = frame[pred_col].to_numpy(dtype="float64")
    err = pred - target
    wsum = weights.sum()
    mae = np.sum(weights * np.abs(err)) / wsum
    rmse = np.sqrt(np.sum(weights * err**2) / wsum)
    bias = np.sum(weights * err) / wsum
    target_mean = np.sum(weights * target) / wsum
    sse = np.sum(weights * err**2)
    sst = np.sum(weights * (target - target_mean) ** 2)
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "r2": float(r2),
    }


def evaluate_annual(
    pairs: pd.DataFrame, intercept: float, slope: float
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate raw and corrected annual values on held-out matched sensors."""

    annual = (
        pairs.groupby(["station_code", "location"], as_index=False)
        .agg(raw=("raw", "mean"), ref=("ref", "mean"), dist_km=("dist_km", "first"))
    )
    annual["corrected"] = np.maximum(0.0, intercept + slope * annual["raw"].to_numpy())
    raw = weighted_metrics(annual, "raw")
    corrected = weighted_metrics(annual, "corrected")
    metrics = {
        "raw_mae": raw["mae"],
        "corrected_mae": corrected["mae"],
        "raw_rmse": raw["rmse"],
        "corrected_rmse": corrected["rmse"],
        "raw_bias": raw["bias"],
        "corrected_bias": corrected["bias"],
        "raw_r2": raw["r2"],
        "corrected_r2": corrected["r2"],
    }
    return metrics, annual


def distance_stats(pairs: pd.DataFrame) -> dict[str, float]:
    """Summarize unique sensor-to-station matching distances."""

    unique_matches = pairs[["location", "station_code", "dist_km"]].drop_duplicates()
    if unique_matches.empty:
        return {
            "median_distance_km": np.nan,
            "p90_distance_km": np.nan,
            "max_distance_km": np.nan,
        }
    return {
        "median_distance_km": float(unique_matches["dist_km"].median()),
        "p90_distance_km": float(unique_matches["dist_km"].quantile(0.9)),
        "max_distance_km": float(unique_matches["dist_km"].max()),
    }


def apply_coefficients_to_annual(
    daily: pd.DataFrame,
    nodes: pd.DataFrame,
    coeffs: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply pollutant coefficients to daily rows and aggregate annual labels."""

    df = daily.copy()
    for pollutant, spec in POLLUTANTS.items():
        df[f"{pollutant}_raw"] = df[spec["lowcost"]]
        if pollutant in coeffs:
            intercept, slope = coeffs[pollutant]
            df[f"{pollutant}_corrected"] = np.maximum(
                0.0, intercept + slope * df[spec["lowcost"]].to_numpy()
            )

    value_cols = [
        c for c in df.columns if c.endswith("_raw") or c.endswith("_corrected")
    ]
    grouped = df.groupby("location")
    annual = grouped[value_cols].mean().reset_index()
    annual["n_days_total"] = grouped["date"].nunique().to_numpy()
    annual = annual.merge(
        nodes[["location", "lat", "lon", "land", "fold"]],
        on="location",
        how="left",
    )
    annual = annual[annual["n_days_total"] >= MIN_DAYS_PER_YEAR].copy()

    # Preserve the existing annual malfunction sanity filter, but do not cap
    # normal corrected values below this threshold.
    corrected_cols = [c for c in annual.columns if c.endswith("_corrected")]
    if corrected_cols:
        keep = (annual[corrected_cols] <= MAX_ANNUAL_UGM3).all(axis=1)
        annual = annual[keep].copy()
    return annual


def write_json(path: Path, obj) -> None:
    """Write deterministic pretty JSON, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def final_selection_table(cv: pd.DataFrame) -> pd.DataFrame:
    """Rank shared method/radius candidates using only inner-fold metrics."""

    choices = cv.dropna(subset=["corrected_rmse", "corrected_mae"]).copy()
    ranked = (
        choices.groupby(["method", "radius_km"], as_index=False)
        .agg(
            mean_corrected_rmse=("corrected_rmse", "mean"),
            mean_corrected_mae=("corrected_mae", "mean"),
            mean_raw_rmse=("raw_rmse", "mean"),
            mean_raw_mae=("raw_mae", "mean"),
            fold_pollutant_rows=("corrected_rmse", "size"),
        )
        .sort_values(["mean_corrected_rmse", "mean_corrected_mae", "method", "radius_km"])
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[
        [
            "rank",
            "method",
            "radius_km",
            "mean_corrected_rmse",
            "mean_corrected_mae",
            "mean_raw_rmse",
            "mean_raw_mae",
            "fold_pollutant_rows",
        ]
    ]


def write_run_metadata(
    paths: Paths,
    *,
    year: int,
    months: list[str],
    methods: list[str],
    radii: list[float],
    final_method: str,
    final_radius: float,
    cv: pd.DataFrame,
) -> None:
    """Write a small reproducibility record for the calibration run."""

    non_test_folds = sorted(
        f for f in cv["fold"].dropna().unique().tolist() if f != TEST_LAND
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "processed_dir": str(paths.processed),
        "year": int(year),
        "source_months": months,
        "sensor_filter": PRIMARY_SENSOR_TYPE,
        "daily_thresholds": {"min_hours_per_day": MIN_HOURS_PER_DAY},
        "annual_thresholds": {
            "min_days_per_year": MIN_DAYS_PER_YEAR,
            "max_corrected_ugm3": MAX_ANNUAL_UGM3,
        },
        "methods_evaluated": methods,
        "radii_km_evaluated": radii,
        "selection_rule": (
            "Lowest mean inner-fold corrected RMSE across pollutants, with "
            "mean corrected MAE as tie-breaker."
        ),
        "selected_method": final_method,
        "selected_radius_km": final_radius,
        "sealed_test_land": TEST_LAND,
        "inner_validation_folds": non_test_folds,
        "leakage_assertions": {
            "sachsen_anhalt_excluded_from_inner_fits": True,
            "validation_fold_sensors_excluded_from_that_fold_fit": True,
            "validation_fold_uba_stations_excluded_from_that_fold_fit": True,
            "final_train_and_test_receive_identical_coefficients": True,
        },
        "output_roots": {
            "calibration": str(paths.calibration_root),
            "corrected_labels": str(paths.corrected_root),
        },
    }
    write_json(paths.calibration_root / "run_metadata.json", metadata)


def write_annual(
    paths: Paths,
    kind: str,
    radius_km: float,
    method: str,
    year: int,
    annual: pd.DataFrame,
    fold: str | None = None,
) -> Path:
    """Write an annual label CSV into the isolated regression output tree."""

    rlabel = f"radius_{radius_label(radius_km)}km"
    if kind == "fold":
        assert fold is not None
        out = (
            paths.corrected_root
            / "folds"
            / rlabel
            / fold_slug(fold)
            / method
            / "annual"
            / f"{year}.csv"
        )
    elif kind == "train":
        out = (
            paths.corrected_root
            / "final"
            / rlabel
            / "train"
            / method
            / "annual"
            / f"{year}.csv"
        )
    elif kind == "test":
        out = (
            paths.corrected_root
            / "final"
            / rlabel
            / f"test_{fold_slug(TEST_LAND)}"
            / method
            / "annual"
            / f"{year}.csv"
        )
    else:
        raise ValueError(kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(out, index=False)
    return out


def methods_from_arg(method: str) -> list[str]:
    """Expand the CLI ``both`` sentinel into concrete method names."""

    return ["ols", "huber"] if method == "both" else [method]


def guard_outputs(paths: Paths, refit: bool) -> None:
    """Avoid overwriting previous alternative calibration outputs by default."""

    cv_path = paths.calibration_root / "cv_results.csv"
    if cv_path.exists() and not refit:
        raise SystemExit(
            f"{cv_path} already exists. Re-run with --refit to overwrite this "
            "alternative calibration output."
        )


def run_cross_validation(
    paths: Paths,
    daily: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    methods: list[str],
    radii: list[float],
    year: int,
) -> tuple[pd.DataFrame, dict]:
    """Run non-test leave-one-fold-out validation for all methods and radii.

    For fold ``F``, both sensors and UBA stations in ``F`` are excluded from
    fitting, as are all Sachsen-Anhalt sensors/stations. Held-out metrics are
    computed only from fold ``F`` pairs.
    """

    non_test_nodes = nodes[nodes["land"] != TEST_LAND].copy()
    non_test_daily = daily[daily["location"].isin(non_test_nodes["location"])].copy()
    non_test_uba = uba[uba["land"] != TEST_LAND].copy()
    folds = sorted(f for f in non_test_uba["fold"].dropna().unique() if f != TEST_LAND)

    rows = []
    coeffs_by_fold = {}
    print(f"\n{len(folds)} inner validation folds ({TEST_LAND} sealed out): {folds}")

    for radius in radii:
        coeffs_by_fold[str(radius)] = {}
        for fold in folds:
            # Strict fold split: validation fold sensors/stations are not allowed
            # to influence the fitted intercept or slope.
            train_nodes = non_test_nodes[non_test_nodes["fold"] != fold].copy()
            val_nodes = non_test_nodes[non_test_nodes["fold"] == fold].copy()
            train_daily = non_test_daily[non_test_daily["location"].isin(train_nodes["location"])]
            val_daily = non_test_daily[non_test_daily["location"].isin(val_nodes["location"])]
            train_uba = non_test_uba[non_test_uba["fold"] != fold].copy()
            val_uba = non_test_uba[non_test_uba["fold"] == fold].copy()

            coeffs_by_fold[str(radius)][fold] = {}
            print(f"\n=== radius={radius:g} km fold={fold} ===")
            for pollutant in POLLUTANTS:
                train_pairs = make_pairs(train_daily, train_nodes, train_uba, radius, pollutant)
                val_pairs = make_pairs(val_daily, val_nodes, val_uba, radius, pollutant)

                # Defensive leakage check. This should be redundant with the
                # filtered inputs above, but the assertion makes split mistakes loud.
                bad_train_sensor = set(train_pairs["sensor_fold"].unique()) & {fold, TEST_LAND}
                bad_train_station = set(train_pairs["station_fold"].unique()) & {fold, TEST_LAND}
                if bad_train_sensor or bad_train_station:
                    raise AssertionError(
                        f"Leakage in fit data for fold={fold}, pollutant={pollutant}: "
                        f"sensor_folds={bad_train_sensor}, station_folds={bad_train_station}"
                    )
                print(
                    f"  {pollutant}: fit pairs={len(train_pairs):,}, "
                    f"fit sensors={train_pairs['location'].nunique():,}, "
                    f"fit UBA stations={train_pairs['station_code'].nunique():,}; "
                    f"validation pairs={len(val_pairs):,}"
                )

                for method in methods:
                    fit = fit_method(train_pairs, method)
                    if fit is None or val_pairs.empty:
                        intercept, slope = np.nan, np.nan
                        metrics = {
                            "raw_mae": np.nan,
                            "corrected_mae": np.nan,
                            "raw_rmse": np.nan,
                            "corrected_rmse": np.nan,
                            "raw_bias": np.nan,
                            "corrected_bias": np.nan,
                            "raw_r2": np.nan,
                            "corrected_r2": np.nan,
                        }
                    else:
                        intercept, slope = fit
                        metrics, _ = evaluate_annual(val_pairs, intercept, slope)
                        coeffs_by_fold[str(radius)][fold].setdefault(method, {})[
                            pollutant
                        ] = {"intercept": intercept, "slope": slope}

                    stats = distance_stats(train_pairs)
                    rows.append(
                        {
                            "method": method,
                            "radius_km": radius,
                            "fold": fold,
                            "pollutant": pollutant,
                            "intercept": intercept,
                            "slope": slope,
                            **metrics,
                            "paired_days": int(len(train_pairs)),
                            "lowcost_locations": int(train_pairs["location"].nunique()),
                            "uba_stations": int(train_pairs["station_code"].nunique()),
                            **stats,
                        }
                    )

            for method in methods:
                coeffs = {
                    pollutant: (
                        coeffs_by_fold[str(radius)][fold].get(method, {}).get(pollutant, {}).get("intercept"),
                        coeffs_by_fold[str(radius)][fold].get(method, {}).get(pollutant, {}).get("slope"),
                    )
                    for pollutant in POLLUTANTS
                }
                coeffs = {
                    k: v for k, v in coeffs.items() if v[0] is not None and v[1] is not None
                }
                if coeffs:
                    annual = apply_coefficients_to_annual(non_test_daily, non_test_nodes, coeffs)
                    out = write_annual(paths, "fold", radius, method, year, annual, fold=fold)
                    print(f"  wrote fold labels for {method}: {len(annual):,} sensors -> {out}")

    return pd.DataFrame(rows), coeffs_by_fold


def aggregate_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level rows to mean/std summaries per method/radius/pollutant."""

    metric_cols = [
        "raw_mae",
        "corrected_mae",
        "raw_rmse",
        "corrected_rmse",
        "raw_bias",
        "corrected_bias",
        "raw_r2",
        "corrected_r2",
        "paired_days",
        "lowcost_locations",
        "uba_stations",
        "median_distance_km",
        "p90_distance_km",
        "max_distance_km",
    ]
    summary = (
        cv.groupby(["method", "radius_km", "pollutant"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def select_final(cv: pd.DataFrame, final_method: str | None, final_radius: float | None) -> tuple[str, float]:
    """Choose the final method/radius using validation metrics only."""

    choices = cv.dropna(subset=["corrected_rmse", "corrected_mae"]).copy()
    if choices.empty:
        raise RuntimeError("No valid inner-fold metrics available for final selection.")
    if final_method:
        choices = choices[choices["method"] == final_method]
    if final_radius is not None:
        choices = choices[np.isclose(choices["radius_km"], final_radius)]
    if choices.empty:
        raise RuntimeError("Explicit final method/radius did not match any valid CV result.")
    ranked = (
        choices.groupby(["method", "radius_km"], as_index=False)
        .agg(mean_rmse=("corrected_rmse", "mean"), mean_mae=("corrected_mae", "mean"))
        .sort_values(["mean_rmse", "mean_mae", "method", "radius_km"])
    )
    best = ranked.iloc[0]
    return str(best["method"]), float(best["radius_km"])


def fit_final(
    paths: Paths,
    daily: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    method: str,
    radius: float,
    year: int,
) -> dict:
    """Fit final coefficients on all non-test data and write train/test labels."""

    train_nodes = nodes[nodes["land"] != TEST_LAND].copy()
    test_nodes = nodes[nodes["land"] == TEST_LAND].copy()
    train_daily = daily[daily["location"].isin(train_nodes["location"])].copy()
    test_daily = daily[daily["location"].isin(test_nodes["location"])].copy()
    train_uba = uba[uba["land"] != TEST_LAND].copy()

    final_coeffs = {}
    for pollutant in POLLUTANTS:
        pairs = make_pairs(train_daily, train_nodes, train_uba, radius, pollutant)
        # The final fit may use every non-test fold, but it still must not see
        # Sachsen-Anhalt sensors or stations.
        bad_sensor = set(pairs["sensor_fold"].unique()) & {TEST_LAND}
        bad_station = set(pairs["station_fold"].unique()) & {TEST_LAND}
        if bad_sensor or bad_station:
            raise AssertionError(
                f"Sachsen-Anhalt leaked into final fit for {pollutant}: "
                f"sensor_folds={bad_sensor}, station_folds={bad_station}"
            )
        fit = fit_method(pairs, method)
        if fit is None:
            raise RuntimeError(f"Could not fit final {method} model for {pollutant}")
        intercept, slope = fit
        final_coeffs[pollutant] = {
            "intercept": intercept,
            "slope": slope,
            "paired_days": int(len(pairs)),
            "lowcost_locations": int(pairs["location"].nunique()),
            "uba_stations": int(pairs["station_code"].nunique()),
            **distance_stats(pairs),
        }

    coeff_tuple = {
        pollutant: (vals["intercept"], vals["slope"])
        for pollutant, vals in final_coeffs.items()
    }
    train_annual = apply_coefficients_to_annual(train_daily, train_nodes, coeff_tuple)
    test_annual = apply_coefficients_to_annual(test_daily, test_nodes, coeff_tuple)
    train_out = write_annual(paths, "train", radius, method, year, train_annual)
    test_out = write_annual(paths, "test", radius, method, year, test_annual)

    print(
        f"\nFinal selection: method={method}, radius={radius:g} km. "
        "The same coefficients were applied to train and Sachsen-Anhalt test labels."
    )
    print(f"  train labels: {len(train_annual):,} sensors -> {train_out}")
    print(f"  test labels:  {len(test_annual):,} sensors -> {test_out}")
    return final_coeffs


def parse_args() -> argparse.Namespace:
    """Parse the public CLI for the standalone regression calibration."""

    parser = argparse.ArgumentParser(
        description="Run alternative OLS/Huber SDS011 reference adjustment."
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Months to load, e.g. 2024-04 2024-05. Default: all merged hourly PM months found.",
    )
    parser.add_argument(
        "--method",
        choices=["ols", "huber", "both"],
        default="both",
        help="Calibration method(s) to run.",
    )
    parser.add_argument(
        "--radius-km",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0],
        help="One or more nearest-UBA-station radii to evaluate.",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Processed data root. Defaults to data/processed under the repository root.",
    )
    parser.add_argument(
        "--final-method",
        choices=["ols", "huber"],
        default=None,
        help="Explicit final method. Default: select by lowest mean validation RMSE, MAE tie-break.",
    )
    parser.add_argument(
        "--final-radius-km",
        type=float,
        default=None,
        help="Explicit final radius. Default: selected with the final method from validation metrics.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Overwrite existing regression_reference_adjustment outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Load inputs, run cross-validation, select final settings, and write outputs."""

    args = parse_args()
    paths = Paths(Path(args.processed_dir))
    guard_outputs(paths, args.refit)
    require_files(paths, args.year)

    months = args.months or discover_months(paths)
    months = sorted(dict.fromkeys(months))
    if not months:
        raise FileNotFoundError(f"No merged hourly PM month files found in {paths.pm_hourly_dir}")

    methods = methods_from_arg(args.method)
    radii = sorted(dict.fromkeys(args.radius_km))
    print(f"Processed data root: {paths.processed}")
    print(f"Methods: {methods}")
    print(f"Radii: {radii}")

    sensor_land = load_sensor_land(paths)
    daily = load_daily_sds011(paths, months)
    nodes = load_nodes(paths, months, sensor_land)
    daily = daily[daily["location"].isin(nodes["location"])].copy()
    uba = load_uba(paths, args.year)

    cv, coeffs_by_fold = run_cross_validation(
        paths, daily, nodes, uba, methods, radii, args.year
    )
    paths.calibration_root.mkdir(parents=True, exist_ok=True)
    cv_path = paths.calibration_root / "cv_results.csv"
    cv.to_csv(cv_path, index=False)
    summary = aggregate_summary(cv)
    summary_path = paths.calibration_root / "cv_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_json(paths.calibration_root / "fold_coefficients.json", coeffs_by_fold)
    selection_path = paths.calibration_root / "selection_summary.csv"
    final_selection_table(cv).to_csv(selection_path, index=False)

    final_method, final_radius = select_final(
        cv, args.final_method, args.final_radius_km
    )
    final_coeffs = fit_final(
        paths, daily, nodes, uba, final_method, final_radius, args.year
    )
    write_run_metadata(
        paths,
        year=args.year,
        months=months,
        methods=methods,
        radii=radii,
        final_method=final_method,
        final_radius=final_radius,
        cv=cv,
    )
    write_json(
        paths.calibration_root / "final_coefficients.json",
        {
            "selection_rule": (
                "Explicit CLI choice if provided; otherwise lowest mean inner-fold "
                "corrected RMSE with mean corrected MAE as tie-breaker."
            ),
            "method": final_method,
            "radius_km": final_radius,
            "coefficients": final_coeffs,
        },
    )

    print(f"\nWrote CV results -> {cv_path}")
    print(f"Wrote CV summary -> {summary_path}")
    print(f"Wrote selection summary -> {selection_path}")


if __name__ == "__main__":
    main()
