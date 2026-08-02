#!/usr/bin/env python3
"""
Inventory close SDS011-UBA pairs and locally available covariates.

This is an exploratory feasibility script for the next calibration step. It
does not fit a calibration model, does not write corrected labels, and does not
modify existing calibration outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_pm_regression_loo as calib


MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
SUB_1KM_RADII_KM = [0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
DEFAULT_RADII_KM = [*SUB_1KM_RADII_KM, 2.0, 5.0, 10.0, 20.0]
DISTANCE_BANDS_KM = [
    (0.0, 0.025, "0-25 m"),
    (0.025, 0.05, "25-50 m"),
    (0.05, 0.1, "50-100 m"),
    (0.1, 0.15, "100-150 m"),
    (0.15, 0.25, "150-250 m"),
    (0.25, 0.5, "250-500 m"),
    (0.5, 1.0, "500 m-1 km"),
]
WEATHER_DISTANCE_THRESHOLDS_KM = [0.0, 0.1, 0.5, 1.0, 2.0]
HUMIDITY_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "humidity_clip90",
    "frac_gt90",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for a repository-relative EDA run."""

    parser = argparse.ArgumentParser(
        description="EDA inventory for close SDS011-UBA reference calibration pairs."
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
            "Output directory. Defaults to "
            "data/processed/calibration/regression_reference_adjustment/"
            "eda_close_reference_inventory."
        ),
    )
    parser.add_argument(
        "--weather-radius-km",
        type=float,
        default=1.0,
        help="Nearest weather sensor radius used for day-level covariate coverage.",
    )
    return parser.parse_args()


def discover_months(directory: Path, prefix: str = "", suffix: str = ".parquet") -> list[str]:
    """Return YYYY-MM file tokens from one directory."""

    if not directory.exists():
        return []
    months = []
    for path in directory.glob(f"{prefix}*{suffix}"):
        name = path.stem
        if prefix and name.startswith(prefix):
            name = name.removeprefix(prefix)
        if MONTH_RE.match(name):
            months.append(name)
    return sorted(set(months))


def season_name(month: int) -> str:
    """Meteorological season label for a month number."""

    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def schema_columns(path: Path) -> list[str]:
    """Read column names from a CSV or Parquet file without loading it fully."""

    if not path.exists():
        return []
    if path.suffix == ".parquet":
        return read_parquet_columns(path)
    return list(pd.read_csv(path, nrows=0).columns)


def read_parquet_columns(path: Path) -> list[str]:
    """Return Parquet columns with a tiny fallback for older pandas/pyarrow setups."""

    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def inventory_local_data(processed: Path, year: int, months: list[str]) -> pd.DataFrame:
    """Inventory relevant local data products and note likely missing pieces."""

    pm_dir = processed / "hourly" / "pm" / "all_pm_sensors"
    pm_nodes = processed / "hourly" / "pm" / "nodes"
    humidity_dir = processed / "hourly" / "humidity" / "all_sensors"
    humidity_nodes = processed / "hourly" / "humidity" / "nodes"
    daily_humidity = processed / "daily_avg" / "humidity"
    uba_daily = processed / "daily_avg" / "uba" / f"pm_reference_stations_{year}.csv"
    station_land = processed / "uba" / "station_land.csv"
    sensor_land = processed / "sensor_land.csv"

    def month_text(values: list[str]) -> str:
        return " ".join(values) if values else ""

    rows = [
        {
            "dataset": "merged hourly PM",
            "path": str(pm_dir / "<YYYY-MM>.parquet"),
            "exists": pm_dir.exists(),
            "available_months": month_text(discover_months(pm_dir)),
            "columns": ", ".join(read_parquet_columns(pm_dir / f"{months[0]}.parquet"))
            if months and (pm_dir / f"{months[0]}.parquet").exists()
            else "",
            "notes": "contains sensor_type; SDS011 filter can be applied locally",
        },
        {
            "dataset": "SDS011 node coordinates",
            "path": str(pm_nodes / "sds011_<YYYY-MM>.parquet"),
            "exists": pm_nodes.exists(),
            "available_months": month_text(discover_months(pm_nodes, "sds011_")),
            "columns": ", ".join(read_parquet_columns(pm_nodes / f"sds011_{months[0]}.parquet"))
            if months and (pm_nodes / f"sds011_{months[0]}.parquet").exists()
            else "",
            "notes": "used for sensor-UBA distances",
        },
        {
            "dataset": "merged hourly humidity/weather",
            "path": str(humidity_dir / "<YYYY-MM>.parquet"),
            "exists": humidity_dir.exists(),
            "available_months": month_text(discover_months(humidity_dir)),
            "columns": ", ".join(read_parquet_columns(humidity_dir / f"{months[0]}.parquet"))
            if months and (humidity_dir / f"{months[0]}.parquet").exists()
            else "",
            "notes": "has relative humidity, humidity_clip90, temperature, and BME280 pressure",
        },
        {
            "dataset": "humidity/weather node coordinates",
            "path": str(humidity_nodes / "<sensor_type>_<YYYY-MM>.parquet"),
            "exists": humidity_nodes.exists(),
            "available_months": (
                "bme280: "
                + month_text(discover_months(humidity_nodes, "bme280_"))
                + " | dht22: "
                + month_text(discover_months(humidity_nodes, "dht22_"))
            ),
            "columns": "",
            "notes": "used to locate nearest local weather sensor",
        },
        {
            "dataset": "daily humidity/weather",
            "path": str(daily_humidity / "<sensor_type>_<YYYY-MM>.csv"),
            "exists": daily_humidity.exists(),
            "available_months": (
                "bme280: "
                + month_text(discover_months(daily_humidity, "bme280_", ".csv"))
                + " | dht22: "
                + month_text(discover_months(daily_humidity, "dht22_", ".csv"))
            ),
            "columns": ", ".join(schema_columns(next(iter(sorted(daily_humidity.glob("*.csv"))), Path())))
            if daily_humidity.exists() and any(daily_humidity.glob("*.csv"))
            else "",
            "notes": "partial local daily products; this EDA rebuilds daily coverage from hourly all_sensors",
        },
        {
            "dataset": "UBA daily PM reference",
            "path": str(uba_daily),
            "exists": uba_daily.exists(),
            "available_months": "daily 2024 file" if uba_daily.exists() else "",
            "columns": ", ".join(schema_columns(uba_daily)),
            "notes": "daily PM10 and PM2.5 are available locally",
        },
        {
            "dataset": "UBA hourly PM reference",
            "path": str(processed / "hourly" / "uba"),
            "exists": (processed / "hourly" / "uba").exists(),
            "available_months": "",
            "columns": "",
            "notes": "not found locally",
        },
        {
            "dataset": "UBA station Land lookup",
            "path": str(station_land),
            "exists": station_land.exists(),
            "available_months": "",
            "columns": ", ".join(schema_columns(station_land)),
            "notes": "Land/fold available; station type/class metadata is not present here",
        },
        {
            "dataset": "SDS011 sensor Land lookup",
            "path": str(sensor_land),
            "exists": sensor_land.exists(),
            "available_months": "",
            "columns": ", ".join(schema_columns(sensor_land)),
            "notes": "corrected state assignment",
        },
    ]
    return pd.DataFrame(rows)


def station_type_inventory(processed: Path) -> pd.DataFrame:
    """Scan local UBA files for station-type-like columns."""

    candidates = sorted((processed / "uba").glob("*")) + sorted(
        (processed / "daily_avg" / "uba").glob("*")
    )
    keywords = ("type", "station_type", "classification", "traffic", "background", "area")
    rows = []
    for path in candidates:
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.suffix not in {".csv", ".json", ".parquet"}:
            continue
        try:
            cols = read_parquet_columns(path) if path.suffix == ".parquet" else schema_columns(path)
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "columns": "",
                    "station_type_like_columns": "",
                    "status": f"could not inspect: {type(exc).__name__}",
                }
            )
            continue
        matches = [c for c in cols if any(k in c.lower() for k in keywords)]
        rows.append(
            {
                "path": str(path),
                "columns": ", ".join(cols),
                "station_type_like_columns": ", ".join(matches),
                "status": "station-type metadata found" if matches else "no station-type metadata",
            }
        )
    return pd.DataFrame(rows)


def all_sensor_station_links(
    nodes: pd.DataFrame, uba: pd.DataFrame, max_radius_km: float
) -> pd.DataFrame:
    """Return all SDS011-UBA station links within max radius plus nearest flags."""

    stations = (
        uba.drop_duplicates("station_code")[
            ["station_code", "lat", "lon", "land", "fold"]
        ]
        .rename(
            columns={
                "lat": "station_lat",
                "lon": "station_lon",
                "land": "station_land",
                "fold": "station_fold",
            }
        )
        .reset_index(drop=True)
    )
    sensor_cols = (
        nodes[["location", "lat", "lon", "land", "fold"]]
        .rename(
            columns={
                "lat": "sensor_lat",
                "lon": "sensor_lon",
                "land": "sensor_land",
                "fold": "sensor_fold",
            }
        )
        .reset_index(drop=True)
    )
    distances = calib.haversine_km(
        sensor_cols["sensor_lat"].to_numpy()[:, None],
        sensor_cols["sensor_lon"].to_numpy()[:, None],
        stations["station_lat"].to_numpy()[None, :],
        stations["station_lon"].to_numpy()[None, :],
    )
    nearest_idx = distances.argmin(axis=1)
    sensor_idx, station_idx = np.where(distances <= max_radius_km)
    links = sensor_cols.iloc[sensor_idx].reset_index(drop=True)
    station_part = stations.iloc[station_idx].reset_index(drop=True)
    out = pd.concat([links, station_part], axis=1)
    out["dist_km"] = distances[sensor_idx, station_idx]
    out["is_nearest_station"] = station_idx == nearest_idx[sensor_idx]
    return out.sort_values(["dist_km", "location", "station_code"]).reset_index(drop=True)


def summarize_static_links(links: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    """Summarize all close links and nearest-only calibration-style links."""

    rows = []
    for radius in radii:
        sub = links[links["dist_km"] <= radius]
        nearest = sub[sub["is_nearest_station"]]
        rows.append(
            {
                "radius_km": radius,
                "all_sensor_station_pairs": int(len(sub)),
                "all_unique_sds011_sensors": int(sub["location"].nunique()),
                "all_unique_uba_stations": int(sub["station_code"].nunique()),
                "nearest_sensor_station_pairs": int(len(nearest)),
                "nearest_unique_sds011_sensors": int(nearest["location"].nunique()),
                "nearest_unique_uba_stations": int(nearest["station_code"].nunique()),
                "median_nearest_distance_km": float(nearest["dist_km"].median())
                if not nearest.empty
                else np.nan,
                "p90_nearest_distance_km": float(nearest["dist_km"].quantile(0.9))
                if not nearest.empty
                else np.nan,
                "max_nearest_distance_km": float(nearest["dist_km"].max())
                if not nearest.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_geography(links: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    """Count nearest close links by radius, sensor fold, and station fold."""

    rows = []
    for radius in radii:
        sub = links[(links["dist_km"] <= radius) & links["is_nearest_station"]]
        if sub.empty:
            continue
        grouped = sub.groupby(["sensor_fold", "station_fold"], dropna=False)
        for (sensor_fold, station_fold), group in grouped:
            rows.append(
                {
                    "radius_km": radius,
                    "sensor_fold": sensor_fold,
                    "station_fold": station_fold,
                    "nearest_pairs": int(len(group)),
                    "unique_sds011_sensors": int(group["location"].nunique()),
                    "unique_uba_stations": int(group["station_code"].nunique()),
                    "median_distance_km": float(group["dist_km"].median()),
                    "p90_distance_km": float(group["dist_km"].quantile(0.9)),
                }
            )
    return pd.DataFrame(rows)


def build_daily_pair_summaries(
    daily_pm: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    radii: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, str], pd.DataFrame]]:
    """Build seasonal/monthly coverage summaries for nearest close pairs."""

    overall_rows = []
    time_rows = []
    pairs_by_key: dict[tuple[float, str], pd.DataFrame] = {}
    for radius in radii:
        for pollutant in calib.POLLUTANTS:
            pairs = calib.make_pairs(daily_pm, nodes, uba, radius, pollutant)
            pairs_by_key[(radius, pollutant)] = pairs
            if pairs.empty:
                overall_rows.append(
                    {
                        "radius_km": radius,
                        "pollutant": pollutant,
                        "paired_sensor_days": 0,
                        "unique_sds011_sensors": 0,
                        "unique_uba_stations": 0,
                        "median_days_per_sensor_station": np.nan,
                    }
                )
                continue
            days_per_link = pairs.groupby(["location", "station_code"])["date"].nunique()
            overall_rows.append(
                {
                    "radius_km": radius,
                    "pollutant": pollutant,
                    "paired_sensor_days": int(len(pairs)),
                    "unique_sds011_sensors": int(pairs["location"].nunique()),
                    "unique_uba_stations": int(pairs["station_code"].nunique()),
                    "median_days_per_sensor_station": float(days_per_link.median()),
                    "p10_days_per_sensor_station": float(days_per_link.quantile(0.1)),
                    "p90_days_per_sensor_station": float(days_per_link.quantile(0.9)),
                }
            )
            tmp = pairs.copy()
            dates = pd.to_datetime(tmp["date"])
            tmp["month"] = dates.dt.strftime("%Y-%m")
            tmp["season"] = dates.dt.month.map(season_name)
            for (period_type, period), group in tmp.groupby(["season", "month"]):
                time_rows.append(
                    {
                        "radius_km": radius,
                        "pollutant": pollutant,
                        "season": period_type,
                        "month": period,
                        "paired_sensor_days": int(len(group)),
                        "unique_sds011_sensors": int(group["location"].nunique()),
                        "unique_uba_stations": int(group["station_code"].nunique()),
                    }
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(time_rows), pairs_by_key


def load_humidity_nodes(processed: Path, months: list[str]) -> pd.DataFrame:
    """Load BME280/DHT22 node coordinates and collapse them by location."""

    node_dir = processed / "hourly" / "humidity" / "nodes"
    parts = []
    for sensor_type in ("bme280", "dht22"):
        for month in months:
            path = node_dir / f"{sensor_type}_{month}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(path, columns=["location", "lat", "lon"])
            part["sensor_type"] = sensor_type
            part["month"] = month
            parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["weather_location", "weather_lat", "weather_lon", "weather_sensor_types"]
        )
    raw = pd.concat(parts, ignore_index=True)
    grouped = (
        raw.groupby("location", as_index=False)
        .agg(
            weather_lat=("lat", "median"),
            weather_lon=("lon", "median"),
            weather_sensor_types=("sensor_type", lambda s: ",".join(sorted(set(s)))),
            weather_months_seen=("month", "nunique"),
        )
        .rename(columns={"location": "weather_location"})
    )
    return grouped


def nearest_weather_matches(
    sensors: pd.DataFrame, weather_nodes: pd.DataFrame
) -> pd.DataFrame:
    """Find each SDS011 sensor's nearest local humidity/weather sensor."""

    if sensors.empty or weather_nodes.empty:
        return pd.DataFrame(
            columns=[
                "location",
                "nearest_weather_location",
                "nearest_weather_distance_km",
                "weather_sensor_types",
            ]
        )
    distances = calib.haversine_km(
        sensors["lat"].to_numpy()[:, None],
        sensors["lon"].to_numpy()[:, None],
        weather_nodes["weather_lat"].to_numpy()[None, :],
        weather_nodes["weather_lon"].to_numpy()[None, :],
    )
    best = distances.argmin(axis=1)
    out = pd.DataFrame(
        {
            "location": sensors["location"].to_numpy(),
            "nearest_weather_location": weather_nodes["weather_location"].to_numpy()[best],
            "nearest_weather_distance_km": distances[np.arange(len(sensors)), best],
            "weather_sensor_types": weather_nodes["weather_sensor_types"].to_numpy()[best],
        }
    )
    out["same_location_weather"] = (
        out["location"].astype(str) == out["nearest_weather_location"].astype(str)
    )
    return out


def load_daily_humidity(
    processed: Path,
    months: list[str],
    target_locations: set[int],
) -> pd.DataFrame:
    """Rebuild daily local weather covariates from merged hourly humidity files."""

    if not target_locations:
        return pd.DataFrame(columns=["location", "date", "n_weather_hours"] + HUMIDITY_COLUMNS)
    hum_dir = processed / "hourly" / "humidity" / "all_sensors"
    daily_parts = []
    columns = ["location", "hour", *HUMIDITY_COLUMNS]
    target_list = sorted(int(x) for x in target_locations)
    for month in months:
        path = hum_dir / f"{month}.parquet"
        if not path.exists():
            continue
        try:
            h = pd.read_parquet(path, columns=columns, filters=[("location", "in", target_list)])
        except Exception:
            h = pd.read_parquet(path, columns=columns)
            h = h[h["location"].isin(target_list)]
        if h.empty:
            continue
        h["hour"] = pd.to_datetime(h["hour"], errors="coerce")
        h = h.dropna(subset=["hour"])
        h["date"] = (h["hour"] + pd.Timedelta(hours=calib.UTC_TO_MEZ_HOURS)).dt.date
        for col in HUMIDITY_COLUMNS:
            h[col] = pd.to_numeric(h[col], errors="coerce")
        daily = (
            h.groupby(["location", "date"], as_index=False)
            .agg(
                temperature=("temperature", "mean"),
                humidity=("humidity", "mean"),
                pressure=("pressure", "mean"),
                humidity_clip90=("humidity_clip90", "mean"),
                frac_gt90=("frac_gt90", "mean"),
                n_weather_hours=("hour", "nunique"),
            )
        )
        daily_parts.append(daily)
        print(f"{month}: humidity rows for target locations -> {len(h):,}")
    if not daily_parts:
        return pd.DataFrame(columns=["location", "date", "n_weather_hours"] + HUMIDITY_COLUMNS)
    daily = pd.concat(daily_parts, ignore_index=True)

    # UTC-to-MEZ date shifting can create duplicate location-date fragments at
    # month boundaries. Combine them with hour weights while preserving NaN for
    # covariates that were truly absent, especially BME280-only pressure.
    agg_spec = {"n_weather_hours": ("n_weather_hours", "sum")}
    for col in HUMIDITY_COLUMNS:
        valid = daily[col].notna()
        daily[f"{col}_weighted"] = np.where(
            valid, daily[col] * daily["n_weather_hours"], 0.0
        )
        daily[f"{col}_valid_hours"] = np.where(valid, daily["n_weather_hours"], 0.0)
        agg_spec[f"{col}_weighted"] = (f"{col}_weighted", "sum")
        agg_spec[f"{col}_valid_hours"] = (f"{col}_valid_hours", "sum")
    daily = daily.groupby(["location", "date"], as_index=False).agg(**agg_spec)
    for col in HUMIDITY_COLUMNS:
        valid_hours = daily[f"{col}_valid_hours"]
        daily[col] = np.where(
            valid_hours > 0,
            daily[f"{col}_weighted"] / valid_hours,
            np.nan,
        )
    drop_cols = [
        c for col in HUMIDITY_COLUMNS for c in (f"{col}_weighted", f"{col}_valid_hours")
    ]
    daily = daily.drop(columns=drop_cols)
    return daily[daily["n_weather_hours"] >= calib.MIN_HOURS_PER_DAY].copy()


def summarize_weather_matches(
    links: pd.DataFrame,
    weather_matches: pd.DataFrame,
    radii: list[float],
) -> pd.DataFrame:
    """Summarize static nearest-weather availability for close SDS011 sensors."""

    rows = []
    for radius in radii:
        close_locations = set(
            links.loc[
                (links["dist_km"] <= radius) & links["is_nearest_station"], "location"
            ]
        )
        sub = weather_matches[weather_matches["location"].isin(close_locations)]
        row = {
            "radius_km": radius,
            "close_sds011_sensors": len(close_locations),
            "same_location_weather_sensors": int(sub["same_location_weather"].sum()),
        }
        for threshold in WEATHER_DISTANCE_THRESHOLDS_KM:
            row[f"nearest_weather_within_{threshold:g}km"] = int(
                (sub["nearest_weather_distance_km"] <= threshold).sum()
            )
        row["median_nearest_weather_distance_km"] = float(
            sub["nearest_weather_distance_km"].median()
        ) if not sub.empty else np.nan
        row["p90_nearest_weather_distance_km"] = float(
            sub["nearest_weather_distance_km"].quantile(0.9)
        ) if not sub.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_daily_weather_coverage(
    pairs_by_key: dict[tuple[float, str], pd.DataFrame],
    weather_matches: pd.DataFrame,
    weather_daily: pd.DataFrame,
    weather_radius_km: float,
) -> pd.DataFrame:
    """Count calibration pair-days that can receive local weather covariates."""

    rows = []
    weather_daily = weather_daily.copy()
    same_daily = weather_daily[["location", "date", *HUMIDITY_COLUMNS]].copy()
    nearest_daily = weather_daily.rename(columns={"location": "nearest_weather_location"})
    match_map = weather_matches[
        [
            "location",
            "nearest_weather_location",
            "nearest_weather_distance_km",
            "same_location_weather",
        ]
    ].copy()
    match_map = match_map[match_map["nearest_weather_distance_km"] <= weather_radius_km]
    for (radius, pollutant), pairs in pairs_by_key.items():
        if pairs.empty:
            rows.append(
                {
                    "radius_km": radius,
                    "pollutant": pollutant,
                    "paired_sensor_days": 0,
                    "same_location_weather_days": 0,
                    "nearest_weather_days": 0,
                }
            )
            continue
        base = pairs[["location", "station_code", "date"]].drop_duplicates().copy()
        same = base.merge(same_daily, on=["location", "date"], how="left")
        near = (
            base.merge(match_map, on="location", how="left")
            .merge(nearest_daily, on=["nearest_weather_location", "date"], how="left")
        )

        def nonnull_count(frame: pd.DataFrame, col: str) -> int:
            return int(frame[col].notna().sum()) if col in frame.columns else 0

        row = {
            "radius_km": radius,
            "pollutant": pollutant,
            "paired_sensor_days": int(len(base)),
            "same_location_weather_days": nonnull_count(same, "humidity"),
            "same_location_weather_fraction": float(same["humidity"].notna().mean()),
            "nearest_weather_radius_km": weather_radius_km,
            "nearest_weather_days": nonnull_count(near, "humidity"),
            "nearest_weather_fraction": float(near["humidity"].notna().mean()),
        }
        for col in HUMIDITY_COLUMNS:
            row[f"same_location_{col}_days"] = nonnull_count(same, col)
            row[f"nearest_{col}_days"] = nonnull_count(near, col)
        rows.append(row)
    return pd.DataFrame(rows)


def radius_label(radius_km: float) -> str:
    """Human-readable distance threshold label."""

    metres = int(round(radius_km * 1000))
    return f"{metres} m" if metres < 1000 else f"{metres // 1000} km"


def make_pairs_from_links(
    daily_pm: pd.DataFrame,
    links: pd.DataFrame,
    uba: pd.DataFrame,
    pollutant: str,
) -> pd.DataFrame:
    """Build daily PM/reference pairs for an explicit sensor-station link table."""

    spec = calib.POLLUTANTS[pollutant]
    ref = uba[["station_code", "date", spec["ref"]]].rename(
        columns={spec["ref"]: "ref"}
    )
    out = (
        daily_pm[["location", "date", spec["lowcost"]]]
        .rename(columns={spec["lowcost"]: "raw"})
        .merge(
            links[
                [
                    "location",
                    "station_code",
                    "dist_km",
                    "sensor_land",
                    "sensor_fold",
                    "station_land",
                    "station_fold",
                ]
            ],
            on="location",
        )
        .merge(ref, on=["station_code", "date"])
        .dropna(subset=["raw", "ref"])
    )
    return out[(out["raw"] > 0) & np.isfinite(out["raw"]) & np.isfinite(out["ref"])].copy()


def format_values(values: pd.Series) -> str:
    """Compact semicolon-delimited sorted unique values for CSV cells."""

    return "; ".join(str(v) for v in sorted(values.dropna().unique()))


def weather_available_locations(weather_daily: pd.DataFrame) -> set[int]:
    """Locations with at least one daily RH and temperature observation."""

    if weather_daily.empty:
        return set()
    available = weather_daily[
        weather_daily["humidity"].notna() & weather_daily["temperature"].notna()
    ]
    return set(available["location"].dropna().astype(int))


def pair_weather_fractions(
    pairs: pd.DataFrame,
    weather_matches: pd.DataFrame,
    weather_daily: pd.DataFrame,
    nearest_radius_km: float,
) -> dict[str, float]:
    """Compute same-location and nearest-weather RH/temperature pair-day coverage."""

    if pairs.empty:
        return {
            "same_location_weather_pair_days": 0,
            "same_location_weather_pair_day_pct": np.nan,
            "nearest_weather_pair_days": 0,
            "nearest_weather_pair_day_pct": np.nan,
        }

    base = pairs[["location", "station_code", "date"]].drop_duplicates().copy()
    same_daily = weather_daily[["location", "date", "humidity", "temperature"]].copy()
    same = base.merge(same_daily, on=["location", "date"], how="left")
    same_ok = same["humidity"].notna() & same["temperature"].notna()

    match_map = weather_matches[
        ["location", "nearest_weather_location", "nearest_weather_distance_km"]
    ].copy()
    match_map = match_map[match_map["nearest_weather_distance_km"] <= nearest_radius_km]
    nearest_daily = weather_daily[
        ["location", "date", "humidity", "temperature"]
    ].rename(columns={"location": "nearest_weather_location"})
    near = (
        base.merge(match_map, on="location", how="left")
        .merge(nearest_daily, on=["nearest_weather_location", "date"], how="left")
    )
    near_ok = near["humidity"].notna() & near["temperature"].notna()
    return {
        "same_location_weather_pair_days": int(same_ok.sum()),
        "same_location_weather_pair_day_pct": float(100.0 * same_ok.mean()),
        "nearest_weather_pair_days": int(near_ok.sum()),
        "nearest_weather_pair_day_pct": float(100.0 * near_ok.mean()),
    }


def summarize_sub_1km_thresholds(
    pairs_by_key: dict[tuple[float, str], pd.DataFrame],
    links: pd.DataFrame,
    weather_matches: pd.DataFrame,
    weather_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize requested cumulative sub-1 km thresholds by pollutant."""

    rows = []
    weather_locations = weather_available_locations(weather_daily)
    nearest_weather_500 = weather_matches[
        weather_matches["nearest_weather_distance_km"] <= 0.5
    ]
    for radius in SUB_1KM_RADII_KM:
        static_nearest = links[
            (links["dist_km"] <= radius) & links["is_nearest_station"]
        ].copy()
        for pollutant in calib.POLLUTANTS:
            pairs = pairs_by_key.get((radius, pollutant), pd.DataFrame()).copy()
            unique_pairs = pairs[["location", "station_code"]].drop_duplicates()
            close_locations = set(unique_pairs["location"].dropna().astype(int))
            station_sensor_counts = unique_pairs.groupby("station_code")["location"].nunique()
            days_per_sensor = pairs.groupby("location")["date"].nunique()
            dates = pd.to_datetime(pairs["date"]) if not pairs.empty else pd.Series(dtype="datetime64[ns]")
            weather = pair_weather_fractions(pairs, weather_matches, weather_daily, 0.5)
            same_location_locations = set(
                weather_matches.loc[
                    weather_matches["same_location_weather"], "location"
                ].dropna().astype(int)
            )
            nearest_weather_locations = set(
                nearest_weather_500["location"].dropna().astype(int)
            )
            nearest_weather_with_data = set(
                nearest_weather_500.loc[
                    nearest_weather_500["nearest_weather_location"].isin(weather_locations),
                    "location",
                ]
                .dropna()
                .astype(int)
            )
            rows.append(
                {
                    "threshold_km": radius,
                    "threshold_label": "<=" + radius_label(radius),
                    "threshold_m": int(round(radius * 1000)),
                    "pollutant": pollutant,
                    "unique_sds011_locations": int(unique_pairs["location"].nunique()),
                    "unique_uba_stations": int(unique_pairs["station_code"].nunique()),
                    "sensor_station_pairs": int(len(unique_pairs)),
                    "paired_sensor_days": int(len(pairs)),
                    "median_paired_days_per_sensor": float(days_per_sensor.median())
                    if not days_per_sensor.empty
                    else np.nan,
                    "fold_groups_represented": int(pairs["sensor_fold"].nunique())
                    if "sensor_fold" in pairs
                    else 0,
                    "fold_group_names": format_values(pairs["sensor_fold"])
                    if "sensor_fold" in pairs and not pairs.empty
                    else "",
                    "months_represented": int(dates.dt.strftime("%Y-%m").nunique())
                    if not pairs.empty
                    else 0,
                    "month_names": "; ".join(sorted(dates.dt.strftime("%Y-%m").unique()))
                    if not pairs.empty
                    else "",
                    "seasons_represented": int(dates.dt.month.map(season_name).nunique())
                    if not pairs.empty
                    else 0,
                    "season_names": "; ".join(sorted(dates.dt.month.map(season_name).unique()))
                    if not pairs.empty
                    else "",
                    "sensors_with_same_location_rh_temperature": int(
                        len(close_locations & same_location_locations & weather_locations)
                    ),
                    **weather,
                    "sensors_with_nearest_weather_within_500m": int(
                        len(close_locations & nearest_weather_locations)
                    ),
                    "sensors_with_nearest_weather_within_500m_and_data": int(
                        len(close_locations & nearest_weather_with_data)
                    ),
                    "uba_stations_with_exactly_one_sds011": int((station_sensor_counts == 1).sum()),
                    "uba_stations_with_multiple_sds011": int((station_sensor_counts > 1).sum()),
                    "max_sds011_sensors_assigned_to_one_uba": int(station_sensor_counts.max())
                    if not station_sensor_counts.empty
                    else 0,
                    "static_nearest_sensors_before_pm_ref_filter": int(
                        static_nearest["location"].nunique()
                    ),
                    "static_nearest_uba_before_pm_ref_filter": int(
                        static_nearest["station_code"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_distance_bands(
    pairs_by_key: dict[tuple[float, str], pd.DataFrame],
) -> pd.DataFrame:
    """Summarize non-cumulative sub-1 km distance bands by pollutant."""

    rows = []
    max_pairs = {
        pollutant: pairs_by_key.get((1.0, pollutant), pd.DataFrame()).copy()
        for pollutant in calib.POLLUTANTS
    }
    for lower, upper, label in DISTANCE_BANDS_KM:
        for pollutant, pairs in max_pairs.items():
            if pairs.empty:
                band = pairs
            else:
                lower_ok = pairs["dist_km"] >= lower if lower == 0 else pairs["dist_km"] > lower
                band = pairs[lower_ok & (pairs["dist_km"] <= upper)].copy()
            unique_pairs = band[["location", "station_code"]].drop_duplicates()
            days_per_sensor = band.groupby("location")["date"].nunique()
            dates = pd.to_datetime(band["date"]) if not band.empty else pd.Series(dtype="datetime64[ns]")
            rows.append(
                {
                    "distance_band": label,
                    "lower_m_exclusive": int(round(lower * 1000)) if lower > 0 else 0,
                    "upper_m_inclusive": int(round(upper * 1000)),
                    "pollutant": pollutant,
                    "unique_sds011_locations": int(unique_pairs["location"].nunique()),
                    "unique_uba_stations": int(unique_pairs["station_code"].nunique()),
                    "sensor_station_pairs": int(len(unique_pairs)),
                    "paired_sensor_days": int(len(band)),
                    "median_paired_days_per_sensor": float(days_per_sensor.median())
                    if not days_per_sensor.empty
                    else np.nan,
                    "fold_groups_represented": int(band["sensor_fold"].nunique())
                    if "sensor_fold" in band
                    else 0,
                    "months_represented": int(dates.dt.strftime("%Y-%m").nunique())
                    if not band.empty
                    else 0,
                    "seasons_represented": int(dates.dt.month.map(season_name).nunique())
                    if not band.empty
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def coordinate_decimal_places(value: float) -> int:
    """Infer apparent decimal places from the stored coordinate value."""

    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return len(text.split(".", 1)[1]) if "." in text else 0


def coordinate_diagnostics(links: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Inspect coordinate precision and duplicate-coordinate evidence within 100 m."""

    close = links[links["dist_km"] <= 0.1].copy()
    if close.empty:
        return {"pairs_within_100m": 0}, close
    close["distance_m"] = close["dist_km"] * 1000.0
    close["exact_zero_distance"] = close["distance_m"] <= 1e-6
    close["sensor_coord_key"] = (
        close["sensor_lat"].round(8).astype(str) + "," + close["sensor_lon"].round(8).astype(str)
    )
    close["station_coord_key"] = (
        close["station_lat"].round(8).astype(str) + "," + close["station_lon"].round(8).astype(str)
    )
    sensor_coord_counts = close.groupby("sensor_coord_key")["location"].nunique()
    station_coord_counts = close.groupby("station_coord_key")["station_code"].nunique()
    exact_coord_match = (
        np.isclose(close["sensor_lat"], close["station_lat"], rtol=0.0, atol=1e-10)
        & np.isclose(close["sensor_lon"], close["station_lon"], rtol=0.0, atol=1e-10)
    )
    explanation = "No exact zero-distance matches."
    if close["exact_zero_distance"].any():
        explanation = (
            "Exact zero-distance matches have identical stored coordinates; treat them as "
            "candidate co-locations only after checking station/site metadata."
        )
    elif (close["distance_m"] < 10).any():
        explanation = (
            "Some near-zero distances are below 10 m, but stored coordinates are not exact; "
            "rounding or nearby placement cannot be ruled out."
        )
    summary = {
        "pairs_within_100m": int(len(close)),
        "unique_sds011_locations_within_100m": int(close["location"].nunique()),
        "unique_uba_stations_within_100m": int(close["station_code"].nunique()),
        "exact_zero_distance_pairs": int(close["exact_zero_distance"].sum()),
        "exact_coordinate_match_pairs": int(exact_coord_match.sum()),
        "sensor_coordinate_groups_with_multiple_locations": int((sensor_coord_counts > 1).sum()),
        "station_coordinate_groups_with_multiple_stations": int((station_coord_counts > 1).sum()),
        "sensor_lat_decimal_places_min": int(close["sensor_lat"].map(coordinate_decimal_places).min()),
        "sensor_lat_decimal_places_max": int(close["sensor_lat"].map(coordinate_decimal_places).max()),
        "sensor_lon_decimal_places_min": int(close["sensor_lon"].map(coordinate_decimal_places).min()),
        "sensor_lon_decimal_places_max": int(close["sensor_lon"].map(coordinate_decimal_places).max()),
        "station_lat_decimal_places_min": int(close["station_lat"].map(coordinate_decimal_places).min()),
        "station_lat_decimal_places_max": int(close["station_lat"].map(coordinate_decimal_places).max()),
        "station_lon_decimal_places_min": int(close["station_lon"].map(coordinate_decimal_places).min()),
        "station_lon_decimal_places_max": int(close["station_lon"].map(coordinate_decimal_places).max()),
        "interpretation": explanation,
    }
    return summary, close


def pairs_within_250m_table(
    links: pd.DataFrame,
    daily_pm: pd.DataFrame,
    uba: pd.DataFrame,
    weather_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build requested row-level diagnostics for all SDS011-UBA pairs within 250 m."""

    close = links[links["dist_km"] <= 0.25].copy()
    if close.empty:
        return close
    pm10 = make_pairs_from_links(daily_pm, close, uba, "PM10")
    pm25 = make_pairs_from_links(daily_pm, close, uba, "PM2.5")
    pm10_days = (
        pm10.groupby(["location", "station_code"])["date"]
        .nunique()
        .rename("paired_pm10_days")
        .reset_index()
    )
    pm25_days = (
        pm25.groupby(["location", "station_code"])["date"]
        .nunique()
        .rename("paired_pm25_days")
        .reset_index()
    )
    out = (
        close.merge(pm10_days, on=["location", "station_code"], how="left")
        .merge(pm25_days, on=["location", "station_code"], how="left")
        .merge(
            weather_matches[["location", "same_location_weather"]],
            on="location",
            how="left",
        )
    )
    out["paired_pm10_days"] = out["paired_pm10_days"].fillna(0).astype(int)
    out["paired_pm25_days"] = out["paired_pm25_days"].fillna(0).astype(int)
    out["pm10_available"] = out["paired_pm10_days"] > 0
    out["pm25_available"] = out["paired_pm25_days"] > 0
    out["distance_m"] = out["dist_km"] * 1000.0
    out["exact_zero_distance"] = out["distance_m"] <= 1e-6
    out["sensor_coordinate_duplicate_count"] = out.groupby(
        ["sensor_lat", "sensor_lon"]
    )["location"].transform("nunique")
    return out[
        [
            "location",
            "sensor_lat",
            "sensor_lon",
            "station_code",
            "station_lat",
            "station_lon",
            "distance_m",
            "sensor_land",
            "sensor_fold",
            "station_land",
            "station_fold",
            "pm10_available",
            "pm25_available",
            "same_location_weather",
            "paired_pm10_days",
            "paired_pm25_days",
            "is_nearest_station",
            "exact_zero_distance",
            "sensor_coordinate_duplicate_count",
        ]
    ].sort_values(["distance_m", "location", "station_code"])


def feasibility_category(row: pd.Series) -> str:
    """Plain-language feasibility category for leave-one-UBA-station-out tests."""

    stations = row["unique_uba_stations"]
    sensors = row["unique_sds011_locations"]
    folds = row["fold_groups_represented"]
    weather_pct = row["nearest_weather_pair_day_pct"]
    if stations >= 100 and sensors >= 250 and folds >= 10 and weather_pct >= 60:
        return "best practical compromise"
    if stations >= 50 and sensors >= 100 and folds >= 8:
        return "feasible but small"
    if stations >= 20 and sensors >= 40:
        return "borderline"
    return "not enough independent coverage"


def write_sub_1km_plot(out_dir: Path, static_summary: pd.DataFrame) -> None:
    """Plot cumulative unique SDS011 sensors and UBA stations from 0 to 1 km."""

    os_env_dir = Path("/tmp/matplotlib-cache")
    os_env_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(os_env_dir))
    import matplotlib.pyplot as plt

    sub = static_summary[static_summary["radius_km"].isin(SUB_1KM_RADII_KM)].copy()
    sub = sub.sort_values("radius_km")
    metres = sub["radius_km"] * 1000.0
    plt.figure(figsize=(8, 5))
    plt.plot(
        metres,
        sub["nearest_unique_sds011_sensors"],
        marker="o",
        label="SDS011 sensors",
    )
    plt.plot(
        metres,
        sub["nearest_unique_uba_stations"],
        marker="o",
        label="UBA stations",
    )
    for x in [25, 50, 100, 150, 250, 500, 1000]:
        plt.axvline(x, color="0.85", linewidth=0.8, zorder=0)
    plt.xlabel("Cumulative nearest SDS011-UBA distance threshold (m)")
    plt.ylabel("Unique count")
    plt.title("Sub-1 km close-reference coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sub_1km_cumulative_coverage.png", dpi=170)
    plt.close()


def write_json(path: Path, obj: object) -> None:
    """Write pretty JSON and create parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    """Render a small markdown table without depending on tabulate."""

    if df.empty:
        return "_No rows._"
    shown = df.loc[:, columns].head(max_rows).copy()
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(
    out_dir: Path,
    sub_1km_summary: pd.DataFrame,
    distance_band_summary: pd.DataFrame,
    pairs_250m: pd.DataFrame,
    coordinate_summary: dict,
    static_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    weather_static: pd.DataFrame,
    weather_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    station_type: pd.DataFrame,
    months: list[str],
) -> None:
    """Write a concise reader-facing EDA summary."""

    close_1 = static_summary.loc[np.isclose(static_summary["radius_km"], 1.0)]
    close_5 = static_summary.loc[np.isclose(static_summary["radius_km"], 5.0)]
    station_type_missing = (
        station_type.empty
        or not station_type["station_type_like_columns"].fillna("").astype(bool).any()
    )
    focus = sub_1km_summary[
        sub_1km_summary["threshold_m"].isin([100, 250, 500, 1000])
    ].copy()
    pm10_focus = focus[focus["pollutant"] == "PM10"].copy()
    best_compromise = pm10_focus[
        pm10_focus["leave_one_uba_station_out_feasibility"].eq("best practical compromise")
    ]
    best_compromise_label = (
        best_compromise.iloc[0]["threshold_label"] if not best_compromise.empty else "not clear"
    )
    text = [
        "# Close Reference Inventory EDA",
        "",
        f"Run months: `{', '.join(months)}`",
        "",
        "## Primary CSV",
        "",
        "`sub_1km_pair_summary.csv` is the main decision table for this pass. "
        "`pairs_within_250m.csv` is kept separate because it is a row-level "
        "coordinate audit, not a threshold summary.",
        "",
        "## Sub-1 km Cumulative Thresholds",
        "",
        markdown_table(
            sub_1km_summary,
            [
                "threshold_label",
                "pollutant",
                "unique_sds011_locations",
                "unique_uba_stations",
                "sensor_station_pairs",
                "paired_sensor_days",
                "median_paired_days_per_sensor",
                "fold_groups_represented",
                "months_represented",
                "sensors_with_same_location_rh_temperature",
                "same_location_weather_pair_day_pct",
                "sensors_with_nearest_weather_within_500m",
                "nearest_weather_pair_day_pct",
                "uba_stations_with_exactly_one_sds011",
                "uba_stations_with_multiple_sds011",
                "max_sds011_sensors_assigned_to_one_uba",
            ],
            max_rows=14,
        ),
        "",
        "## Non-Cumulative Distance Bands",
        "",
        markdown_table(
            distance_band_summary,
            [
                "distance_band",
                "pollutant",
                "unique_sds011_locations",
                "unique_uba_stations",
                "sensor_station_pairs",
                "paired_sensor_days",
                "fold_groups_represented",
                "months_represented",
            ],
            max_rows=14,
        ),
        "",
        "## Feasibility For Leave-One-UBA-Station-Out",
        "",
        markdown_table(
            focus,
            [
                "threshold_label",
                "pollutant",
                "unique_sds011_locations",
                "unique_uba_stations",
                "paired_sensor_days",
                "fold_groups_represented",
                "nearest_weather_pair_day_pct",
                "leave_one_uba_station_out_feasibility",
            ],
            max_rows=8,
        ),
        "",
        "- Strongest spatial comparability: `<=100 m`, but only if the independent "
        "UBA-station count and coordinate audit are acceptable.",
        "- Strongest sample size: `<=1 km`.",
        f"- Best practical compromise by the simple EDA rule: `{best_compromise_label}` "
        "for PM10; confirm with PM2.5 needs before modeling.",
        "",
        "## Coordinate Diagnostics Within 100 m",
        "",
        f"- pairs within 100 m: `{coordinate_summary.get('pairs_within_100m', 0)}`",
        f"- exact zero-distance pairs: `{coordinate_summary.get('exact_zero_distance_pairs', 0)}`",
        f"- exact coordinate-match pairs: `{coordinate_summary.get('exact_coordinate_match_pairs', 0)}`",
        f"- sensor coordinate groups with multiple locations: "
        f"`{coordinate_summary.get('sensor_coordinate_groups_with_multiple_locations', 0)}`",
        f"- interpretation: {coordinate_summary.get('interpretation', 'not available')}",
        "",
        "Do not call zero or near-zero pairs co-located unless station/site metadata "
        "confirms they are genuinely at the same monitoring location.",
        "",
        "## Close SDS011-UBA Pairs",
        "",
        markdown_table(
            static_summary,
            [
                "radius_km",
                "nearest_sensor_station_pairs",
                "nearest_unique_sds011_sensors",
                "nearest_unique_uba_stations",
                "median_nearest_distance_km",
                "p90_nearest_distance_km",
            ],
        ),
        "",
        "## Daily Pair Coverage",
        "",
        markdown_table(
            daily_summary,
            [
                "radius_km",
                "pollutant",
                "paired_sensor_days",
                "unique_sds011_sensors",
                "unique_uba_stations",
                "median_days_per_sensor_station",
            ],
        ),
        "",
        "## Local Weather Match Coverage",
        "",
        markdown_table(
            weather_static,
            [
                "radius_km",
                "close_sds011_sensors",
                "same_location_weather_sensors",
                "nearest_weather_within_0.5km",
                "nearest_weather_within_1km",
                "median_nearest_weather_distance_km",
            ],
        ),
        "",
        markdown_table(
            weather_daily,
            [
                "radius_km",
                "pollutant",
                "paired_sensor_days",
                "same_location_weather_fraction",
                "nearest_weather_fraction",
                "nearest_pressure_days",
            ],
        ),
        "",
        "## Local Data Gaps",
        "",
        "- UBA daily PM10 and PM2.5 are available locally.",
        "- UBA hourly PM was not found locally.",
        "- Relative humidity and temperature are available from local Sensor.Community weather sensors.",
        "- Atmospheric pressure is available only where BME280 coverage exists and should be quality-checked before modeling.",
        "- UBA station-type/class metadata was "
        + ("not found locally." if station_type_missing else "found in at least one local file."),
    ]
    if not close_1.empty:
        row = close_1.iloc[0]
        text.extend(
            [
                "",
                "## Quick Feasibility Read",
                "",
                f"Within 1 km, there are `{int(row['nearest_unique_sds011_sensors'])}` "
                f"SDS011 sensors nearest-matched to `{int(row['nearest_unique_uba_stations'])}` "
                "UBA stations.",
            ]
        )
    if not close_5.empty:
        row = close_5.iloc[0]
        text.append(
            f"Within 5 km, this rises to `{int(row['nearest_unique_sds011_sensors'])}` "
            f"SDS011 sensors and `{int(row['nearest_unique_uba_stations'])}` UBA stations."
        )
    text.extend(
        [
            "",
            "## Output Files",
            "",
            "- `local_data_inventory.csv`",
            "- `station_type_inventory.csv`",
            "- `close_links_within_max_radius.csv`",
            "- `pair_counts_by_radius.csv`",
            "- `geographic_counts_by_radius_fold.csv`",
            "- `daily_pair_counts_by_radius.csv`",
            "- `seasonal_monthly_pair_coverage.csv`",
            "- `weather_node_match_by_sensor.csv`",
            "- `weather_match_summary_by_radius.csv`",
            "- `daily_weather_covariate_coverage.csv`",
            "- `sub_1km_pair_summary.csv`",
            "- `distance_band_summary.csv`",
            "- `pairs_within_250m.csv`",
            "- `sub_1km_cumulative_coverage.png`",
            "- `metadata.json`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(text) + "\n")


def main() -> None:
    args = parse_args()
    paths = calib.Paths(Path(args.processed_dir))
    months = args.months or calib.discover_months(paths)
    months = sorted(dict.fromkeys(months))
    radii = sorted(dict.fromkeys(float(r) for r in args.radius_km))
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else paths.calibration_root / "eda_close_reference_inventory"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processed root: {paths.processed}")
    print(f"Months: {months}")
    print(f"Radii: {radii}")
    print(f"Output: {out_dir}")

    inventory = inventory_local_data(paths.processed, args.year, months)
    station_type = station_type_inventory(paths.processed)

    calib.require_files(paths, args.year)
    sensor_land = calib.load_sensor_land(paths)
    uba = calib.load_uba(paths, args.year)
    nodes = calib.load_nodes(paths, months, sensor_land)
    daily_pm = calib.load_daily_sds011(paths, months)
    daily_pm = daily_pm[daily_pm["location"].isin(nodes["location"])].copy()

    max_radius = max(radii)
    links = all_sensor_station_links(nodes, uba, max_radius)
    static_summary = summarize_static_links(links, radii)
    geographic_summary = summarize_geography(links, radii)
    daily_summary, seasonal_summary, pairs_by_key = build_daily_pair_summaries(
        daily_pm, nodes, uba, radii
    )

    close_locations = set(
        links.loc[
            (links["dist_km"] <= max_radius) & links["is_nearest_station"], "location"
        ]
    )
    weather_nodes = load_humidity_nodes(paths.processed, months)
    weather_matches = nearest_weather_matches(
        nodes[nodes["location"].isin(close_locations)].copy(), weather_nodes
    )
    weather_static = summarize_weather_matches(links, weather_matches, radii)
    weather_locations = set(weather_matches["location"].dropna().astype(int))
    nearby_weather_locations = set(
        weather_matches.loc[
            weather_matches["nearest_weather_distance_km"] <= max(args.weather_radius_km, 0.5),
            "nearest_weather_location",
        ]
        .dropna()
        .astype(int)
    )
    weather_daily = load_daily_humidity(
        paths.processed, months, weather_locations | nearby_weather_locations
    )
    weather_daily_summary = summarize_daily_weather_coverage(
        pairs_by_key, weather_matches, weather_daily, args.weather_radius_km
    )
    sub_1km_summary = summarize_sub_1km_thresholds(
        pairs_by_key, links, weather_matches, weather_daily
    )
    sub_1km_summary["leave_one_uba_station_out_feasibility"] = sub_1km_summary.apply(
        feasibility_category, axis=1
    )
    distance_band_summary = summarize_distance_bands(pairs_by_key)
    coordinate_summary, _pairs_100m = coordinate_diagnostics(links)
    pairs_250m = pairs_within_250m_table(links, daily_pm, uba, weather_matches)

    inventory.to_csv(out_dir / "local_data_inventory.csv", index=False)
    station_type.to_csv(out_dir / "station_type_inventory.csv", index=False)
    links.to_csv(out_dir / "close_links_within_max_radius.csv", index=False)
    static_summary.to_csv(out_dir / "pair_counts_by_radius.csv", index=False)
    geographic_summary.to_csv(out_dir / "geographic_counts_by_radius_fold.csv", index=False)
    daily_summary.to_csv(out_dir / "daily_pair_counts_by_radius.csv", index=False)
    seasonal_summary.to_csv(out_dir / "seasonal_monthly_pair_coverage.csv", index=False)
    weather_matches.to_csv(out_dir / "weather_node_match_by_sensor.csv", index=False)
    weather_static.to_csv(out_dir / "weather_match_summary_by_radius.csv", index=False)
    weather_daily_summary.to_csv(out_dir / "daily_weather_covariate_coverage.csv", index=False)
    sub_1km_summary.to_csv(out_dir / "sub_1km_pair_summary.csv", index=False)
    distance_band_summary.to_csv(out_dir / "distance_band_summary.csv", index=False)
    pairs_250m.to_csv(out_dir / "pairs_within_250m.csv", index=False)
    write_sub_1km_plot(out_dir, static_summary)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": args.year,
        "months": months,
        "radii_km": radii,
        "sub_1km_thresholds_km": SUB_1KM_RADII_KM,
        "distance_bands_km": DISTANCE_BANDS_KM,
        "processed_dir": str(paths.processed),
        "output_dir": str(out_dir),
        "sensor_filter": calib.PRIMARY_SENSOR_TYPE,
        "pm_daily_threshold_min_hours": calib.MIN_HOURS_PER_DAY,
        "weather_daily_threshold_min_hours": calib.MIN_HOURS_PER_DAY,
        "weather_radius_km": args.weather_radius_km,
        "coordinate_diagnostics_within_100m": coordinate_summary,
        "notes": [
            "EDA only; no calibration model fit.",
            "Nearest SDS011-UBA links match the current regression-branch distance policy.",
            "All-pair counts are also written for feasibility context.",
        ],
    }
    write_json(out_dir / "metadata.json", metadata)
    write_summary(
        out_dir,
        sub_1km_summary,
        distance_band_summary,
        pairs_250m,
        coordinate_summary,
        static_summary,
        daily_summary,
        weather_static,
        weather_daily_summary,
        inventory,
        station_type,
        months,
    )

    print("\nClose nearest-link summary:")
    print(static_summary.to_string(index=False))
    print("\nDaily pair summary:")
    print(daily_summary.to_string(index=False))
    print("\nWeather match summary:")
    print(weather_static.to_string(index=False))
    print("\nDaily weather covariate coverage:")
    print(weather_daily_summary.to_string(index=False))
    print("\nSub-1 km cumulative summary:")
    print(sub_1km_summary.to_string(index=False))
    print("\nDistance-band summary:")
    print(distance_band_summary.to_string(index=False))
    print("\nCoordinate diagnostics within 100 m:")
    print(json.dumps(coordinate_summary, indent=2, sort_keys=True))
    print(f"\nWrote EDA outputs -> {out_dir}")


if __name__ == "__main__":
    main()
