#!/usr/bin/env python3
"""
Download HYRAS-DE 2024 daily temperature/RH and extract nearest cells for SDS011.

This is an acquisition/validation pass only. It does not fit PM calibration
models or write corrected labels.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer


BASE_DIR = Path(__file__).resolve().parent.parent
REGRESSION_DIR = BASE_DIR / "03_scripts_calibration" / "experiments" / "nearby_reference_regression"
sys.path.insert(0, str(REGRESSION_DIR))

import calibrate_pm_regression_loo as calib  # noqa: E402
import eda_close_reference_inventory as eda  # noqa: E402


HYRAS_VERSION = "HYRAS-DE v6-1"
HYRAS_URLS = {
    "temperature": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/tas_hyras_1_2024_v6-1_de.nc",
    "rh": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/humidity/hurs_hyras_1_2024_v6-1_de.nc",
}
UBA_BASE = "https://luftdaten.umweltbundesamt.de/api-proxy"
MONTHS_2024 = [f"2024-{month:02d}" for month in range(1, 13)]
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/extract HYRAS-DE 2024 weather and validate against local weather."
    )
    parser.add_argument("--processed-dir", default=str(BASE_DIR / "data" / "processed"))
    parser.add_argument("--raw-dir", default=str(BASE_DIR / "data" / "raw" / "dwd" / "hyras" / "2024"))
    parser.add_argument("--output-parquet", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def download_file(url: str, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return "reused"
    part = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part.replace(out)
    return "downloaded"


def open_dataset(path: Path) -> xr.Dataset:
    for kwargs in ({}, {"engine": "h5netcdf"}, {"engine": "scipy"}):
        try:
            return xr.open_dataset(path, **kwargs)
        except Exception:
            continue
    return xr.open_dataset(path)


def pick_time_coord(ds: xr.Dataset) -> str:
    for name in ds.coords:
        if np.issubdtype(ds[name].dtype, np.datetime64):
            return name
    for name in ds.dims:
        if name.lower() == "time":
            return name
    raise ValueError("Could not identify HYRAS time coordinate")


def pick_xy_coords(ds: xr.Dataset) -> tuple[str, str]:
    names = list(ds.coords) + list(ds.dims)
    x_candidates = [n for n in names if n.lower() in {"x", "easting", "rlon"}]
    y_candidates = [n for n in names if n.lower() in {"y", "northing", "rlat"}]
    if not x_candidates or not y_candidates:
        raise ValueError(f"Could not identify HYRAS x/y coordinates from {names}")
    return x_candidates[0], y_candidates[0]


def pick_data_var(ds: xr.Dataset, family: str) -> str:
    preferred = {"temperature": ("tas", "temperature"), "rh": ("hurs", "humidity")}
    time = pick_time_coord(ds)
    for name, arr in ds.data_vars.items():
        dims = set(arr.dims)
        lname = name.lower()
        if time in dims and any(token in lname for token in preferred[family]):
            return name
    for name, arr in ds.data_vars.items():
        if time in arr.dims and arr.ndim >= 3:
            return name
    raise ValueError(f"Could not identify HYRAS data variable for {family}")


def validate_netcdf(path: Path, family: str) -> dict[str, object]:
    with open_dataset(path) as ds:
        var = pick_data_var(ds, family)
        time = pick_time_coord(ds)
        x_name, y_name = pick_xy_coords(ds)
        dates = pd.to_datetime(ds[time].values)
        dates_2024 = dates[(dates >= "2024-01-01") & (dates <= "2024-12-31")]
        expected_days = 366 if calendar.isleap(2024) else 365
        if len(dates_2024) != expected_days:
            raise ValueError(
                f"{path.name} has {len(dates_2024)} 2024 dates, expected {expected_days}"
            )
        units = str(ds[var].attrs.get("units", ""))
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "variable": var,
            "time_coord": time,
            "x_coord": x_name,
            "y_coord": y_name,
            "units": units,
            "date_count_2024": len(dates_2024),
            "expected_dates_2024": expected_days,
            "dims": json.dumps({k: int(v) for k, v in ds.sizes.items()}, sort_keys=True),
        }


def load_sds011_nodes(processed: Path) -> pd.DataFrame:
    node_dir = processed / "hourly" / "pm" / "nodes"
    parts = []
    for month in MONTHS_2024:
        path = node_dir / f"sds011_{month}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path, columns=["location", "lat", "lon"]))
    if not parts:
        raise FileNotFoundError(f"No SDS011 node files found under {node_dir}")
    nodes = pd.concat(parts, ignore_index=True)
    nodes = (
        nodes.groupby("location", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"))
    )
    sensor_land = pd.read_csv(processed / "sensor_land.csv")
    sensors = nodes.merge(sensor_land[["location", "land"]], on="location", how="inner")
    sensors = sensors.dropna(subset=["land", "lat", "lon"])
    mask = sensors["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"]) & sensors[
        "lon"
    ].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])
    return sensors.loc[mask].copy()


def extract_nearest_hyras(nc_paths: dict[str, Path], sensors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_sensor, y_sensor = transformer.transform(sensors["lon"].to_numpy(), sensors["lat"].to_numpy())
    loc_index = pd.Index(sensors["location"].astype(int), name="location")
    pieces = []
    grid_info: dict[str, object] = {}

    for family, path in nc_paths.items():
        ds = open_dataset(path)
        var = pick_data_var(ds, family)
        time = pick_time_coord(ds)
        x_name, y_name = pick_xy_coords(ds)
        selected = ds[var].sel(
            {
                x_name: xr.DataArray(x_sensor, dims="location", coords={"location": loc_index}),
                y_name: xr.DataArray(y_sensor, dims="location", coords={"location": loc_index}),
            },
            method="nearest",
        )
        selected = selected.transpose(time, "location")
        try:
            stacked = selected.to_pandas().stack(future_stack=True)
        except TypeError:
            stacked = selected.to_pandas().stack(dropna=False)
        frame = stacked.rename(family).reset_index()
        frame = frame.rename(columns={time: "date"})
        grid_x = selected[x_name].to_pandas()
        grid_y = selected[y_name].to_pandas()
        if isinstance(grid_x, pd.DataFrame):
            grid_x = grid_x.iloc[0]
        if isinstance(grid_y, pd.DataFrame):
            grid_y = grid_y.iloc[:, 0]
        grid = pd.DataFrame(
            {
                "location": loc_index.to_numpy(),
                "hyras_grid_x": np.asarray(grid_x, dtype="float64"),
                "hyras_grid_y": np.asarray(grid_y, dtype="float64"),
            }
        )
        pieces.append(frame.merge(grid, on="location", how="left"))
        grid_info[family] = {
            "variable": var,
            "x_coord": x_name,
            "y_coord": y_name,
            "units": str(ds[var].attrs.get("units", "")),
        }
        ds.close()

    out = pieces[0].merge(
        pieces[1][["date", "location", "rh", "hyras_grid_x", "hyras_grid_y"]],
        on=["date", "location"],
        suffixes=("_temperature", "_rh"),
    )
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.rename(columns={"temperature": "hyras_temperature_c", "rh": "hyras_rh_pct"})
    if "hyras_grid_x_temperature" in out.columns:
        out["hyras_grid_x"] = out["hyras_grid_x_temperature"]
        out["hyras_grid_y"] = out["hyras_grid_y_temperature"]
    sensor_xy = pd.DataFrame(
        {"location": loc_index.to_numpy(), "sensor_x": x_sensor, "sensor_y": y_sensor}
    )
    out = out.merge(sensor_xy, on="location", how="left")
    out["hyras_grid_distance_m"] = np.sqrt(
        (out["hyras_grid_x"] - out["sensor_x"]) ** 2
        + (out["hyras_grid_y"] - out["sensor_y"]) ** 2
    )
    out["hyras_version"] = HYRAS_VERSION
    keep = [
        "location",
        "date",
        "hyras_temperature_c",
        "hyras_rh_pct",
        "hyras_grid_x",
        "hyras_grid_y",
        "hyras_grid_distance_m",
        "hyras_version",
    ]
    return out[keep], grid_info


def daily_local_weather(processed: Path, locations: set[int]) -> pd.DataFrame:
    weather = eda.load_daily_humidity(processed, MONTHS_2024, locations)
    if weather.empty:
        return weather
    node_parts = []
    for sensor_type in ("bme280", "dht22"):
        for month in MONTHS_2024:
            path = processed / "hourly" / "humidity" / "nodes" / f"{sensor_type}_{month}.parquet"
            if path.exists():
                part = pd.read_parquet(path, columns=["location"])
                part["weather_sensor_type"] = sensor_type
                node_parts.append(part)
    node_types = (
        pd.concat(node_parts, ignore_index=True)
        .drop_duplicates()
        .groupby("location", as_index=False)["weather_sensor_type"]
        .agg(lambda s: ",".join(sorted(set(s))))
    )
    return weather.merge(node_types, on="location", how="left")


def metric_row(group: pd.DataFrame, prefix: str) -> dict[str, float]:
    x = pd.to_numeric(group[f"hyras_{prefix}"], errors="coerce")
    y = pd.to_numeric(group[prefix], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if valid.empty:
        return {"matched_days": 0}
    err = valid["y"] - valid["x"]
    return {
        "matched_days": int(len(valid)),
        "pearson": float(valid["x"].corr(valid["y"], method="pearson")) if len(valid) > 1 else np.nan,
        "spearman": float(valid["x"].corr(valid["y"], method="spearman")) if len(valid) > 1 else np.nan,
        "bias_local_minus_hyras": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
    }


def compare_hyras_local(hyras: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    merged = hyras.merge(local, on=["location", "date"], how="inner")
    merged = merged.rename(columns={"hyras_temperature_c": "hyras_temperature", "hyras_rh_pct": "hyras_humidity"})
    rows = []
    for sensor_type, group in merged.groupby("weather_sensor_type", dropna=False):
        for subset_name, subset in {
            "all": group,
            "exclude_saturated_streams": group[
                group.groupby("location")["humidity"].transform(lambda s: (s >= 99.0).mean()) < 0.20
            ],
        }.items():
            for metric_name, prefix in (("temperature", "temperature"), ("relative_humidity", "humidity")):
                row = {
                    "result_type": "hyras_vs_local_weather",
                    "weather_sensor_type": sensor_type,
                    "subset": subset_name,
                    "metric": metric_name,
                    "matched_sensors": int(subset["location"].nunique()),
                    "rh_ge_90_pct": float((subset["humidity"] >= 90).mean() * 100) if len(subset) else np.nan,
                    "rh_ge_95_pct": float((subset["humidity"] >= 95).mean() * 100) if len(subset) else np.nan,
                    "apparent_saturation_pct": float((subset["humidity"] >= 99).mean() * 100) if len(subset) else np.nan,
                }
                row.update(metric_row(subset, prefix))
                rows.append(row)
        group = group.copy()
        group["month"] = pd.to_datetime(group["date"]).dt.strftime("%Y-%m")
        for month, month_group in group.groupby("month"):
            for metric_name, prefix in (("temperature", "temperature"), ("relative_humidity", "humidity")):
                row = {
                    "result_type": "hyras_vs_local_weather",
                    "weather_sensor_type": sensor_type,
                    "subset": f"month:{month}",
                    "metric": metric_name,
                    "matched_sensors": int(month_group["location"].nunique()),
                    "rh_ge_90_pct": float((month_group["humidity"] >= 90).mean() * 100),
                    "rh_ge_95_pct": float((month_group["humidity"] >= 95).mean() * 100),
                    "apparent_saturation_pct": float((month_group["humidity"] >= 99).mean() * 100),
                }
                row.update(metric_row(month_group, prefix))
                rows.append(row)
    return pd.DataFrame(rows)


def close_pair_station_type_inventory(processed: Path, station_metadata_path: Path) -> pd.DataFrame:
    links_path = processed / "calibration" / "regression_reference_adjustment" / "eda_close_reference_inventory" / "close_links_within_max_radius.csv"
    if not links_path.exists() or not station_metadata_path.exists():
        return pd.DataFrame()
    links = pd.read_csv(links_path)
    metadata = pd.read_csv(station_metadata_path)
    paths = calib.Paths(processed=processed)
    months = calib.discover_months(paths)
    sensor_land = calib.load_sensor_land(paths)
    nodes = calib.load_nodes(paths, months, sensor_land)
    uba = calib.load_uba(paths, 2024)
    daily_pm = calib.load_daily_sds011(paths, months)
    rows = []
    for radius in (0.25, 0.5, 1.0):
        close = links[(links["dist_km"] <= radius) & links["is_nearest_station"]].merge(
            metadata, on="station_code", how="left"
        )
        for pollutant in ("PM10", "PM2.5"):
            pairs = calib.make_pairs(daily_pm, nodes, uba, radius, pollutant).merge(
                metadata[["station_code", "station_type_label", "station_setting_label"]],
                on="station_code",
                how="left",
            )
            available = close[close[f"{pollutant}_available_2024"].fillna(False)] if f"{pollutant}_available_2024" in close else close
            for typ, group in available.groupby("station_type_label", dropna=False):
                settings = ";".join(sorted(group["station_setting_label"].dropna().astype(str).unique()))
                pair_group = pairs[pairs["station_type_label"].fillna("unknown").eq(typ if pd.notna(typ) else "unknown")]
                rows.append(
                    {
                        "result_type": "uba_station_type_inventory",
                        "radius_km": radius,
                        "pollutant": pollutant,
                        "station_type_label": typ if pd.notna(typ) else "unknown",
                        "station_settings": settings,
                        "close_sds011_sensors": int(group["location"].nunique()),
                        "independent_uba_stations": int(group["station_code"].nunique()),
                        "paired_days": int(len(pair_group)),
                        "unknown_classifications": int(group["station_type_label"].isna().sum()),
                        "traffic_station_count": int(group["station_type_label"].astype(str).str.contains("verkehr", case=False, na=False).sum()),
                        "background_station_count": int(group["station_type_label"].astype(str).str.contains("hintergrund", case=False, na=False).sum()),
                        "industrial_station_count": int(group["station_type_label"].astype(str).str.contains("industrie", case=False, na=False).sum()),
                    }
                )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame) -> str:
    """Tiny markdown table writer to avoid depending on tabulate."""

    if frame.empty:
        return ""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            out[col] = out[col].fillna("")
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def count_json_values(value: object) -> int:
    if isinstance(value, list):
        if value and all(not isinstance(item, (list, dict)) for item in value):
            return 1
        return sum(count_json_values(item) for item in value)
    if isinstance(value, dict):
        return sum(count_json_values(item) for item in value.values())
    return 0


def hourly_uba_check(processed: Path, station_metadata_path: Path) -> pd.DataFrame:
    if not station_metadata_path.exists():
        return pd.DataFrame()
    meta = pd.read_csv(station_metadata_path)
    close_path = processed / "calibration" / "regression_reference_adjustment" / "eda_close_reference_inventory" / "close_links_within_max_radius.csv"
    if close_path.exists():
        close_codes = pd.read_csv(close_path).query("dist_km <= 0.5 and is_nearest_station")[
            "station_code"
        ].drop_duplicates()
        candidates = (
            meta[meta["station_code"].isin(close_codes) & meta["PM2.5_available_2024"].fillna(False)][
                "station_code"
            ]
            .dropna()
            .head(2)
        )
    else:
        candidates = meta[meta.get("PM2.5_available_2024", False).fillna(False)]["station_code"].dropna().head(2)
    rows = []
    for station in candidates:
        for pollutant, component in {"PM10": 1, "PM2.5": 9}.items():
            url = (
                f"{UBA_BASE}/measures/json?date_from=2024-01-01&date_to=2024-01-02"
                f"&time_from=1&time_to=24&station={station}&component={component}&scope=2&lang=de"
            )
            row = {"result_type": "hourly_uba_availability_check", "station_code": station, "pollutant": pollutant, "url": url}
            try:
                response = requests.get(url, timeout=60)
                row["status_code"] = response.status_code
                payload = response.json()
                row["returned_values"] = count_json_values(payload.get("data", {})) if isinstance(payload, dict) else 0
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    results: pd.DataFrame,
    netcdf_info: list[dict[str, object]],
    hyras: pd.DataFrame,
    urls: dict[str, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    weather = results[results["result_type"].eq("hyras_vs_local_weather")] if not results.empty else pd.DataFrame()
    station_types = results[results["result_type"].eq("uba_station_type_inventory")] if not results.empty else pd.DataFrame()
    hourly = results[results["result_type"].eq("hourly_uba_availability_check")] if not results.empty else pd.DataFrame()
    lines = [
        "# External Weather And UBA Metadata Pass",
        "",
        "This pass downloads HYRAS-DE gridded ambient weather and UBA v4 station metadata. It does not fit calibration models or write corrected labels.",
        "",
        "## Retrieved URLs",
        *(f"- {key}: {url}" for key, url in urls.items()),
        "",
        "## HYRAS Extraction",
        f"- SDS011 locations extracted: {hyras['location'].nunique():,}",
        f"- dates: {hyras['date'].nunique():,}",
        f"- missing HYRAS values: {int(hyras[['hyras_temperature_c', 'hyras_rh_pct']].isna().sum().sum()):,}",
        f"- median grid distance: {hyras['hyras_grid_distance_m'].median():.1f} m",
        f"- max grid distance: {hyras['hyras_grid_distance_m'].max():.1f} m",
        "",
        "## NetCDF Verification",
    ]
    for info in netcdf_info:
        lines.append(
            f"- {Path(str(info['path'])).name}: {info['bytes']:,} bytes, variable `{info['variable']}`, units `{info['units']}`, dates {info['date_count_2024']}, dims {info['dims']}"
        )
    lines += ["", "## HYRAS Versus Local Weather"]
    if weather.empty:
        lines.append("No matched local weather rows found.")
    else:
        view = weather[weather["subset"].isin(["all", "exclude_saturated_streams"])]
        view = view[
            [
                "weather_sensor_type",
                "subset",
                "metric",
                "matched_sensors",
                "matched_days",
                "pearson",
                "spearman",
                "bias_local_minus_hyras",
                "mae",
                "rmse",
                "rh_ge_90_pct",
                "rh_ge_95_pct",
                "apparent_saturation_pct",
            ]
        ]
        lines.append(md_table(view))
    lines += ["", "## UBA Station Type Inventory"]
    if station_types.empty:
        lines.append("No station-type inventory rows available.")
    else:
        station_types = station_types[
            [
                "radius_km",
                "pollutant",
                "station_type_label",
                "station_settings",
                "close_sds011_sensors",
                "independent_uba_stations",
                "paired_days",
                "unknown_classifications",
            ]
        ]
        lines.append(md_table(station_types))
    lines += ["", "## Hourly UBA Availability Check"]
    if hourly.empty:
        lines.append("No hourly availability check rows available.")
    else:
        hourly = hourly[["station_code", "pollutant", "status_code", "returned_values", "url"]]
        lines.append(md_table(hourly))
    lines += [
        "",
        "## Interpretation",
        "- HYRAS is a consistent gridded ambient-weather product, not device-level truth.",
        "- It is a credible replacement for saturated DHT22 humidity streams for calibration sensitivity work.",
        "- A calibration rerun is justified after adding a HYRAS-weather option and a DHT22-saturation sensitivity filter.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    out_parquet = Path(args.output_parquet) if args.output_parquet else processed / "daily_weather" / "hyras_2024_sds011.parquet"
    report_dir = Path(args.report_dir) if args.report_dir else processed / "calibration" / "regression_reference_adjustment" / "external_weather_metadata"
    station_metadata = processed / "uba" / "station_metadata.csv"

    nc_paths = {key: raw_dir / Path(urlparse(url).path).name for key, url in HYRAS_URLS.items()}
    urls = dict(HYRAS_URLS)
    if not args.skip_download:
        for key, url in HYRAS_URLS.items():
            print(f"{key}: {download_file(url, nc_paths[key])} {nc_paths[key]}")
    netcdf_info = [validate_netcdf(nc_paths[key], key) for key in ("temperature", "rh")]

    sensors = load_sds011_nodes(processed)
    hyras, _ = extract_nearest_hyras(nc_paths, sensors)
    expected_days = 366 if calendar.isleap(2024) else 365
    if hyras["date"].nunique() != expected_days:
        raise ValueError(f"Expected {expected_days} HYRAS dates, got {hyras['date'].nunique()}")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    hyras.to_parquet(out_parquet, index=False)
    print(f"Saved {len(hyras):,} HYRAS sensor-days -> {out_parquet}")

    local = daily_local_weather(processed, set(sensors["location"].astype(int)))
    result_parts = [compare_hyras_local(hyras, local)]
    result_parts.append(close_pair_station_type_inventory(processed, station_metadata))
    result_parts.append(hourly_uba_check(processed, station_metadata))
    results = pd.concat([p for p in result_parts if not p.empty], ignore_index=True, sort=False)
    results_path = report_dir / "results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    urls["UBA measures hourly scope 2 check"] = f"{UBA_BASE}/measures/json"
    write_report(report_dir / "REPORT.md", results, netcdf_info, hyras, urls)
    print(f"Saved report -> {report_dir / 'REPORT.md'}")
    print(f"Saved results -> {results_path}")


if __name__ == "__main__":
    main()
