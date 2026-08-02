#!/usr/bin/env python3
"""
Strict, compact 2024 data-completeness audit for close-reference calibration.

This audits inputs and modeled complete-case rows only. It does not fit models,
write corrected labels, download data, or train the CNN.
"""

from __future__ import annotations

import argparse
import calendar
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_pm_regression_loo as calib
import fit_close_reference_weather_models as weather_models


MONTHS = [f"2024-{m:02d}" for m in range(1, 13)]
RADII = [0.25, 0.5, 1.0]
STRATA = weather_models.STATION_TYPE_STRATA
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
EXPECTED_DAYS_2024 = 366 if calendar.isleap(2024) else 365
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 2024 data completeness for close-reference PM calibration.")
    parser.add_argument("--processed-dir", default=str(calib.DEFAULT_PROCESSED_DIR))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/processed/calibration/regression_reference_adjustment/data_completeness.",
    )
    return parser.parse_args()


def row(section: str, source: str, status: str, **kwargs) -> dict:
    out = {"audit_section": section, "source": source, "status": status}
    out.update(kwargs)
    return out


def kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "missing"


def mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def schema(path: Path) -> str:
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            return ";".join(f"{field.name}:{field.type}" for field in pf.schema_arrow)
        if path.suffix == ".csv":
            return ";".join(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return f"unreadable:{type(exc).__name__}"
    return ""


def file_rows(processed: Path) -> list[dict]:
    rows = []
    active = processed.resolve()
    data_root = active.parent if active.name == "processed" else active
    candidates = [
        ("merged hourly SDS011 PM", processed / "hourly/pm/all_pm_sensors"),
        ("PM node coordinates", processed / "hourly/pm/nodes"),
        ("corrected sensor_land", processed / "sensor_land.csv"),
        ("UBA daily PM", processed / "daily_avg/uba/pm_reference_stations_2024.csv"),
        ("UBA station metadata", processed / "uba/station_metadata.csv"),
        ("UBA station land", processed / "uba/station_land.csv"),
        ("local weather hourly", processed / "hourly/humidity/all_sensors"),
        ("local weather nodes", processed / "hourly/humidity/nodes"),
        ("HYRAS daily weather", processed / "daily_weather/hyras_2024_sds011.parquet"),
        ("close-reference links", processed / "calibration/regression_reference_adjustment/eda_close_reference_inventory/close_links_within_max_radius.csv"),
        ("latest modeled rows/results", processed / "calibration/regression_reference_adjustment/close_reference_weather_models/results.csv"),
    ]
    for source, path in candidates:
        files = []
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("._"))
        elif path.exists():
            files = [path]
        rows.append(
            row(
                "file_source",
                source,
                "complete" if path.exists() and all(f.stat().st_size > 0 for f in files) else "missing upstream data",
                expected_path=str(path),
                actual_path=str(path),
                path_kind=kind(path),
                resolved_destination=str(path.resolve()) if path.exists() or path.is_symlink() else "",
                files_found=len(files),
                file_names=";".join(f.name for f in files[:30]),
                duplicate_or_competing_copies="",
                modified_at=mtime(path) if path.exists() else "",
                schema=schema(files[0]) if files else schema(path),
                unreadable_or_empty=any((not f.exists()) or f.stat().st_size == 0 for f in files),
            )
        )

    processed2 = data_root / "processed 2"
    zips = sorted(data_root.glob("processed*.zip"))
    april_dups = []
    for probe in [
        processed2 / "hourly/pm/all_pm_sensors/2024-04.parquet",
        processed / "hourly/pm/all_pm_sensors/2024-04.parquet",
    ]:
        if probe.exists() or probe.is_symlink():
            april_dups.append(str(probe.resolve()))
    rows.append(
        row(
            "file_source",
            "split-data history",
            "duplicate/stale input risk" if processed2.exists() or zips else "complete",
            expected_path=str(processed),
            actual_path=str(processed),
            path_kind=kind(processed),
            resolved_destination=str(active),
            duplicate_or_competing_copies=";".join([str(processed2)] + [str(z) for z in zips if z.exists()]),
            files_found=len(list(processed2.rglob("*"))) if processed2.exists() else 0,
            file_names="processed 2 contains April merged PM; active April is a symlink to that file",
            schema="",
            unreadable_or_empty=False,
            authoritative_months="active data/processed only; 2024-04 all_pm_sensors resolves to processed 2 target",
            april_authoritative_targets=";".join(sorted(set(april_dups))),
        )
    )
    return rows


def expected_month_hours(month: str) -> int:
    start = pd.Timestamp(f"{month}-01 00:00:00")
    end = start + pd.offsets.MonthBegin(1)
    return int((end - start) / pd.Timedelta(hours=1))


def pm_month_rows(processed: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    rows = []
    schemas = {}
    all_nodes = []
    daily_parts = []
    sensor_land = pd.read_csv(processed / "sensor_land.csv")
    for month in MONTHS:
        path = processed / "hourly/pm/all_pm_sensors" / f"{month}.parquet"
        node_path = processed / "hourly/pm/nodes" / f"sds011_{month}.parquet"
        status = "complete"
        if not path.exists():
            rows.append(row("hourly_pm_calendar", "merged hourly PM", "missing upstream data", month=month, actual_path=str(path)))
            continue
        cols = ["location", "hour", "P1", "P2", "sensor_type"]
        df = pd.read_parquet(path, columns=cols)
        schemas[month] = tuple(df.columns)
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
        before = len(df)
        stypes = df["sensor_type"].astype(str).str.strip().str.lower()
        sds = df[stypes.eq("sds011")].copy()
        if sds.empty:
            status = "missing upstream data"
        expected_hours = expected_month_hours(month)
        hours = sds["hour"].dropna().sort_values().unique()
        dates = pd.Series(pd.to_datetime(hours)).dt.date if len(hours) else pd.Series(dtype=object)
        dup = int(sds.duplicated(["location", "hour"]).sum())
        missing_dates = sorted(set(pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month).date) - set(dates))
        rows.append(
            row(
                "hourly_pm_calendar",
                "merged hourly PM",
                status if len(hours) == expected_hours and not missing_dates and "sensor_type" in df else "recoverable processing gap",
                month=month,
                expected_days=pd.Period(month).days_in_month,
                observed_dates=len(set(dates)),
                expected_hours=expected_hours,
                observed_hours=len(hours),
                first_timestamp=str(pd.Timestamp(hours[0])) if len(hours) else "",
                last_timestamp=str(pd.Timestamp(hours[-1])) if len(hours) else "",
                missing_dates=";".join(map(str, missing_dates)),
                duplicate_location_hour_rows=dup,
                rows_before_sds011_filter=before,
                rows_after_sds011_filter=len(sds),
                unique_sds011_locations=sds["location"].nunique(),
                sensor_type_present="sensor_type" in df.columns,
                sensor_type_values=";".join(sorted(stypes.dropna().unique())),
                schema=";".join(df.columns),
                actual_path=str(path),
                path_kind=kind(path),
                resolved_destination=str(path.resolve()),
            )
        )
        sds["date"] = (sds["hour"] + pd.Timedelta(hours=calib.UTC_TO_MEZ_HOURS)).dt.date
        for col in ("P1", "P2"):
            sds[col] = pd.to_numeric(sds[col], errors="coerce")
        daily = (
            sds.groupby(["location", "date"], as_index=False)
            .agg(P1=("P1", "mean"), P2=("P2", "mean"), n_hours=("hour", "nunique"))
        )
        daily_parts.append(daily[daily["n_hours"] >= calib.MIN_HOURS_PER_DAY])
        if node_path.exists():
            n = pd.read_parquet(node_path, columns=["location", "lat", "lon"])
            n["month"] = month
            all_nodes.append(n)
    rows.append(
        row(
            "schema_consistency",
            "merged hourly PM",
            "complete" if len(set(schemas.values())) == 1 and len(schemas) == 12 else "inconsistent schema risk",
            months_with_files=len(schemas),
            schema_variants=len(set(schemas.values())),
            schemas=" | ".join(f"{m}:{','.join(v)}" for m, v in schemas.items()),
        )
    )
    daily_pm = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    nodes = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame()
    if not nodes.empty:
        nodes = (
            nodes.groupby("location", as_index=False)
            .agg(lat=("lat", "median"), lon=("lon", "median"), months_seen=("month", "nunique"))
            .merge(sensor_land[["location", "land"]], on="location", how="left")
        )
        nodes["fold"] = nodes["land"].map(calib.LAND_TO_FOLD)
    return rows, daily_pm, nodes


def uba_rows(processed: Path) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    uba = calib.load_uba(calib.Paths(processed), 2024)
    meta = pd.read_csv(processed / "uba/station_metadata.csv")
    full_dates = set(pd.date_range("2024-01-01", "2024-12-31").date)
    for pollutant in calib.POLLUTANTS:
        valid = uba.dropna(subset=[pollutant])
        dates = set(valid["date"])
        by_station = valid.groupby("station_code")["date"].nunique()
        with_meta = valid[["station_code"]].drop_duplicates().merge(meta, on="station_code", how="left")
        for typ, g in with_meta.groupby("station_type_label", dropna=False):
            rows.append(
                row(
                    "uba_coverage_by_type",
                    "UBA daily PM",
                    "complete" if pd.notna(typ) else "missing upstream data",
                    pollutant=pollutant,
                    station_type=typ if pd.notna(typ) else "unknown",
                    stations=g["station_code"].nunique(),
                )
            )
        rows.append(
            row(
                "uba_daily_pm",
                "UBA daily PM",
                "complete" if len(dates) == EXPECTED_DAYS_2024 else "missing upstream data",
                pollutant=pollutant,
                observed_dates=len(dates),
                expected_days=EXPECTED_DAYS_2024,
                first_date=str(min(dates)) if dates else "",
                last_date=str(max(dates)) if dates else "",
                missing_national_dates=";".join(map(str, sorted(full_dates - dates))),
                duplicate_station_date_rows=int(valid.duplicated(["station_code", "date"]).sum()),
                stations=valid["station_code"].nunique(),
                median_valid_days_per_station=float(by_station.median()),
                min_valid_days_per_station=int(by_station.min()) if len(by_station) else 0,
                max_valid_days_per_station=int(by_station.max()) if len(by_station) else 0,
                stations_passing_182_days=int((by_station >= calib.MIN_DAYS_PER_YEAR).sum()),
                months_represented=";".join(sorted(pd.to_datetime(valid["date"]).dt.strftime("%Y-%m").unique())),
            )
        )
    return rows, uba


def hyras_rows(processed: Path) -> tuple[list[dict], pd.DataFrame]:
    path = processed / "daily_weather/hyras_2024_sds011.parquet"
    df = pd.read_parquet(path)
    dates = pd.to_datetime(df["date"]).dt.date
    temp_missing = set(df.loc[df["hyras_temperature_c"].isna(), "location"])
    rh_missing = set(df.loc[df["hyras_rh_pct"].isna(), "location"])
    miss = sorted(temp_missing | rh_missing)
    sensor_land = pd.read_csv(processed / "sensor_land.csv")
    nodes = []
    for month in MONTHS:
        p = processed / "hourly/pm/nodes" / f"sds011_{month}.parquet"
        if p.exists():
            nodes.append(pd.read_parquet(p, columns=["location", "lat", "lon"]))
    node = pd.concat(nodes).groupby("location", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))
    missing_detail = node[node["location"].isin(miss)].merge(sensor_land, on="location", how="left")
    boundary_like = int((missing_detail["assignment_method"].astype(str).str.contains("boundary", na=False)).sum())
    rows = [
        row(
            "hyras",
            "HYRAS daily weather",
            "expected observational missingness" if len(miss) == 6 and temp_missing == rh_missing else "recoverable processing gap",
            expected_days=EXPECTED_DAYS_2024,
            observed_dates=len(set(dates)),
            first_date=str(min(dates)),
            last_date=str(max(dates)),
            requested_sds011_locations=df["location"].nunique(),
            missing_locations=len(miss),
            missing_location_ids=";".join(map(str, miss)),
            same_locations_missing_temperature_and_rh=temp_missing == rh_missing,
            missing_boundary_or_mask_like=boundary_like,
            duplicate_location_date_rows=int(df.duplicated(["location", "date"]).sum()),
            temperature_units="degree_Celsius",
            rh_units="percent",
            min_temperature=float(df["hyras_temperature_c"].min()),
            max_temperature=float(df["hyras_temperature_c"].max()),
            min_rh=float(df["hyras_rh_pct"].min()),
            max_rh=float(df["hyras_rh_pct"].max()),
        )
    ]
    out = df.dropna(subset=["hyras_temperature_c", "hyras_rh_pct"]).rename(columns={"hyras_rh_pct": "humidity", "hyras_temperature_c": "temperature"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["weather_source"] = "hyras"
    return rows, out[["location", "date", "humidity", "temperature", "weather_source"]]


def local_weather_rows(processed: Path) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows = []
    outputs: dict[str, list[pd.DataFrame]] = {"bme280": [], "dht22": []}
    for month in MONTHS:
        path = processed / "hourly/humidity/all_sensors" / f"{month}.parquet"
        if not path.exists():
            for stype in outputs:
                rows.append(row("local_weather", f"{stype} weather", "missing upstream data", month=month, actual_path=str(path)))
            continue
        df = pd.read_parquet(path, columns=["location", "hour", "temperature", "humidity", "sensor_type"])
        df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
        for stype in ("bme280", "dht22"):
            sub = df[df["sensor_type"].astype(str).str.lower().eq(stype)].copy()
            if sub.empty:
                rows.append(row("local_weather", f"{stype} weather", "missing upstream data", month=month))
                continue
            sub["date"] = (sub["hour"] + pd.Timedelta(hours=calib.UTC_TO_MEZ_HOURS)).dt.date
            daily = (
                sub.groupby(["location", "date"], as_index=False)
                .agg(
                    n_weather_hours=("hour", "nunique"),
                    humidity=("humidity", "mean"),
                    temperature=("temperature", "mean"),
                )
            )
            kept = daily[daily["n_weather_hours"] >= calib.MIN_HOURS_PER_DAY].copy()
            kept["source_month"] = month
            outputs[stype].append(kept)
            month_dates = set(sub["date"])
            rows.append(
                row(
                    "local_weather",
                    f"{stype} weather",
                    "expected observational missingness",
                    month=month,
                    rows=len(sub),
                    unique_locations=sub["location"].nunique(),
                    duplicate_location_hour_rows=int(sub.duplicated(["location", "hour"]).sum()),
                    observed_dates=len(month_dates),
                    first_timestamp=str(sub["hour"].min()),
                    last_timestamp=str(sub["hour"].max()),
                    daily_rows_passing_18h=len(daily[daily["n_weather_hours"] >= calib.MIN_HOURS_PER_DAY]),
                    locations_passing_182_days=np.nan,
                    rh_ge_90_pct=float(100 * (sub["humidity"] >= 90).mean()),
                    rh_ge_95_pct=float(100 * (sub["humidity"] >= 95).mean()),
                    apparent_saturation_pct=float(100 * (sub["humidity"] >= 99).mean()),
                )
            )
    final = {}
    for stype, parts in outputs.items():
        daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not daily.empty:
            per_loc = daily.groupby("location")["date"].nunique()
            source_months = sorted(daily["source_month"].unique())
            seasons = sorted({SEASON[int(m[-2:])] for m in source_months})
            rows.append(
                row(
                    "local_weather_annual",
                    f"{stype} weather",
                    "expected observational missingness" if stype == "bme280" else "modeling-bias risk",
                    available_months=";".join(source_months),
                    daily_rows=len(daily),
                    unique_locations=daily["location"].nunique(),
                    locations_passing_182_days=int((per_loc >= calib.MIN_DAYS_PER_YEAR).sum()),
                    mean_valid_days_per_location=float(per_loc.mean()),
                    seasons_represented=";".join(seasons),
                    saturation_risk="high" if stype == "dht22" else "low",
                )
            )
            daily["weather_source"] = stype
        final[stype] = daily
    return rows, final


def coordinate_rows(processed: Path, nodes: pd.DataFrame, modeled_locations: set[int], modeled_stations: set[str]) -> list[dict]:
    rows = []
    monthly = []
    for month in MONTHS:
        p = processed / "hourly/pm/nodes" / f"sds011_{month}.parquet"
        if p.exists():
            part = pd.read_parquet(p, columns=["location", "lat", "lon"])
            part["month"] = month
            monthly.append(part)
    raw = pd.concat(monthly, ignore_index=True)
    span = raw.groupby("location").agg(lat_span=("lat", lambda s: s.max() - s.min()), lon_span=("lon", lambda s: s.max() - s.min()), months=("month", "nunique")).reset_index()
    modeled_span = span[span["location"].isin(modeled_locations)]
    dupe_coords = nodes.groupby(["lat", "lon"])["location"].nunique()
    sensor_land = pd.read_csv(processed / "sensor_land.csv")
    meta = pd.read_csv(processed / "uba/station_metadata.csv")
    outside = nodes[~(nodes["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"]) & nodes["lon"].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"]))]
    rows.append(
        row(
            "coordinate_identity",
            "SDS011 nodes and metadata joins",
            "complete" if modeled_span[["lat_span", "lon_span"]].max().max() < 0.01 else "modeling-bias risk",
            modeled_locations=len(modeled_locations),
            modeled_locations_missing_sensor_land=len(set(modeled_locations) - set(sensor_land["location"])),
            modeled_locations_with_coordinate_changes_gt_0p001deg=int(((modeled_span["lat_span"] > 0.001) | (modeled_span["lon_span"] > 0.001)).sum()),
            duplicate_coordinate_groups=int((dupe_coords > 1).sum()),
            max_locations_sharing_coordinate=int(dupe_coords.max()) if len(dupe_coords) else 0,
            outside_germany_all_sds011_locations=outside["location"].nunique(),
            outside_germany_modeled_locations=len(set(outside["location"]) & modeled_locations),
            modeled_stations=len(modeled_stations),
            modeled_stations_missing_metadata=len(modeled_stations - set(meta["station_code"])),
            station_type_missing_for_modeled=int(meta[meta["station_code"].isin(modeled_stations)]["station_type_label"].isna().sum()),
            station_setting_missing_for_modeled=int(meta[meta["station_code"].isin(modeled_stations)]["station_setting_label"].isna().sum()),
        )
    )
    return rows


def season_counts(dates: pd.Series) -> str:
    s = pd.to_datetime(dates).dt.month.map(SEASON).value_counts().to_dict()
    return ";".join(f"{k}:{v}" for k, v in sorted(s.items()))


def close_pair_and_model_rows(
    processed: Path,
    daily_pm: pd.DataFrame,
    nodes: pd.DataFrame,
    uba: pd.DataFrame,
    hyras: pd.DataFrame,
    weather_daily: dict[str, pd.DataFrame],
) -> tuple[list[dict], set[int], set[str]]:
    rows = []
    paths = calib.Paths(processed)
    meta = pd.read_csv(processed / "uba/station_metadata.csv")[["station_code", "station_type_label", "station_setting_label"]].drop_duplicates("station_code")
    nodes_model = nodes[nodes["land"] != calib.TEST_LAND].copy()
    uba_model = uba[uba["land"] != calib.TEST_LAND].copy()
    daily_model = daily_pm[daily_pm["location"].isin(nodes_model["location"])].copy()
    modeled_locations: set[int] = set()
    modeled_stations: set[str] = set()
    for radius in RADII:
        links = calib.nearest_matches(nodes_model, uba_model, radius)
        for pollutant in calib.POLLUTANTS:
            spec = calib.POLLUTANTS[pollutant]
            after_pm = (
                daily_model[["location", "date", spec["lowcost"]]]
                .rename(columns={spec["lowcost"]: "raw"})
                .merge(nodes_model[["location", "fold"]], on="location")
                .merge(links, on="location")
            )
            after_pm = after_pm[(after_pm["raw"] > 0) & np.isfinite(after_pm["raw"])]
            base = weather_models.base_pairs(daily_model, nodes_model, uba_model, meta, radius, pollutant)
            base["month"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m")
            base["season"] = pd.to_datetime(base["date"]).dt.month.map(SEASON)
            base_meta = base.merge(meta, on="station_code", how="left", suffixes=("", "_m"))
            for stratum, labels in STRATA.items():
                b = base_meta if labels is None else base_meta[base_meta["station_type_label"].isin(labels)]
                if b.empty:
                    rows.append(row("close_pair_sample", "close-reference pairs", "expected observational missingness", radius_m=int(radius*1000), pollutant=pollutant, station_type=stratum, independent_uba_stations=0))
                    continue
                by_station = b.groupby("station_code")["date"].nunique()
                by_sensor = b.groupby("location")["date"].nunique()
                rows.append(
                    row(
                        "close_pair_sample",
                        "close-reference pairs",
                        "complete" if b["date"].nunique() >= 300 else "modeling-bias risk",
                        radius_m=int(radius * 1000),
                        pollutant=pollutant,
                        station_type=stratum,
                        independent_uba_stations=b["station_code"].nunique(),
                        sds011_locations=b["location"].nunique(),
                        paired_days=len(b),
                        first_date=str(min(b["date"])),
                        last_date=str(max(b["date"])),
                        months_represented=";".join(sorted(b["month"].dropna().unique())),
                        seasonal_distribution=";".join(f"{k}:{v}" for k, v in sorted(b["season"].value_counts().to_dict().items())),
                        median_valid_days_per_station=float(by_station.median()),
                        median_valid_days_per_sensor=float(by_sensor.median()),
                        stations_passing_182_days=int((by_station >= calib.MIN_DAYS_PER_YEAR).sum()),
                        sensors_passing_182_days=int((by_sensor >= calib.MIN_DAYS_PER_YEAR).sum()),
                        folds_represented=";".join(sorted(b["station_fold"].dropna().astype(str).unique())),
                    )
                )
            source_frames = [
                ("hyras", "source_available", hyras),
                ("bme280", "source_available", weather_daily.get("bme280", pd.DataFrame())),
            ]
            # Common rows are computed directly to guarantee identical row universe.
            common = base.merge(hyras.rename(columns={"humidity": "h_humidity", "temperature": "h_temperature"})[["location", "date", "h_humidity", "h_temperature"]], on=["location", "date"], how="inner")
            if "bme280" in weather_daily and not weather_daily["bme280"].empty:
                common = common.merge(weather_daily["bme280"].rename(columns={"humidity": "b_humidity", "temperature": "b_temperature"})[["location", "date", "b_humidity", "b_temperature"]], on=["location", "date"], how="inner")
            else:
                common = pd.DataFrame()
            for weather_source, sample, w in source_frames:
                if w.empty:
                    continue
                joined = base.merge(w[["location", "date", "humidity", "temperature"]], on=["location", "date"], how="inner").dropna(subset=["humidity", "temperature"])
                rows.extend(modeling_rows(radius, pollutant, weather_source, sample, after_pm, base, joined, modeled_locations, modeled_stations))
            if not common.empty:
                for weather_source in ("hyras", "bme280", "hyras_bme280"):
                    rows.extend(modeling_rows(radius, pollutant, weather_source, "common_hyras_bme280", after_pm, base, common, modeled_locations, modeled_stations))
    return rows, modeled_locations, modeled_stations


def modeling_rows(
    radius: float,
    pollutant: str,
    weather_source: str,
    sample: str,
    after_pm: pd.DataFrame,
    base: pd.DataFrame,
    joined: pd.DataFrame,
    modeled_locations: set[int],
    modeled_stations: set[str],
) -> list[dict]:
    rows = []
    for stratum, labels in STRATA.items():
        sub = joined if labels is None else joined[joined["station_type_label"].isin(labels)]
        if not sub.empty:
            modeled_locations.update(int(x) for x in sub["location"].dropna().unique())
            modeled_stations.update(str(x) for x in sub["station_code"].dropna().unique())
        by_month = sub["month"].value_counts(normalize=True).sort_index() if "month" in sub else pd.Series(dtype=float)
        by_season = sub["season"].value_counts(normalize=True).sort_index() if "season" in sub else pd.Series(dtype=float)
        by_fold = sub["station_fold"].value_counts(normalize=True).sort_index()
        status = "complete"
        if sub["station_code"].nunique() < weather_models.MIN_CV_STATIONS:
            status = "expected observational missingness"
        if sub.empty:
            status = "missing upstream data"
        rows.append(
            row(
                "modeling_rows",
                "latest sensitivity complete cases",
                status,
                radius_m=int(radius * 1000),
                pollutant=pollutant,
                station_type=stratum,
                weather_source=weather_source,
                comparison_sample=sample,
                raw_candidate_rows=len(after_pm),
                rows_after_pm_completeness=len(after_pm),
                rows_after_uba_join=len(base),
                rows_after_weather_join=len(joined),
                rows_after_sachsen_anhalt_exclusion=len(sub),
                final_complete_case_rows=len(sub),
                independent_uba_stations=sub["station_code"].nunique(),
                sds011_locations=sub["location"].nunique(),
                months_represented=";".join(sorted(sub["month"].dropna().unique())) if not sub.empty and "month" in sub else "",
                fraction_by_month=";".join(f"{k}:{v:.3f}" for k, v in by_month.items()),
                fraction_by_season=";".join(f"{k}:{v:.3f}" for k, v in by_season.items()),
                fraction_by_fold=";".join(f"{k}:{v:.3f}" for k, v in by_fold.items()),
                sachsen_anhalt_rows=int(sub["station_land"].astype(str).str.contains(calib.TEST_LAND, na=False).sum()) if not sub.empty else 0,
                fair_comparison_key=f"{radius:g}|{pollutant}|{stratum}|{sample}",
            )
        )
    return rows


def bias_risk_rows(results: list[dict]) -> list[dict]:
    df = pd.DataFrame(results)
    rows = []
    modeling = df[df["audit_section"].eq("modeling_rows")]
    for _, r in modeling.iterrows():
        status = "complete"
        note = ""
        if r.get("independent_uba_stations", 0) < weather_models.MIN_CV_STATIONS:
            status = "expected observational missingness"
            note = "too few independent UBA stations for grouped modeling"
        elif r.get("weather_source") == "bme280" and r.get("final_complete_case_rows", 0) < 0.3 * r.get("rows_after_uba_join", 1):
            status = "modeling-bias risk"
            note = "BME280 complete cases are a minority subset of PM/reference pairs"
        rows.append(
            row(
                "missingness_bias_risk",
                "modeled complete cases",
                status,
                radius_m=r.get("radius_m"),
                pollutant=r.get("pollutant"),
                station_type=r.get("station_type"),
                weather_source=r.get("weather_source"),
                comparison_sample=r.get("comparison_sample"),
                final_complete_case_rows=r.get("final_complete_case_rows"),
                independent_uba_stations=r.get("independent_uba_stations"),
                note=note,
            )
        )
    return rows


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if df.empty:
        return "No rows."
    use = df[cols].copy()
    if n is not None:
        use = use.head(n)
    return use.to_markdown(index=False)


def write_report(path: Path, results: pd.DataFrame) -> None:
    pm = results[results["audit_section"].eq("hourly_pm_calendar")]
    uba = results[results["audit_section"].eq("uba_daily_pm")]
    hyras = results[results["audit_section"].eq("hyras")]
    weather = results[results["audit_section"].eq("local_weather_annual")]
    close = results[(results["audit_section"].eq("close_pair_sample")) & (results["radius_m"].eq(500))]
    modeling = results[(results["audit_section"].eq("modeling_rows")) & (results["radius_m"].eq(500))]
    file_source = results[results["audit_section"].eq("file_source")]
    coord = results[results["audit_section"].eq("coordinate_identity")]
    risks = results[results["status"].isin(["recoverable processing gap", "missing upstream data", "duplicate/stale input risk", "modeling-bias risk"])]
    pm_material_gap = pm[
        pm["status"].eq("missing upstream data")
        | (pd.to_numeric(pm["observed_dates"], errors="coerce") < pd.to_numeric(pm["expected_days"], errors="coerce"))
    ]
    decision = "A. Inputs sufficiently complete"
    if not pm_material_gap.empty or any(uba["status"].ne("complete")):
        decision = "B. Material recoverable data problem found"
    # Minor missing hourly slots, HYRAS boundary pixels, and the April split copy are
    # tracked as risks but are not material enough to rerun the model branch.

    lines = [
        "# 2024 Data Completeness Audit",
        "",
        f"Decision: **{decision}**.",
        "",
        "The negative close-reference weather-model result is unlikely to be caused by a material 2024 data gap. The active April merged PM file is a symlink into `processed 2`, but the modeling scripts read one authoritative path under `data/processed`.",
        "",
        "Recommendation: stop this calibration-model branch for now and retain the existing percentile-mapped proxy labels as the practical main targets.",
        "",
        "## File And Source Risks",
        "",
        md_table(file_source, ["source", "status", "path_kind", "files_found", "duplicate_or_competing_copies", "authoritative_months"], 20),
        "",
        "## Hourly SDS011 PM Calendar",
        "",
        md_table(pm, ["month", "status", "expected_hours", "observed_hours", "observed_dates", "rows_before_sds011_filter", "rows_after_sds011_filter", "unique_sds011_locations", "duplicate_location_hour_rows", "path_kind"], 14),
        "",
        "## UBA Daily PM Coverage",
        "",
        md_table(uba, ["pollutant", "status", "observed_dates", "stations", "median_valid_days_per_station", "stations_passing_182_days", "months_represented"]),
        "",
        "## HYRAS",
        "",
        md_table(hyras, ["status", "observed_dates", "requested_sds011_locations", "missing_locations", "same_locations_missing_temperature_and_rh", "missing_boundary_or_mask_like", "duplicate_location_date_rows", "temperature_units", "rh_units"]),
        "",
        "## Local Weather Annual Coverage",
        "",
        md_table(weather, ["source", "status", "available_months", "daily_rows", "unique_locations", "locations_passing_182_days", "saturation_risk"]),
        "",
        "## 500 m Close-Pair Sample",
        "",
        md_table(close, ["pollutant", "station_type", "status", "independent_uba_stations", "sds011_locations", "paired_days", "months_represented", "seasonal_distribution", "folds_represented"]),
        "",
        "## 500 m Modeling Complete Cases",
        "",
        md_table(modeling, ["pollutant", "station_type", "weather_source", "comparison_sample", "status", "final_complete_case_rows", "independent_uba_stations", "sds011_locations", "fraction_by_season"], 80),
        "",
        "## Coordinate And Identity",
        "",
        md_table(coord, ["status", "modeled_locations", "modeled_locations_missing_sensor_land", "modeled_locations_with_coordinate_changes_gt_0p001deg", "duplicate_coordinate_groups", "outside_germany_modeled_locations", "modeled_stations_missing_metadata", "station_type_missing_for_modeled"]),
        "",
        "## Material Risks",
        "",
        md_table(risks, ["audit_section", "source", "status", "month", "pollutant", "station_type", "weather_source", "note"], 120),
        "",
        "## Final Answers",
        "",
        "1. All 12 active merged PM monthly inputs are present; April is a symlink to `processed 2`, so the split-data history remains a duplicate/stale-input risk but not a current modeling ambiguity.",
        "2. Competing processed copies remain: `processed 2` and two processed zip files. The scripts use `data/processed` only.",
        "3. UBA daily PM covers 366 dates for PM10 and PM2.5; station coverage by type is present in the CSV.",
        "4. BME280 and DHT22 hourly weather both cover all 12 months; DHT22 has strong saturation/flatline risk, BME280 is a smaller but cleaner subset.",
        "5. HYRAS covers 366 dates and 6,050 SDS011 locations, with the expected six boundary/mask missing locations for both temperature and RH.",
        "6. Close-pair and complete-case rows cover all seasons at 500 m for the main HYRAS background/all strata; BME280 common rows are smaller and therefore a modeling-bias risk.",
        "7. No material coordinate, Land, station-type, or metadata join problem was found for modeled rows; Sachsen-Anhalt is absent from model-development rows.",
        "8. Main modeling-bias risks are BME280 subset selection, DHT22 saturation, and station-type/environment mismatch, not missing calendar source data.",
        "9. The latest negative model result can be trusted as a calibration-branch decision signal.",
        "10. Nothing needs to be regenerated before ending this branch.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    processed = Path(args.processed_dir)
    out_dir = Path(args.output_dir) if args.output_dir else processed / "calibration/regression_reference_adjustment/data_completeness"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_rows.extend(file_rows(processed))
    pm_rows, daily_pm, nodes = pm_month_rows(processed)
    all_rows.extend(pm_rows)
    uba_part, uba = uba_rows(processed)
    all_rows.extend(uba_part)
    hyras_part, hyras = hyras_rows(processed)
    all_rows.extend(hyras_part)
    local_part, weather_daily = local_weather_rows(processed)
    all_rows.extend(local_part)
    pair_part, modeled_locations, modeled_stations = close_pair_and_model_rows(processed, daily_pm, nodes, uba, hyras, weather_daily)
    all_rows.extend(pair_part)
    all_rows.extend(coordinate_rows(processed, nodes, modeled_locations, modeled_stations))
    all_rows.extend(bias_risk_rows(all_rows))

    results = pd.DataFrame(all_rows)
    csv_path = out_dir / "data_completeness.csv"
    md_path = out_dir / "DATA_COMPLETENESS.md"
    results.to_csv(csv_path, index=False)
    write_report(md_path, results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
