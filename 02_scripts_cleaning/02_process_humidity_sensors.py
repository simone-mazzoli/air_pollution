"""
Computes hourly/daily/monthly temp/humidity(/pressure) averages per node from
Sensor.Community archives.

Keyed on `location` (the physical node) rather than `sensor_id`, so the humidity sensor
can be joined to the PM sensor sitting in the same enclosure.

Alongside the plain hourly RH mean this emits `humidity_clip90`: the mean of the
per-reading value clipped at 90% RH. The Koehler growth factor blows up as RH approaches
100, so the clip has to be applied per reading -- clipping an hourly mean bounds nothing.
Persisting the clipped mean here keeps the per-reading clip exact without persisting raw
readings. `n_gt90` / `frac_gt90` carry the clipped fraction as a per-node quality flag.

All timestamps are naive UTC, exactly as they appear in the raw archives.

Input:  data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip
Output: data/processed/hourly/humidity/<sensor_type>_<YYYY-MM>.parquet
        data/processed/hourly/humidity/nodes/<sensor_type>_<YYYY-MM>.parquet
        data/processed/hourly/humidity/all_sensors/<YYYY-MM>.parquet
        data/processed/daily_avg/humidity/<sensor_type>_<YYYY-MM>.csv
        data/processed/monthly_avg/humidity/<sensor_type>_<YYYY-MM>.csv
        data/processed/monthly_avg/humidity/all_sensors/<YYYY-MM>.csv
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

HOURLY_DIR = OUT_DIR / "hourly" / "humidity"
NODES_DIR = HOURLY_DIR / "nodes"
HOURLY_MERGED_DIR = HOURLY_DIR / "all_sensors"
DAILY_DIR = OUT_DIR / "daily_avg" / "humidity"
MONTHLY_DIR = OUT_DIR / "monthly_avg" / "humidity"
MONTHLY_MERGED_DIR = MONTHLY_DIR / "all_sensors"

# ordered by preference: a node carrying both instruments resolves to the earlier one in
# the merged file. bme280 first -- it drifts less than the dht22. averaging the two would
# mix instruments with different response characteristics.
HUMIDITY_SENSOR_TYPES = ["bme280", "dht22"]

REQUIRED_COLS = ["sensor_id", "location", "lat", "lon", "timestamp"]

SENSOR_MEASUREMENT_COLS = {
    "dht22": ["temperature", "humidity"],
    "bme280": ["temperature", "humidity", "pressure"],
}

# derived per reading, aggregated the same way as the raw measurements
DERIVED_COLS = ["humidity_clip90", "humidity_gt90"]

RH_CLIP = 90.0

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

CHUNKSIZE = 2_000_000

MIN_READINGS_PER_HOUR = 5
MIN_HOURS_PER_DAY = 18
MIN_DAYS_PER_MONTH = 10

DTYPES = {
    "sensor_id": "int32",
    "location": "int32",
    "lat": "float32",
    "lon": "float32",
}

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def verify_columns(zip_path: Path, sensor_type: str) -> list[str]:
    # fail loud if a real header doesn't match what we hardcoded
    expected = SENSOR_MEASUREMENT_COLS[sensor_type]
    header = pd.read_csv(zip_path, sep=";", nrows=0)
    available = set(header.columns)

    missing = [c for c in REQUIRED_COLS + expected if c not in available]
    if missing:
        raise ValueError(
            f"expected columns {missing} not found for sensor_type={sensor_type}; "
            f"actual header was: {list(header.columns)}"
        )

    return REQUIRED_COLS + expected


def _read_chunks(zip_path: Path, usecols: list[str], engine: str):
    kwargs = dict(
        sep=";",
        chunksize=CHUNKSIZE,
        usecols=usecols,
        parse_dates=False,
        on_bad_lines="skip",
        engine=engine,
    )
    if engine == "c":
        kwargs["dtype"] = DTYPES
    return pd.read_csv(zip_path, **kwargs)


def _apply_range_filters(df: pd.DataFrame, measurement_cols: list[str]) -> pd.DataFrame:
    # invalid values are set to NaN rather than dropping the whole row: a bad
    # temperature reading should not cost us the humidity reading beside it.
    # count/sum skip NaN, so the per-column counts stay honest.
    if "humidity" in measurement_cols:
        # strict bounds: the dht22 rails at exactly 0.0 and 100.0 when it fails
        bad = ~df["humidity"].between(0, 100, inclusive="neither")
        df.loc[bad, "humidity"] = pd.NA

    if "temperature" in measurement_cols:
        bad = ~df["temperature"].between(-50, 60)
        df.loc[bad, "temperature"] = pd.NA

    if "pressure" in measurement_cols:
        # raw units are inconsistent across firmware versions (Pa vs hPa), so this is
        # only a non-positive sanity check. pressure is not used in the correction.
        df.loc[df["pressure"] <= 0, "pressure"] = pd.NA

    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["humidity_clip90"] = df["humidity"].clip(upper=RH_CLIP)
    df["humidity_gt90"] = (df["humidity"] > RH_CLIP).astype("float32")
    df.loc[df["humidity"].isna(), "humidity_gt90"] = pd.NA
    return df


def _process_chunks(reader, measurement_cols: list[str]):
    agg_cols = measurement_cols + DERIVED_COLS
    hourly_parts = []
    node_parts = []
    total_rows_seen = 0
    total_rows_kept = 0
    start_time = time.time()

    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)

        for col in ("lat", "lon", "location", "sensor_id"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        mask = (
            chunk["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"])
            & chunk["lon"].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])
            & chunk["location"].notna()
        )
        germany_chunk = chunk.loc[mask].copy()
        del chunk

        if germany_chunk.empty:
            print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
                  f"0 in Germany bbox ({time.time() - start_time:.0f}s elapsed)")
            continue

        # errors="coerce" so one bad row doesn't downgrade the whole column to str
        germany_chunk["timestamp"] = pd.to_datetime(
            germany_chunk["timestamp"], errors="coerce"
        )
        germany_chunk = germany_chunk.dropna(subset=["timestamp"])

        for col in measurement_cols:
            germany_chunk[col] = pd.to_numeric(germany_chunk[col], errors="coerce")

        germany_chunk = _apply_range_filters(germany_chunk, measurement_cols)
        germany_chunk = _add_derived(germany_chunk)

        germany_chunk = germany_chunk.dropna(subset=measurement_cols, how="all")

        if germany_chunk.empty:
            print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
                  f"0 valid Germany rows after cleaning measurements "
                  f"({time.time() - start_time:.0f}s elapsed)")
            continue

        germany_chunk["hour"] = germany_chunk["timestamp"].dt.floor("h")

        # lat/lon deliberately kept OUT of the groupby key: GPS jitter or a
        # re-registration gives one physical node two coordinate pairs, which silently
        # splits it into two groups and halves the counts the threshold filters see.
        node_parts.append(
            germany_chunk[["location", "sensor_id", "lat", "lon"]].drop_duplicates()
        )

        agg = (
            germany_chunk.groupby(["location", "sensor_id", "hour"])[agg_cols]
            .agg(["sum", "count"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        hourly_parts.append(agg)
        total_rows_kept += len(germany_chunk)
        print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
              f"{len(germany_chunk):,} valid Germany rows -> {len(agg):,} node-hour aggregates "
              f"({time.time() - start_time:.0f}s elapsed)")
        del germany_chunk, agg

    return hourly_parts, node_parts, total_rows_seen, total_rows_kept


def build_hourly(hourly_parts: list[pd.DataFrame],
                 measurement_cols: list[str]) -> pd.DataFrame:
    agg_cols = measurement_cols + DERIVED_COLS
    combined = pd.concat(hourly_parts, ignore_index=True)

    sum_count_cols = {}
    for col in agg_cols:
        sum_count_cols[f"{col}_sum"] = (f"{col}_sum", "sum")
        sum_count_cols[f"{col}_count"] = (f"{col}_count", "sum")

    hourly = (
        combined.groupby(["location", "sensor_id", "hour"])
        .agg(**sum_count_cols)
        .reset_index()
    )
    del combined

    # the old `.any(axis=1)` across all count columns was defensible when this script
    # stood alone. now that its only job is supplying RH, an hour with 25 temperature
    # readings and 2 humidity readings must not survive on the strength of the
    # temperature count.
    hourly = hourly[hourly["humidity_count"] >= MIN_READINGS_PER_HOUR].copy()

    if hourly.empty:
        return hourly

    for col in measurement_cols + ["humidity_clip90"]:
        hourly[col] = (hourly[f"{col}_sum"] / hourly[f"{col}_count"]).astype("float32")

    hourly["n_gt90"] = hourly["humidity_gt90_sum"].astype("int16")
    hourly["frac_gt90"] = (
        hourly["humidity_gt90_sum"] / hourly["humidity_gt90_count"]
    ).astype("float32")

    count_cols = [f"{c}_count" for c in measurement_cols]
    for col in count_cols:
        hourly[col] = hourly[col].astype("int16")

    out_cols = (["location", "sensor_id", "hour"] + measurement_cols
                + ["humidity_clip90", "n_gt90", "frac_gt90"] + count_cols)
    return hourly[out_cols]


def build_nodes(node_parts: list[pd.DataFrame]) -> pd.DataFrame:
    nodes = pd.concat(node_parts, ignore_index=True).drop_duplicates()
    # median over the month collapses GPS jitter without being pulled by a single
    # bad fix the way a mean would be
    nodes = (
        nodes.groupby(["location", "sensor_id"], as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"))
    )
    nodes["lat"] = nodes["lat"].astype("float32")
    nodes["lon"] = nodes["lon"].astype("float32")
    return nodes


def hourly_to_daily(hourly: pd.DataFrame, measurement_cols: list[str]) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["hour"].dt.date

    mean_cols = measurement_cols + ["humidity_clip90", "frac_gt90"]
    agg_spec = {col: (col, "mean") for col in mean_cols}
    agg_spec["n_hours"] = ("hour", "nunique")

    daily = (
        h.groupby(["location", "sensor_id", "date"])
        .agg(**agg_spec)
        .reset_index()
    )
    return daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()


def daily_to_monthly(daily: pd.DataFrame, measurement_cols: list[str]) -> pd.DataFrame:
    d = daily.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")

    mean_cols = measurement_cols + ["humidity_clip90", "frac_gt90"]
    agg_spec = {col: (col, "mean") for col in mean_cols}
    agg_spec["n_days"] = ("date", "nunique")

    monthly = (
        d.groupby(["location", "sensor_id", "month"])
        .agg(**agg_spec)
        .reset_index()
    )
    return monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH].copy()


def process_zip(zip_path: Path, sensor_type: str, paths: dict) -> bool:
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"    file size: {zip_size_mb:.1f} MiB")

    usecols = verify_columns(zip_path, sensor_type)
    measurement_cols = SENSOR_MEASUREMENT_COLS[sensor_type]
    print(f"    measurement columns: {measurement_cols}")

    try:
        reader = _read_chunks(zip_path, usecols, engine="c")
        parts, node_parts, seen, kept = _process_chunks(reader, measurement_cols)
    except (IndexError, ValueError) as exc:
        # C engine + usecols + on_bad_lines=skip has a bug on malformed rows,
        # python engine doesn't
        print(f"    C engine failed ({exc!r}), retrying with engine='python' "
              f"(slower, more tolerant of malformed rows)...")
        reader = _read_chunks(zip_path, usecols, engine="python")
        parts, node_parts, seen, kept = _process_chunks(reader, measurement_cols)

    print(f"    done reading: {seen:,} rows scanned | {kept:,} valid Germany rows")

    if not parts:
        print("    no Germany data in this file -- skipping output")
        return False

    hourly = build_hourly(parts, measurement_cols)
    del parts

    if hourly.empty:
        print("    no node-hours survived the hourly filter -- skipping output")
        return False

    nodes = build_nodes(node_parts)
    del node_parts

    daily = hourly_to_daily(hourly, measurement_cols)
    if daily.empty:
        print("    no node-days survived the daily filter -- skipping output")
        return False

    monthly = daily_to_monthly(daily, measurement_cols)
    if monthly.empty:
        print("    no nodes survived the monthly filter -- skipping output")
        return False

    # lat/lon joined back only on the analysis-facing outputs
    daily_out = daily.merge(nodes, on=["location", "sensor_id"], how="left")
    monthly_out = monthly.merge(nodes, on=["location", "sensor_id"], how="left")

    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    hourly.to_parquet(paths["hourly"], index=False)
    nodes.to_parquet(paths["nodes"], index=False)
    daily_out.to_csv(paths["daily"], index=False)
    monthly_out.to_csv(paths["monthly"], index=False)

    print(f"    saved: hourly {len(hourly):,} rows | nodes {len(nodes):,} | "
          f"daily {len(daily_out):,} | monthly {len(monthly_out):,}")
    return True


def discover_months() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(
        p.name for p in RAW_DIR.iterdir() if p.is_dir() and MONTH_RE.match(p.name)
    )


def output_paths(month: str, sensor_type: str) -> dict:
    return {
        "hourly": HOURLY_DIR / f"{sensor_type}_{month}.parquet",
        "nodes": NODES_DIR / f"{sensor_type}_{month}.parquet",
        "daily": DAILY_DIR / f"{sensor_type}_{month}.csv",
        "monthly": MONTHLY_DIR / f"{sensor_type}_{month}.csv",
    }


def process_month_sensor(month: str, sensor_type: str, force: bool) -> bool:
    zip_path = RAW_DIR / month / f"{month}_{sensor_type}.zip"
    paths = output_paths(month, sensor_type)

    if not zip_path.exists():
        print(f"  [{sensor_type} {month}] no raw file, skipping")
        return False

    if not force and all(p.exists() for p in paths.values()):
        print(f"  [{sensor_type} {month}] already processed, skipping")
        return True

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing and not force:
        print(f"  [{sensor_type} {month}] processing {zip_path.name} "
              f"(missing: {', '.join(missing)})...")
    else:
        print(f"  [{sensor_type} {month}] processing {zip_path.name}...")

    try:
        return process_zip(zip_path, sensor_type, paths)
    except Exception as exc:
        print(f"  [{sensor_type} {month}] FAILED: {exc}")
        return False


def _dedupe_by_priority(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = {t: i for i, t in enumerate(HUMIDITY_SENSOR_TYPES)}
    df = df.copy()
    df["_prio"] = df["sensor_type"].map(order).fillna(len(order))
    df = df.sort_values("_prio").drop_duplicates(subset=keys, keep="first")
    return df.drop(columns="_prio")


def merge_month(month: str, sensor_types: list[str], force: bool) -> None:
    hourly_out = HOURLY_MERGED_DIR / f"{month}.parquet"
    monthly_out = MONTHLY_MERGED_DIR / f"{month}.csv"

    if not force and hourly_out.exists() and monthly_out.exists():
        print(f"  [{month}] merged files already exist, skipping")
        return

    hourly_parts, monthly_parts = [], []
    for sensor_type in sensor_types:
        paths = output_paths(month, sensor_type)
        if paths["hourly"].exists():
            df = pd.read_parquet(paths["hourly"])
            df["sensor_type"] = sensor_type
            hourly_parts.append(df)
        if paths["monthly"].exists():
            df = pd.read_csv(paths["monthly"])
            df["sensor_type"] = sensor_type
            monthly_parts.append(df)

    if not hourly_parts:
        return

    # this is the file the correction step joins against, so it must be one RH row per
    # (location, hour)
    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = _dedupe_by_priority(hourly, ["location", "hour"])
    HOURLY_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(hourly_out, index=False)

    monthly = pd.concat(monthly_parts, ignore_index=True)
    monthly = _dedupe_by_priority(monthly, ["location", "month"])
    MONTHLY_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_out, index=False)

    print(f"  [{month}] merged {len(hourly_parts)} sensor type(s) -> "
          f"hourly {len(hourly):,} rows, monthly {len(monthly):,} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--months", nargs="+", default=None,
        help="Months to process, e.g. 2024-01 2024-02 (default: all found in data/raw)",
    )
    parser.add_argument(
        "--types", nargs="+", default=HUMIDITY_SENSOR_TYPES,
        help=f"Humidity sensor types to process (default: {HUMIDITY_SENSOR_TYPES})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if output files already exist",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip creating the combined all_sensors files per month",
    )
    args = parser.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is required for parquet output: pip install pyarrow")

    months = args.months or discover_months()
    if not months:
        print(f"No month folders found under {RAW_DIR}. Nothing to do.")
        return

    sensor_types = list(dict.fromkeys(args.types))

    print(f"Processing {len(months)} month(s) x {len(sensor_types)} sensor type(s)\n")

    for month in months:
        print(f"Month {month}:")
        for sensor_type in sensor_types:
            process_month_sensor(month, sensor_type, args.force)
        if not args.no_merge:
            merge_month(month, sensor_types, args.force)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
