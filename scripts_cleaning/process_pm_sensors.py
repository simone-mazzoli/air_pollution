"""
Computes hourly/daily/monthly PM10/PM2.5 averages per node from Sensor.Community archives.

Keyed on `location` (the physical node) rather than `sensor_id`, so the PM sensor can be
joined to the humidity sensor sitting in the same enclosure.

All timestamps are naive UTC, exactly as they appear in the raw archives. Conversion is
deferred to the correction step. Note for that step: UBA reports in MEZ, which is a fixed
UTC+1 with no DST -- it is NOT Europe/Berlin.

Input:  data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip
Output: data/processed/hourly/pm/<sensor_type>_<YYYY-MM>.parquet
        data/processed/hourly/pm/nodes/<sensor_type>_<YYYY-MM>.parquet
        data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet
        data/processed/daily_avg/<sensor_type>_<YYYY-MM>.csv
        data/processed/monthly_avg/<sensor_type>_<YYYY-MM>.csv
        data/processed/monthly_avg/all_pm_sensors/<YYYY-MM>.csv
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

HOURLY_DIR = OUT_DIR / "hourly" / "pm"
NODES_DIR = HOURLY_DIR / "nodes"
HOURLY_MERGED_DIR = HOURLY_DIR / "all_pm_sensors"
DAILY_DIR = OUT_DIR / "daily_avg"
MONTHLY_DIR = OUT_DIR / "monthly_avg"
MONTHLY_MERGED_DIR = MONTHLY_DIR / "all_pm_sensors"

# ordered by preference: if one location somehow carries two PM sensors, the earlier
# type wins in the merged file. sds011 first because that is what the calibration is
# fitted on.
PM_SENSOR_TYPES = [
    "sds011",
    "sps30",
    "pms7003",
    "pms5003",
    "pms6003",
    "pms3003",
    "pms1003",
]

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

CHUNKSIZE = 2_000_000

# two-level temporal coverage filter. counting raw readings per day (the old approach)
# does not distinguish 12 readings spread over 24h from 12 readings inside one hour;
# PM has a strong diurnal cycle so partial-day coverage biases the daily mean.
MIN_READINGS_PER_HOUR = 5
MIN_HOURS_PER_DAY = 18
MIN_DAYS_PER_MONTH = 10

# physical plausibility, applied per reading before any aggregation
PM_MAX = 1000.0  # SDS011 rails near 1999.9

REQUIRED_COLS = ["sensor_id", "location", "lat", "lon", "timestamp"]
MEASUREMENT_COLS = ["P1", "P2"]  # P1 = PM10, P2 = PM2.5

# P1/P2 deliberately left as str: raw data contains "unavailable" in places
DTYPES = {
    "sensor_id": "int32",
    "location": "int32",
    "lat": "float32",
    "lon": "float32",
}

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def verify_columns(zip_path: Path) -> list[str]:
    # fail loud if a real header doesn't match what we hardcoded
    header = pd.read_csv(zip_path, sep=";", nrows=0)
    available = set(header.columns)

    missing = [c for c in REQUIRED_COLS + MEASUREMENT_COLS if c not in available]
    if missing:
        raise ValueError(
            f"expected columns {missing} not found; "
            f"actual header was: {list(header.columns)}"
        )

    return REQUIRED_COLS + MEASUREMENT_COLS


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


def _process_chunks(reader):
    hourly_parts = []
    node_parts = []
    total_rows_seen = 0
    total_rows_kept = 0
    total_rows_implausible = 0
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

        for col in MEASUREMENT_COLS:
            germany_chunk[col] = pd.to_numeric(germany_chunk[col], errors="coerce")
        germany_chunk = germany_chunk.dropna(subset=MEASUREMENT_COLS)

        # physical plausibility. P2 > P1 means PM2.5 exceeds PM10, which is impossible
        # and is a reliable corruption signal. must run per reading -- averaging first
        # hides it.
        n_before = len(germany_chunk)
        plausible = (
            germany_chunk["P1"].between(0, PM_MAX)
            & germany_chunk["P2"].between(0, PM_MAX)
            & (germany_chunk["P2"] <= germany_chunk["P1"])
        )
        germany_chunk = germany_chunk.loc[plausible]
        total_rows_implausible += n_before - len(germany_chunk)

        # errors="coerce" so one bad row doesn't downgrade the whole column to str
        germany_chunk["timestamp"] = pd.to_datetime(
            germany_chunk["timestamp"], errors="coerce"
        )
        germany_chunk = germany_chunk.dropna(subset=["timestamp"])

        if germany_chunk.empty:
            print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
                  f"0 valid Germany rows after cleaning "
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
            germany_chunk.groupby(["location", "sensor_id", "hour"])[MEASUREMENT_COLS]
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

    return hourly_parts, node_parts, total_rows_seen, total_rows_kept, total_rows_implausible


def build_hourly(hourly_parts: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(hourly_parts, ignore_index=True)

    sum_count_cols = {}
    for col in MEASUREMENT_COLS:
        sum_count_cols[f"{col}_sum"] = (f"{col}_sum", "sum")
        sum_count_cols[f"{col}_count"] = (f"{col}_count", "sum")

    hourly = (
        combined.groupby(["location", "sensor_id", "hour"])
        .agg(**sum_count_cols)
        .reset_index()
    )
    del combined

    # both fractions must clear the threshold -- they come from the same reading, so a
    # split count means something is wrong with the row
    keep = (
        (hourly["P1_count"] >= MIN_READINGS_PER_HOUR)
        & (hourly["P2_count"] >= MIN_READINGS_PER_HOUR)
    )
    hourly = hourly.loc[keep].copy()

    if hourly.empty:
        return hourly

    for col in MEASUREMENT_COLS:
        hourly[col] = (hourly[f"{col}_sum"] / hourly[f"{col}_count"]).astype("float32")
        hourly[f"{col}_count"] = hourly[f"{col}_count"].astype("int16")

    return hourly[["location", "sensor_id", "hour"] + MEASUREMENT_COLS
                  + [f"{c}_count" for c in MEASUREMENT_COLS]]


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


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["hour"].dt.date

    # equal weight per hour, not per reading: a node that happens to report more often
    # at night should not have its night hours count for more
    agg_spec = {col: (col, "mean") for col in MEASUREMENT_COLS}
    agg_spec["n_hours"] = ("hour", "nunique")

    daily = (
        h.groupby(["location", "sensor_id", "date"])
        .agg(**agg_spec)
        .reset_index()
    )
    return daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")

    agg_spec = {col: (col, "mean") for col in MEASUREMENT_COLS}
    agg_spec["n_days"] = ("date", "nunique")

    monthly = (
        d.groupby(["location", "sensor_id", "month"])
        .agg(**agg_spec)
        .reset_index()
    )
    return monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH].copy()


def process_zip(zip_path: Path, paths: dict) -> bool:
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"    file size: {zip_size_mb:.1f} MiB")

    usecols = verify_columns(zip_path)

    try:
        reader = _read_chunks(zip_path, usecols, engine="c")
        parts, node_parts, seen, kept, implausible = _process_chunks(reader)
    except (IndexError, ValueError) as exc:
        # C engine + usecols + on_bad_lines=skip has a bug on malformed rows,
        # python engine doesn't
        print(f"    C engine failed ({exc!r}), retrying with engine='python' "
              f"(slower, more tolerant of malformed rows)...")
        reader = _read_chunks(zip_path, usecols, engine="python")
        parts, node_parts, seen, kept, implausible = _process_chunks(reader)

    print(f"    done reading: {seen:,} rows scanned | {kept:,} valid Germany rows | "
          f"{implausible:,} dropped as implausible")

    if not parts:
        print("    no Germany data in this file -- skipping output")
        return False

    hourly = build_hourly(parts)
    del parts

    if hourly.empty:
        print("    no node-hours survived the hourly filter -- skipping output")
        return False

    nodes = build_nodes(node_parts)
    del node_parts

    daily = hourly_to_daily(hourly)
    if daily.empty:
        print("    no node-days survived the daily filter -- skipping output")
        return False

    monthly = daily_to_monthly(daily)
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
        return process_zip(zip_path, paths)
    except Exception as exc:
        print(f"  [{sensor_type} {month}] FAILED: {exc}")
        return False


def _dedupe_by_priority(df: pd.DataFrame, keys: list[str],
                        sensor_types: list[str]) -> pd.DataFrame:
    order = {t: i for i, t in enumerate(PM_SENSOR_TYPES)}
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

    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = _dedupe_by_priority(hourly, ["location", "hour"], sensor_types)
    HOURLY_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(hourly_out, index=False)

    monthly = pd.concat(monthly_parts, ignore_index=True)
    monthly = _dedupe_by_priority(monthly, ["location", "month"], sensor_types)
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
        "--types", nargs="+", default=PM_SENSOR_TYPES,
        help=f"PM sensor types to process (default: {PM_SENSOR_TYPES})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess even if output files already exist",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip creating the combined all_pm_sensors files per month",
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
