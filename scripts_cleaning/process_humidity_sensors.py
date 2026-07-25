"""
Computes daily/monthly temp/humidity(/pressure) averages per sensor from Sensor.Community archives.
Input:  data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip
Output: data/processed/daily_avg/humidity/<sensor_type>_<YYYY-MM>.csv
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
DAILY_DIR = OUT_DIR / "daily_avg" / "humidity"
MONTHLY_DIR = OUT_DIR / "monthly_avg" / "humidity"
MERGED_DIR = MONTHLY_DIR / "all_sensors"

HUMIDITY_SENSOR_TYPES = ["dht22", "bme280"]

REQUIRED_COLS = ["sensor_id", "lat", "lon", "timestamp"]

SENSOR_MEASUREMENT_COLS = {
    "dht22": ["temperature", "humidity"],
    "bme280": ["temperature", "humidity", "pressure"],
}

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

CHUNKSIZE = 2_000_000
MIN_READINGS_PER_DAY = 12
MIN_DAYS_PER_MONTH = 10

DTYPES = {"sensor_id": "int32", "lat": "float32", "lon": "float32"}

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


def _process_chunks(reader, measurement_cols: list[str]):
    daily_parts = []
    total_rows_seen = 0
    total_rows_kept = 0
    start_time = time.time()

    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)

        mask = (
            chunk["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"])
            & chunk["lon"].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])
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

        germany_chunk = germany_chunk.dropna(subset=measurement_cols, how="all")

        if germany_chunk.empty:
            print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
                  f"0 valid Germany rows after cleaning measurements "
                  f"({time.time() - start_time:.0f}s elapsed)")
            continue

        germany_chunk["date"] = germany_chunk["timestamp"].dt.date

        agg_dict = {col: ["sum", "count"] for col in measurement_cols}
        agg = (
            germany_chunk.groupby(["sensor_id", "lat", "lon", "date"])[measurement_cols]
            .agg(agg_dict)
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        daily_parts.append(agg)
        total_rows_kept += len(germany_chunk)
        print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
              f"{len(germany_chunk):,} valid Germany rows -> {len(agg):,} sensor-day aggregates "
              f"({time.time() - start_time:.0f}s elapsed)")
        del germany_chunk, agg

    return daily_parts, total_rows_seen, total_rows_kept


def process_zip(zip_path: Path, sensor_type: str, daily_out_path: Path, monthly_out_path: Path) -> bool:
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"    file size: {zip_size_mb:.1f} MiB")

    usecols = verify_columns(zip_path, sensor_type)
    measurement_cols = SENSOR_MEASUREMENT_COLS[sensor_type]
    print(f"    measurement columns: {measurement_cols}")

    try:
        reader = _read_chunks(zip_path, usecols, engine="c")
        daily_parts, total_rows_seen, total_rows_kept = _process_chunks(reader, measurement_cols)
    except (IndexError, ValueError) as exc:
        # C engine + usecols + on_bad_lines=skip has a bug on malformed rows, python engine doesn't
        print(f"    C engine failed ({exc!r}), retrying with engine='python' "
              f"(slower, more tolerant of malformed rows)...")
        reader = _read_chunks(zip_path, usecols, engine="python")
        daily_parts, total_rows_seen, total_rows_kept = _process_chunks(reader, measurement_cols)

    print(f"    done reading: {total_rows_seen:,} rows scanned | "
          f"{total_rows_kept:,} valid Germany rows")

    if not daily_parts:
        print("    no Germany data in this file -- skipping output")
        return False

    combined = pd.concat(daily_parts, ignore_index=True)
    del daily_parts

    sum_count_cols = {}
    for col in measurement_cols:
        sum_count_cols[f"{col}_sum"] = (f"{col}_sum", "sum")
        sum_count_cols[f"{col}_count"] = (f"{col}_count", "sum")

    daily_agg = (
        combined.groupby(["sensor_id", "lat", "lon", "date"])
        .agg(**sum_count_cols)
        .reset_index()
    )
    del combined

    count_cols = [f"{col}_count" for col in measurement_cols]
    keep_mask = (daily_agg[count_cols] >= MIN_READINGS_PER_DAY).any(axis=1)  # any col clearing threshold keeps the row, not all
    daily_agg = daily_agg[keep_mask].copy()

    if daily_agg.empty:
        print("    no sensor-days survived the daily filter -- skipping output")
        return False

    for col in measurement_cols:
        daily_agg[col] = daily_agg[f"{col}_sum"] / daily_agg[f"{col}_count"]

    daily_out = daily_agg[["sensor_id", "lat", "lon", "date"] + measurement_cols
                          + count_cols]

    daily_agg["month"] = pd.to_datetime(daily_agg["date"]).dt.to_period("M")
    agg_spec = {col: (col, "mean") for col in measurement_cols}
    agg_spec["n_days"] = ("date", "nunique")
    monthly_avg = (
        daily_agg.groupby(["sensor_id", "lat", "lon", "month"])
        .agg(**agg_spec)
        .reset_index()
    )

    monthly_avg = monthly_avg[monthly_avg["n_days"] >= MIN_DAYS_PER_MONTH].copy()

    if monthly_avg.empty:
        print("    no sensors survived the monthly filter -- skipping output")
        return False

    daily_out_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_out.to_csv(daily_out_path, index=False)
    monthly_avg.to_csv(monthly_out_path, index=False)

    print(f"    saved: daily_avg/humidity/{daily_out_path.name} ({len(daily_out)} rows), "
          f"monthly_avg/humidity/{monthly_out_path.name} ({len(monthly_avg)} rows)")
    return True


def discover_months() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return sorted(
        p.name for p in RAW_DIR.iterdir() if p.is_dir() and MONTH_RE.match(p.name)
    )


def process_month_sensor(month: str, sensor_type: str, force: bool) -> Path | None:
    zip_path = RAW_DIR / month / f"{month}_{sensor_type}.zip"
    daily_out_path = DAILY_DIR / f"{sensor_type}_{month}.csv"
    monthly_out_path = MONTHLY_DIR / f"{sensor_type}_{month}.csv"

    if not zip_path.exists():
        print(f"  [{sensor_type} {month}] no raw file, skipping")
        return None

    if not force and daily_out_path.exists() and monthly_out_path.exists():
        print(f"  [{sensor_type} {month}] already processed, skipping")
        return monthly_out_path

    print(f"  [{sensor_type} {month}] processing {zip_path.name}...")
    try:
        ok = process_zip(zip_path, sensor_type, daily_out_path, monthly_out_path)
    except Exception as exc:
        print(f"  [{sensor_type} {month}] FAILED: {exc}")
        return None

    return monthly_out_path if ok else None


def merge_month_all_sensors(month: str, sensor_types: list[str]) -> None:
    parts = []
    for sensor_type in sensor_types:
        path = MONTHLY_DIR / f"{sensor_type}_{month}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["sensor_type"] = sensor_type
            parts.append(df)

    if not parts:
        return

    merged = pd.concat(parts, ignore_index=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MERGED_DIR / f"{month}.csv"
    merged.to_csv(out_path, index=False)
    print(f"  [{month}] merged {len(parts)} sensor type(s) -> "
          f"monthly_avg/humidity/all_sensors/{out_path.name} ({len(merged)} rows)")


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
        help="Skip creating the combined all_sensors file per month",
    )
    args = parser.parse_args()

    months = args.months or discover_months()
    if not months:
        print(f"No month folders found under {RAW_DIR}. Nothing to do.")
        return

    sensor_types = list(dict.fromkeys(args.types))
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(months)} month(s) x {len(sensor_types)} sensor type(s)\n")

    for month in months:
        print(f"Month {month}:")
        for sensor_type in sensor_types:
            process_month_sensor(month, sensor_type, args.force)
        if not args.no_merge:
            merge_month_all_sensors(month, sensor_types)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
