"""
Computes daily/monthly PM10/PM2.5 averages per sensor from Sensor.Community archives.
Input:  data/raw/<YYYY-MM>/<YYYY-MM>_<sensor_type>.zip
Output: data/processed/daily_avg/<sensor_type>_<YYYY-MM>.csv
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
DAILY_DIR = OUT_DIR / "daily_avg"
MONTHLY_DIR = OUT_DIR / "monthly_avg"
MERGED_DIR = MONTHLY_DIR / "all_pm_sensors"

PM_SENSOR_TYPES = [
    "sds011",
    "pms1003",
    "pms3003",
    "pms5003",
    "pms6003",
    "pms7003",
    "sps30",  # measures 4 size fractions natively, double check header is just P1/P2
]

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

CHUNKSIZE = 2_000_000

MIN_READINGS_PER_DAY = 12
MIN_DAYS_PER_MONTH = 10

USECOLS = ["sensor_id", "lat", "lon", "timestamp", "P1", "P2"]
DTYPES = {"sensor_id": "int32", "lat": "float32", "lon": "float32"}  # P1/P2 left as str, raw data has "unavailable" sometimes

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def process_zip(zip_path: Path, daily_out_path: Path, monthly_out_path: Path) -> bool:
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"    file size: {zip_size_mb:.1f} MiB")

    reader = pd.read_csv(
        zip_path,
        sep=";",
        chunksize=CHUNKSIZE,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["timestamp"],
        on_bad_lines="skip",
    )

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

        germany_chunk["P1"] = pd.to_numeric(germany_chunk["P1"], errors="coerce")
        germany_chunk["P2"] = pd.to_numeric(germany_chunk["P2"], errors="coerce")
        germany_chunk = germany_chunk.dropna(subset=["P1", "P2"])

        if germany_chunk.empty:
            print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
                  f"0 valid Germany rows after cleaning P1/P2 "
                  f"({time.time() - start_time:.0f}s elapsed)")
            continue

        germany_chunk["date"] = germany_chunk["timestamp"].dt.date

        agg = (
            germany_chunk.groupby(["sensor_id", "lat", "lon", "date"])[["P1", "P2"]]
            .agg(["sum", "count"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        daily_parts.append(agg)
        total_rows_kept += len(germany_chunk)
        print(f"    chunk {i}: {total_rows_seen:,} rows read so far, "
              f"{len(germany_chunk):,} valid Germany rows -> {len(agg):,} sensor-day aggregates "
              f"({time.time() - start_time:.0f}s elapsed)")
        del germany_chunk, agg

    print(f"    done reading: {total_rows_seen:,} rows scanned | "
          f"{total_rows_kept:,} valid Germany rows | {time.time() - start_time:.0f}s")

    if not daily_parts:
        print("    no Germany data in this file -- skipping output")
        return False

    combined = pd.concat(daily_parts, ignore_index=True)
    del daily_parts

    daily_agg = (
        combined.groupby(["sensor_id", "lat", "lon", "date"])
        .agg(
            P1_sum=("P1_sum", "sum"),
            P1_count=("P1_count", "sum"),
            P2_sum=("P2_sum", "sum"),
            P2_count=("P2_count", "sum"),
        )
        .reset_index()
    )
    del combined

    daily_agg = daily_agg[
        (daily_agg["P1_count"] >= MIN_READINGS_PER_DAY)
        & (daily_agg["P2_count"] >= MIN_READINGS_PER_DAY)
    ].copy()

    if daily_agg.empty:
        print("    no sensor-days survived the daily filter -- skipping output")
        return False

    daily_agg["P1"] = daily_agg["P1_sum"] / daily_agg["P1_count"]
    daily_agg["P2"] = daily_agg["P2_sum"] / daily_agg["P2_count"]

    daily_out = daily_agg[["sensor_id", "lat", "lon", "date", "P1", "P2",
                           "P1_count", "P2_count"]]

    daily_agg["month"] = pd.to_datetime(daily_agg["date"]).dt.to_period("M")
    monthly_avg = (
        daily_agg.groupby(["sensor_id", "lat", "lon", "month"])
        .agg(
            P1=("P1", "mean"),
            P2=("P2", "mean"),
            n_days=("date", "nunique"),
        )
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

    print(f"    saved: daily_avg/{daily_out_path.name} ({len(daily_out)} rows), "
          f"monthly_avg/{monthly_out_path.name} ({len(monthly_avg)} rows)")
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
        ok = process_zip(zip_path, daily_out_path, monthly_out_path)
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
          f"monthly_avg/all_pm_sensors/{out_path.name} ({len(merged)} rows)")


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
        help="Skip creating the combined ALLPM file per month",
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
