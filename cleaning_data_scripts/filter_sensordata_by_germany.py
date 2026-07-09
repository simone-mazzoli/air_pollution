"""
Filter Sensor.Community SDS011 monthly archive data to Germany and compute
monthly average PM10 (P1) and PM2.5 (P2) per sensor.

Quality filtering applied:
  1. Daily level  : a sensor-day is kept only if it has at least
                    MIN_READINGS_PER_DAY valid P1 AND P2 readings.
  2. Monthly level: a sensor-month is kept only if it has at least
                    MIN_DAYS_PER_MONTH valid days (after step 1).

NOTE: The "enough valid days in the YEAR" criterion cannot be enforced here,
since this script processes one month at a time. The monthly output keeps
the `n_days` column so a later script that merges all 12 months can apply
the yearly cut (e.g. sum of n_days per sensor >= threshold).

Expects:
    <project_root>/data/raw/<YYYY-MM>/<YYYY-MM>_sds011.zip

Produces:
    <project_root>/data/processed/germany_daily_avg_<YYYY-MM>.csv
    <project_root>/data/processed/germany_monthly_avg_<YYYY-MM>.csv
"""

import pandas as pd
from pathlib import Path

#------------------
#config:
MONTH = "2024-03"

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "data" / "raw" / MONTH / f"{MONTH}_sds011.zip"

OUT_DIR = BASE_DIR / "data" / "processed"
DAILY_OUT_PATH = OUT_DIR / f"germany_daily_avg_sds_{MONTH}.csv"
MONTHLY_OUT_PATH = OUT_DIR / f"germany_monthly_avg_sds_{MONTH}.csv"

# Germany's real bounding box (approx, with small padding for border sensors):
#   south tip (Lake Constance)  ~47.27 N
#   north tip (Sylt)            ~55.06 N
#   west edge (NL/BE border)    ~5.87 E
#   east edge (Saxony/Poland)   ~15.04 E
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

CHUNKSIZE = 2_000_000

#quality filters 
# Minimal daily QC
MIN_READINGS_PER_DAY = 12

# Keep only sensor-months with more than X valid days.
MIN_DAYS_PER_MONTH = 10
# ---------------------------------------------------------------------------

USECOLS = ["sensor_id", "lat", "lon", "timestamp", "P1", "P2"]
DTYPES = {"sensor_id": "int32", "lat": "float32", "lon": "float32"}
# NOTE: P1/P2 intentionally NOT forced to float32 at parse time — the raw
# data sometimes contains the literal string "unavailable" instead of a
# number, which crashes the C parser if the dtype is numeric. We read them
# as plain strings and coerce to numeric ourselves below.


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {ZIP_PATH}")

    reader = pd.read_csv(
        ZIP_PATH,
        sep=";",
        chunksize=CHUNKSIZE,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["timestamp"],
        on_bad_lines="skip",
    )

    daily_parts = []  # small per-chunk aggregates, not raw rows
    total_rows_seen = 0
    total_rows_kept = 0

    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)

        # --- filter to Germany bounding box ---
        mask = (
            chunk["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"])
            & chunk["lon"].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])
        )
        germany_chunk = chunk.loc[mask].copy()
        del chunk

        if germany_chunk.empty:
            print(f"chunk {i}: 0 rows in Germany bbox")
            continue

        # --- clean P1/P2: coerce to numeric, drop invalid ("unavailable", etc.) ---
        germany_chunk["P1"] = pd.to_numeric(germany_chunk["P1"], errors="coerce")
        germany_chunk["P2"] = pd.to_numeric(germany_chunk["P2"], errors="coerce")
        germany_chunk = germany_chunk.dropna(subset=["P1", "P2"])

        if germany_chunk.empty:
            print(f"chunk {i}: 0 valid rows after cleaning P1/P2")
            continue

        germany_chunk["date"] = germany_chunk["timestamp"].dt.date

        # --- aggregate this chunk down to (sensor_id, lat, lon, date) sums+counts ---
        # NOTE: no per-day count filtering here! A sensor-day can be split
        # across chunks, so counts are only complete after all chunks are
        # combined. The daily filter is applied after the final groupby below.
        agg = (
            germany_chunk.groupby(["sensor_id", "lat", "lon", "date"])[["P1", "P2"]]
            .agg(["sum", "count"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()

        daily_parts.append(agg)
        total_rows_kept += len(germany_chunk)

        print(
            f"chunk {i}: {len(germany_chunk)} valid Germany rows "
            f"-> {len(agg)} sensor-day aggregates"
        )

        del germany_chunk, agg

    print(f"\nTotal rows scanned: {total_rows_seen:,}")
    print(f"Total valid Germany rows kept: {total_rows_kept:,}")

    if not daily_parts:
        print("No data found for Germany in this file. Exiting.")
        return

    # --- combine all small per-chunk aggregates ---
    combined = pd.concat(daily_parts, ignore_index=True)
    del daily_parts

    # a sensor-day may appear in multiple chunks -> sum sums/counts before dividing
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

    # --- daily filter: require a minimal number of valid readings per day ---
    n_days_before = len(daily_agg)
    daily_agg = daily_agg[
        (daily_agg["P1_count"] >= MIN_READINGS_PER_DAY)
        & (daily_agg["P2_count"] >= MIN_READINGS_PER_DAY)
    ].copy()
    print(
        f"Daily filter (>= {MIN_READINGS_PER_DAY} readings): "
        f"{n_days_before:,} -> {len(daily_agg):,} sensor-days"
    )

    if daily_agg.empty:
        print("No sensor-days survived the daily filter. Exiting.")
        return

    daily_agg["P1"] = daily_agg["P1_sum"] / daily_agg["P1_count"]
    daily_agg["P2"] = daily_agg["P2_sum"] / daily_agg["P2_count"]

    daily_out = daily_agg[["sensor_id", "lat", "lon", "date", "P1", "P2",
                           "P1_count", "P2_count"]]

    # --- monthly average, computed from daily averages (equal weight per day) ---
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

    # --- monthly filter: require a minimal number of valid days per month ---
    n_sensors_before = len(monthly_avg)
    monthly_avg = monthly_avg[monthly_avg["n_days"] >= MIN_DAYS_PER_MONTH].copy()
    print(
        f"Monthly filter (>= {MIN_DAYS_PER_MONTH} valid days): "
        f"{n_sensors_before:,} -> {len(monthly_avg):,} sensor-months"
    )

    if monthly_avg.empty:
        print("No sensors survived the monthly filter. Exiting.")
        return

    # --- save outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_out.to_csv(DAILY_OUT_PATH, index=False)
    monthly_avg.to_csv(MONTHLY_OUT_PATH, index=False)

    print(f"\nSaved daily averages   -> {DAILY_OUT_PATH}  ({len(daily_out)} rows)")
    print(f"Saved monthly averages -> {MONTHLY_OUT_PATH}  ({len(monthly_avg)} rows)")
    print(f"\nUnique sensors in Germany this month: {monthly_avg['sensor_id'].nunique()}")
    print(monthly_avg.head())


if __name__ == "__main__":
    main()
