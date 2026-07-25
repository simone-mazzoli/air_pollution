"""
Aggregates UBA daily PM10/PM2.5 reference station data into monthly means.
Input:  data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
Output: data/processed/monthly_avg/uba/pm_reference_stations_<YEAR>_<MONTH>.csv (one per month)
"""

from pathlib import Path

import pandas as pd

YEAR = 2024
BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "processed" / "daily_avg" / "uba" / f"pm_reference_stations_{YEAR}.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "monthly_avg" / "uba"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Could not find {IN_PATH} -- run download_uba_measurements.py first")

    df = pd.read_csv(IN_PATH)
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
    df = df.dropna(subset=["Datum"])
    df["month"] = df["Datum"].dt.to_period("M")
    df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for month, group in df.groupby("month"):
        monthly = (
            group.groupby(["station_code", "station_name", "lat", "lon"], as_index=False)
            .agg(PM10=("PM10", "mean"), PM2_5=("PM2.5", "mean"), n_days=("Datum", "nunique"))
        )
        out_path = OUT_DIR / f"pm_reference_stations_{month}.csv"
        monthly.to_csv(out_path, index=False)
        print(f"{month}: {len(monthly)} stations -> {out_path.name}")


if __name__ == "__main__":
    main()
