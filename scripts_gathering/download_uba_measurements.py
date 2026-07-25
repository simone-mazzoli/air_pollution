"""
Downloads daily PM10/PM2.5 means from UBA reference stations.
Input:  data/processed/uba_stations_germany.csv
Output: data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

YEAR = 2024
BASE_DIR = Path(__file__).resolve().parent.parent
STATIONS_PATH = BASE_DIR / "data" / "processed" / "uba_stations_germany.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "daily_avg" / "uba"
OUT_PATH = OUT_DIR / f"pm_reference_stations_{YEAR}.csv"

# old api-proxy endpoint 500s now, this one works
MEASURES_URL = "https://luftdaten.umweltbundesamt.de/api/air-data/v2/measures/csv"

COMPONENTS = {"PM10": 1, "PM2.5": 9}
SCOPE_DAILY_MEAN = 1

REQUEST_TIMEOUT = 30
RETRY_DELAY_SECONDS = 3
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.3


def fetch_component(station_id, component_id: int) -> pd.DataFrame | None:
    params = {
        "data[0][st]": station_id,  # numeric station id, not station code
        "data[0][co]": component_id,
        "data[0][sc]": SCOPE_DAILY_MEAN,
        "date_from": f"{YEAR}-01-01",
        "date_to": f"{YEAR}-12-31",
        "time_from": 1,
        "time_to": 24,
        "lang": "de",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(MEASURES_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            text = response.text.strip()

            if not text:
                return None

            df = pd.read_csv(StringIO(text), sep=";")
            if df.empty:
                return None
            return df

        except requests.RequestException as exc:
            print(f"    [station_id={station_id} comp={component_id}] attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    print(f"    [station_id={station_id} comp={component_id}] giving up after {MAX_RETRIES} attempts")
    return None


def main() -> None:
    if not STATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {STATIONS_PATH} -- run get_uba_stations.py first"
        )

    stations = pd.read_csv(STATIONS_PATH)
    print(f"Loaded {len(stations)} stations from {STATIONS_PATH.name}\n")

    if OUT_PATH.exists():
        print(f"{OUT_PATH.name} already exists, skipping download. Delete it to re-run.")
        return

    results = []

    for i, row in stations.iterrows():
        station_id = row["station id"]
        station_code = row["station code"]
        print(f"[{i + 1}/{len(stations)}] station_id={station_id} ({station_code}, {row['station name']})...")

        component_frames = {}
        for label, comp_id in COMPONENTS.items():
            df = fetch_component(station_id, comp_id)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if df is None:
                continue

            if label not in component_frames:
                print(f"    {label} columns: {list(df.columns)}")

            component_frames[label] = df

        if not component_frames:
            print(f"    no PM10/PM2.5 data for this station, skipping")  # normal, not every station tracks PM
            continue

        for label, df in component_frames.items():
            df = df.rename(columns={df.columns[-1]: label})
            df["station_code"] = station_code
            df["station_name"] = row["station name"]
            df["lat"] = row["station latitude"]
            df["lon"] = row["station longitude"]
            results.append(df)

    if not results:
        print("No PM10/PM2.5 data retrieved for any station. Nothing to save.")
        return

    combined = pd.concat(results, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
