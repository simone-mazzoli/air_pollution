"""
Fetch the full UBA (Umweltbundesamt) air quality reference station list
and filter it down to stations within Germany's bounding box.

This uses the public UBA "stations" endpoint, which returns every station
in the national network (Bund + Laender), regardless of which pollutant
it measures. PM10/PM2.5-specific filtering happens in a later step, once
we know which stations actually report those components.

Produces:
    <project_root>/data/processed/uba_stations_germany.csv
"""

import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "uba_stations_germany.csv"

STATIONS_URL = (
    "https://www.umweltbundesamt.de/api/air_data/v2/stations/json"
    "?use=airquality&lang=de"
    "&date_from=2024-01-01&date_to=2024-12-31"
    "&time_from=1&time_to=24"
)

# Same bounding box as the Sensor.Community processing script, kept
# consistent so both datasets cover the same area.
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}


def main() -> None:
    print(f"Fetching station list from UBA API...")
    response = requests.get(STATIONS_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    indices = payload["indices"]  # column names, in order
    stations = payload["data"]    # dict of station_id -> list of values

    rows = [dict(zip(indices, values)) for values in stations.values()]
    df = pd.DataFrame(rows)

    df["station longitude"] = pd.to_numeric(df["station longitude"], errors="coerce")
    df["station latitude"] = pd.to_numeric(df["station latitude"], errors="coerce")

    print(f"Total stations in network: {len(df)}")

    mask = (
        df["station latitude"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"])
        & df["station longitude"].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])
    )
    df_filtered = df.loc[mask].copy()
    print(f"Stations within bounding box: {len(df_filtered)}")

    keep_cols = [
        "station id", "station code", "station name", "station city",
        "station latitude", "station longitude",
        "network name", "station setting name", "station type name",
        "station active from", "station active to",
    ]
    df_filtered = df_filtered[keep_cols]

    # Drop stations that were decommissioned before your target year, if
    # you only care about stations active during a specific period --
    # here 2024 is used as an example; adjust to match your project year.
    active_mask = (
        df_filtered["station active to"].isna()
        | (pd.to_datetime(df_filtered["station active to"]) >= "2024-01-01")
    )
    df_active = df_filtered.loc[active_mask].copy()
    print(f"Stations active during 2024: {len(df_active)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_active.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print(df_active.head())


if __name__ == "__main__":
    main()
