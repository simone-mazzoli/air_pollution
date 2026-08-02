"""
Downloads daily PM10/PM2.5 means from UBA reference stations.
Input:  data/processed/uba_stations_germany.csv
Output: data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
"""
import argparse
import time
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
YEAR = 2024
BASE_DIR = Path(__file__).resolve().parent.parent
STATIONS_PATH = BASE_DIR / "data" / "processed" / "uba_stations_germany.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "daily_avg" / "uba"
OUT_PATH = OUT_DIR / f"pm_reference_stations_{YEAR}.csv"
STATION_METADATA_PATH = BASE_DIR / "data" / "processed" / "uba" / "station_metadata.csv"
# old api-proxy endpoint 500s now, this one works
MEASURES_URL = "https://luftdaten.umweltbundesamt.de/api/air-data/v2/measures/csv"
V4_PROXY = "https://luftdaten.umweltbundesamt.de/api-proxy"
V4_META_URL = (
    f"{V4_PROXY}/meta/json?use=airquality&date_from={YEAR}-01-01&date_to={YEAR}-12-31"
    "&time_from=1&time_to=24&lang=de"
)
V4_RESOURCE_URLS = {
    "components": f"{V4_PROXY}/components/json",
    "scopes": f"{V4_PROXY}/scopes/json",
    "stations": f"{V4_PROXY}/stations/json?lang=de",
    "stationtypes": f"{V4_PROXY}/stationtypes/json?lang=de",
    "stationsettings": f"{V4_PROXY}/stationsettings/json?lang=de",
}
COMPONENTS = {"PM10": 1, "PM2.5": 9}
SCOPE_DAILY_MEAN = 1
REQUEST_TIMEOUT = 120
RETRY_DELAY_SECONDS = 3
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UBA PM measurements or station metadata.")
    parser.add_argument(
        "--station-metadata",
        action="store_true",
        help="Fetch UBA Air Data API v4 station metadata only.",
    )
    return parser.parse_args()


def records_from_payload(payload: dict) -> pd.DataFrame:
    indices = payload.get("indices")
    data = payload.get("data")
    if not isinstance(indices, list) or data is None:
        return pd.DataFrame()
    values = data.values() if isinstance(data, dict) else data
    rows = [dict(zip(indices, row)) for row in values if isinstance(row, list)]
    return pd.DataFrame(rows)


def fetch_json(url: str) -> tuple[pd.DataFrame, dict]:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            frame = records_from_payload(payload) if isinstance(payload, dict) else pd.DataFrame()
            print(f"{url} -> status={response.status_code}, rows={len(frame)}, columns={list(frame.columns)}")
            return frame, payload if isinstance(payload, dict) else {"raw": payload}
        except requests.RequestException as exc:
            last_exc = exc
            print(f"{url} attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_exc if last_exc else RuntimeError(f"Failed to fetch {url}")


def first_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for col in frame.columns:
        text = col.lower()
        if all(part in text for part in candidates[0].lower().split()):
            return col
    return None


def classify_pm_availability_from_local_daily() -> pd.DataFrame:
    daily_path = BASE_DIR / "data" / "processed" / "daily_avg" / "uba" / f"pm_reference_stations_{YEAR}.csv"
    if not daily_path.exists():
        return pd.DataFrame(columns=["station_code", "PM10_available_2024", "PM2.5_available_2024", "available_scopes"])
    daily = pd.read_csv(daily_path)
    rows = []
    for code, group in daily.groupby("station_code"):
        pm10 = bool(pd.to_numeric(group.get("PM10"), errors="coerce").notna().any())
        pm25 = bool(pd.to_numeric(group.get("PM2.5"), errors="coerce").notna().any())
        scopes = []
        if pm10 or pm25:
            scopes.append("1")
        rows.append(
            {
                "station_code": code,
                "PM10_available_2024": pm10,
                "PM2.5_available_2024": pm25,
                "available_scopes": ",".join(scopes),
            }
        )
    return pd.DataFrame(rows)


def station_metadata() -> None:
    meta_frame, _ = fetch_json(V4_META_URL)
    resource_frames = {}
    for name, url in V4_RESOURCE_URLS.items():
        try:
            resource_frames[name], _ = fetch_json(url)
        except requests.RequestException as exc:
            print(f"{url} unavailable: {type(exc).__name__}: {exc}")
            resource_frames[name] = pd.DataFrame()

    stations = resource_frames.get("stations", pd.DataFrame())
    if stations.empty:
        stations = meta_frame
    if stations.empty:
        raise RuntimeError("UBA v4 metadata returned no station rows")

    code_col = first_col(stations, ["station code", "station_code", "code"])
    id_col = first_col(stations, ["station id", "station_id", "id"])
    name_col = first_col(stations, ["station name", "station_name", "name"])
    lon_col = first_col(stations, ["station longitude", "longitude", "lon"])
    lat_col = first_col(stations, ["station latitude", "latitude", "lat"])
    network_col = first_col(stations, ["network name", "network_name", "network"])
    type_id_col = first_col(stations, ["station type id", "station_type_id"])
    type_label_col = first_col(stations, ["station type name", "station_type_name"])
    setting_id_col = first_col(stations, ["station setting id", "station_setting_id"])
    setting_label_col = first_col(stations, ["station setting name", "station_setting_name"])

    out = pd.DataFrame()
    out["station_code"] = stations[code_col] if code_col else pd.NA
    out["uba_station_id"] = stations[id_col] if id_col else pd.NA
    out["station_name"] = stations[name_col] if name_col else pd.NA
    out["longitude"] = pd.to_numeric(stations[lon_col], errors="coerce") if lon_col else pd.NA
    out["latitude"] = pd.to_numeric(stations[lat_col], errors="coerce") if lat_col else pd.NA
    out["network_or_land"] = stations[network_col] if network_col else pd.NA
    out["station_type_id"] = stations[type_id_col] if type_id_col else pd.NA
    out["station_type_label"] = stations[type_label_col] if type_label_col else pd.NA
    out["station_setting_id"] = stations[setting_id_col] if setting_id_col else pd.NA
    out["station_setting_label"] = stations[setting_label_col] if setting_label_col else pd.NA

    availability = classify_pm_availability_from_local_daily()
    out = out.merge(availability, on="station_code", how="left")
    for col in ("PM10_available_2024", "PM2.5_available_2024"):
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    out["available_scopes"] = out["available_scopes"].fillna("")
    out["metadata_source"] = "UBA Air Data API v4 meta/json plus stations resources; PM availability from local daily UBA file"
    out["retrieval_date"] = datetime.now(timezone.utc).date().isoformat()
    out = out.dropna(subset=["station_code"]).drop_duplicates("station_code")
    STATION_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATION_METADATA_PATH, index=False)
    print(f"Saved -> {STATION_METADATA_PATH} ({len(out)} rows)")


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
    args = parse_args()
    if args.station_metadata:
        station_metadata()
        return
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
            # FIX: the real numeric value is in "Messwert" -- the trailing column
            # is "Einheit" (the unit, e.g. "µg/m³"), not the value. Renaming the
            # last column previously grabbed the unit string instead.
            df["value"] = pd.to_numeric(df["Messwert"], errors="coerce")
            # FIX: keep long-format rows tagged by pollutant instead of stacking
            # two same-shaped frames both claiming a "PM10"/"PM2.5" column --
            # that produced two disjoint row-sets instead of one row per
            # station/date with both pollutants filled in.
            df = df[["Datum", "value"]].copy()
            df["pollutant"] = label
            df["station_code"] = station_code
            df["station_name"] = row["station name"]
            df["lat"] = row["station latitude"]
            df["lon"] = row["station longitude"]
            results.append(df)
    if not results:
        print("No PM10/PM2.5 data retrieved for any station. Nothing to save.")
        return
    combined = pd.concat(results, ignore_index=True)
    # FIX: pivot long -> wide so PM10 and PM2.5 end up as two real columns on
    # the same (station, date) row, which is what load_uba() in correct_pm.py
    # expects.
    combined = combined.pivot_table(
        index=["station_code", "station_name", "lat", "lon", "Datum"],
        columns="pollutant",
        values="value",
        aggfunc="first",
    ).reset_index()
    for col in ("PM10", "PM2.5"):
        if col not in combined.columns:
            combined[col] = pd.NA
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH} ({len(combined)} rows)")
if __name__ == "__main__":
    main()
