"""
Download satellite patches per EEA station for the CNN input streams.

Streams (choose any with --stream, gather across multiple days):
  high_res  Sentinel-2 multispectral, 120x120 @ 10 m/px  (1.2 km, local detail)
  low_res   Sentinel-2 multispectral, 60x60  @ 200 m/px  (12 km, wide context)
  no2       S5P tropospheric NO2 column, 5x5 @ 7 km/px    (35 km)
  aer       S5P UV aerosol index,        5x5 @ 7 km/px
  co        S5P CO column,               5x5 @ 7 km/px

Stations come from stations_to_download.csv (station_code, lat, lon). Patches are
keyed on station_code and written to satellite_eea/<stream>/<station_code>.npy.
Re-running skips existing patches unless --force.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
STATIONS_CSV = PROC / "eea" / "stations_to_download.csv"
OUT_DIR = PROC / "satellite_eea"
CREDENTIALS_PATH = BASE_DIR / "client_id_GEE.txt"
PROJECT_ID = "air-pollution-501614"

YEAR = 2024
DATE_FROM, DATE_TO = f"{YEAR}-01-01", f"{YEAR}-12-31"
MAX_CLOUD_PCT = 20
# EU-wide bounding box (was Germany-only); covers all EEA reporting countries
BBOX = {"lat_min": 34.0, "lat_max": 72.0, "lon_min": -25.0, "lon_max": 45.0}

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# stream -> config. S2 streams carry bands; S5P streams carry a collection+band.
STREAMS = {
    "high_res": {"kind": "s2", "bands": S2_BANDS, "footprint_m": 1200, "px": 120,
                 "folder": "high_res_multispec", "max_nodata": 0.10},
    "low_res":  {"kind": "s2", "bands": S2_BANDS, "footprint_m": 12000, "px": 60,
                 "folder": "low_res_multispec", "max_nodata": 0.10},
    "no2": {"kind": "s5p", "coll": "COPERNICUS/S5P/OFFL/L3_NO2",
            "band": "tropospheric_NO2_column_number_density",
            "folder": "no2_tropomi", "footprint_m": 35000, "px": 5, "max_nodata": 0.30},
    "aer": {"kind": "s5p", "coll": "COPERNICUS/S5P/OFFL/L3_AER_AI",
            "band": "absorbing_aerosol_index",
            "folder": "aer_tropomi", "footprint_m": 35000, "px": 5, "max_nodata": 0.30},
    "co":  {"kind": "s5p", "coll": "COPERNICUS/S5P/OFFL/L3_CO",
            "band": "CO_column_number_density",
            "folder": "co_tropomi", "footprint_m": 35000, "px": 5, "max_nodata": 0.30},
}

MAX_RETRIES, RETRY_DELAY = 3, 5
WEBMERCATOR_R = 6_378_137.0


def lonlat_to_webmercator(lon, lat):
    x = WEBMERCATOR_R * math.radians(lon)
    y = WEBMERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def init_ee():
    import ee
    if CREDENTIALS_PATH.exists():
        raw = CREDENTIALS_PATH.read_text().strip()
        try:
            key = json.loads(raw)
            cred = ee.ServiceAccountCredentials(key["client_email"], key_data=raw)
            ee.Initialize(cred)
            print(f"Authenticated as service account {key['client_email']}")
            return ee
        except (json.JSONDecodeError, KeyError):
            pass
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print(f"Authenticated via OAuth, project={PROJECT_ID}")
    return ee


def load_stations(limit, country):
    df = pd.read_csv(STATIONS_CSV, dtype={"station_code": str})
    df = df.dropna(subset=["lat", "lon"]).drop_duplicates("station_code")
    if country:
        codes = [c.upper() for c in country]
        df = df[df["station_code"].str[:2].isin(codes)]
        print(f"filtered to countries {codes}: {len(df)} stations")
    if limit:
        df = df.head(limit)
    print(f"{len(df)} stations from {STATIONS_CSV.name}\n")
    return df[["station_code", "lat", "lon"]].reset_index(drop=True)


def build_composite(ee, cfg):
    region = ee.Geometry.Rectangle([BBOX["lon_min"], BBOX["lat_min"],
                                    BBOX["lon_max"], BBOX["lat_max"]])
    if cfg["kind"] == "s2":
        coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(DATE_FROM, DATE_TO).filterBounds(region)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
                .select(cfg["bands"]))
        return coll.median()
    coll = (ee.ImageCollection(cfg["coll"]).filterDate(DATE_FROM, DATE_TO)
            .filterBounds(region).select(cfg["band"]))
    return coll.mean()


def fetch_patch(ee, image, cfg, lat, lon):
    x, y = lonlat_to_webmercator(lon, lat)
    half = cfg["footprint_m"] / 2
    px = cfg["px"]
    bands = cfg["bands"] if cfg["kind"] == "s2" else [cfg["band"]]
    region = ee.Geometry.Rectangle([x - half, y - half, x + half, y + half],
                                   proj="EPSG:3857", geodesic=False)
    request = {
        "expression": image.clip(region),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": px, "height": px},
            "affineTransform": {
                "scaleX": cfg["footprint_m"] / px, "scaleY": -cfg["footprint_m"] / px,
                "translateX": x - half, "translateY": y + half,
            },
            "crsCode": "EPSG:3857",
        },
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = ee.data.computePixels(request)
            return np.stack([raw[b] for b in bands], axis=-1).astype("float32")
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None


def nodata_fraction(arr, stream):
    if stream == "aer":                      # AER index can be legitimately negative
        return float(np.mean(~np.isfinite(arr)))
    return float(np.mean(~np.isfinite(arr) | (arr <= 0)))


def run_stream(ee, stream, stations, force):
    cfg = STREAMS[stream]
    out_dir = OUT_DIR / cfg["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / f"manifest_{stream}.csv"
    rows = pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
    image = build_composite(ee, cfg)

    print(f"=== {stream} -> {cfg['folder']} "
          f"({cfg['px']}x{cfg['px']}px, {cfg['footprint_m']/1000:.0f} km) ===")
    n = len(stations)
    fetched = skipped = failed = 0
    for i, row in stations.iterrows():
        code = row["station_code"]
        out_path = out_dir / f"{code}.npy"
        if out_path.exists() and not force:
            skipped += 1
            continue
        arr = fetch_patch(ee, image, cfg, row["lat"], row["lon"])
        if arr is None:
            rows.append({"station_code": code, "stream": stream, "status": "failed",
                         "nodata_frac": None})
            failed += 1
            print(f"  [{i+1}/{n}] {code}: FAILED")
            continue
        frac = nodata_fraction(arr, stream)
        if frac > cfg["max_nodata"]:
            rows.append({"station_code": code, "stream": stream, "status": "corrupted",
                         "nodata_frac": frac})
            print(f"  [{i+1}/{n}] {code}: dropped (nodata {frac:.0%})")
            continue
        np.save(out_path, arr)
        rows.append({"station_code": code, "stream": stream, "status": "ok",
                     "nodata_frac": frac})
        fetched += 1
        print(f"  [{i+1}/{n}] {code}: ok (nodata {frac:.0%})")
        if fetched % 25 == 0:
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            print(f"    -- progress: {fetched} fetched, {skipped} skipped, "
                  f"{failed} failed --")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    ok = sum(1 for r in rows if r["stream"] == stream and r["status"] == "ok")
    print(f"  {stream}: {ok} patches on disk  "
          f"({fetched} new, {skipped} already there, {failed} failed)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", nargs="+", required=True, choices=list(STREAMS),
                    help="which streams to fetch, e.g. --stream high_res low_res no2")
    ap.add_argument("--country", nargs="+", default=None,
                    help="only stations in these countries, e.g. --country AD")
    ap.add_argument("--force", action="store_true", help="re-fetch existing patches")
    ap.add_argument("--limit", type=int, default=None, help="first N stations (testing)")
    args = ap.parse_args()

    ee = init_ee()
    stations = load_stations(args.limit, args.country)
    for stream in args.stream:
        run_stream(ee, stream, stations, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
