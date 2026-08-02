"""
Downloads Sentinel-2 patches per UBA reference station location, for the same two
CNN input streams as download_satellite_patches.py (high-res local detail /
low-res wide context). Identical download logic to that script -- projection,
cloud filtering, retries, band selection -- the only thing that changes is which
points get patches: UBA reference stations instead of low-cost PM sensors.

Multispectral bands, cloud filtering, and coordinate handling are unchanged from
download_satellite_patches.py -- see that script's docstring for the reasoning.

Station coordinates come from the same UBA daily file
active/03_calibrate_pm_loo.py reads (station_code, lat, lon columns), not
from station_land.csv -- station_land.csv only has station_code+land, no
coordinates.

skips a (station_code, stream) pair if its .npy already exists, unless --force.

Input:  client_id_GEE.txt           (GEE service account JSON key, if used)
        data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
Output: data/processed/satellite_uba/<stream>/<station_code>.npy
        data/processed/satellite_uba/manifest.csv   (station_code, stream, status, nodata_frac)
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
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"
OUT_DIR = PROC / "satellite_uba"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
CREDENTIALS_PATH = BASE_DIR / "client_id_GEE.txt"  # only read if it's a service account JSON key
PROJECT_ID = "air-pollution-501614"  # GCP project Earth Engine usage is billed/scoped to
YEAR = 2024
DATE_FROM, DATE_TO = f"{YEAR}-01-01", f"{YEAR}-12-31"
MAX_CLOUD_PCT = 20
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}
# fraction of masked/nodata pixels above which a patch is dropped as corrupted
MAX_NODATA_FRAC = 0.10
# same 10/13 bands as download_satellite_patches.py -- excludes B1 (coastal
# aerosol), B9 (water vapour), B10 (cirrus): 60m atmospheric-correction bands,
# not surface features.
MULTISPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
RGB_SLICE_INDEX = [
    MULTISPECTRAL_BANDS.index("B4"),
    MULTISPECTRAL_BANDS.index("B3"),
    MULTISPECTRAL_BANDS.index("B2"),
]
# same footprint/px config as download_satellite_patches.py, so UBA patches are
# directly comparable to sensor patches
STREAMS = {
    "high_res_multispec": {
        "bands": MULTISPECTRAL_BANDS, "footprint_m": 1200, "px": 120,
    },
    "low_res_multispec": {
        "bands": MULTISPECTRAL_BANDS, "footprint_m": 12000, "px": 60,
    },
}
RETRY_DELAY_SECONDS = 5
MAX_RETRIES = 3
WEBMERCATOR_R = 6_378_137.0  # exact, this is how EPSG:3857 is defined


def lonlat_to_webmercator(lon, lat):
    """Exact closed-form WGS84 -> EPSG:3857 forward projection. Not an
    approximation -- this IS the definition of Web Mercator, so it's correct at
    every latitude, unlike a constant meters-per-degree conversion."""
    x = WEBMERCATOR_R * math.radians(lon)
    y = WEBMERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def rgb_from_multispec(arr):
    """Slices B4/B3/B2 out of a multispectral array -- (H, W, 10) -> (H, W, 3).
    Use this for figures/previews instead of downloading a separate RGB stream."""
    return arr[..., RGB_SLICE_INDEX]


def init_ee():
    import ee
    if CREDENTIALS_PATH.exists():
        raw = CREDENTIALS_PATH.read_text().strip()
        try:
            key = json.loads(raw)
            credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=raw)
            ee.Initialize(credentials)
            print(f"Authenticated as service account {key['client_email']}")
            return ee
        except (json.JSONDecodeError, KeyError):
            pass  # not a service account key -- fall through to interactive OAuth below
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print(f"Authenticated via OAuth, project={PROJECT_ID}")
    return ee


def load_locations():
    """UBA reference station coordinates -- same source file
    active/03_calibrate_pm_loo.py's load_uba() reads (station_code, lat, lon),
    deduplicated to one row per station_code."""
    path = Path(str(UBA_DAILY).format(year=YEAR))
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- this is the same file the calibration script "
            f"reads, run whatever produces it first")
    df = pd.read_csv(path)
    missing_cols = {"station_code", "lat", "lon"} - set(df.columns)
    if missing_cols:
        raise KeyError(f"{path} is missing expected column(s) {missing_cols}")
    df = df.drop_duplicates(subset="station_code")
    print(f"{len(df)} UBA reference stations")
    return df[["station_code", "lat", "lon"]].reset_index(drop=True)


def build_composite(ee, bands):
    """Cloud-filtered, year-long median composite over Germany. Built once and
    reused for every point."""
    region = ee.Geometry.Rectangle([
        BBOX_GERMANY["lon_min"], BBOX_GERMANY["lat_min"],
        BBOX_GERMANY["lon_max"], BBOX_GERMANY["lat_max"],
    ])
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_FROM, DATE_TO)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .select(bands)
    )
    return coll.median()


def fetch_patch(ee, image, lat, lon, footprint_m, px, bands):
    """Pulls one patch as a numpy array via computePixels. Grid and clip region
    are both built in exact EPSG:3857 meters, so scaleX/scaleY are true
    meters-per-pixel and the patch comes out square and correctly sized.
    Returns array shaped (px, px, len(bands))."""
    x, y = lonlat_to_webmercator(lon, lat)
    half = footprint_m / 2
    region = ee.Geometry.Rectangle(
        [x - half, y - half, x + half, y + half], proj="EPSG:3857", geodesic=False
    )
    request = {
        "expression": image.clip(region),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": px, "height": px},
            "affineTransform": {
                "scaleX": footprint_m / px, "scaleY": -footprint_m / px,
                "translateX": x - half, "translateY": y + half,
            },
            "crsCode": "EPSG:3857",
        },
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = ee.data.computePixels(request)
            arr = np.stack([raw[b] for b in bands], axis=-1).astype("float32")
            return arr
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    return None


def nodata_fraction(arr):
    return float(np.mean(~np.isfinite(arr) | (arr <= 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", default=list(STREAMS),
                       help=f"Which streams to fetch (default: all {list(STREAMS)})")
    parser.add_argument("--force", action="store_true",
                       help="Re-fetch even if a patch file already exists")
    parser.add_argument("--limit", type=int, default=None,
                       help="Only process the first N stations (for testing)")
    args = parser.parse_args()
    ee = init_ee()
    locations = load_locations()
    if args.limit:
        locations = locations.head(args.limit)
    print(f"{len(locations)} UBA stations to process\n")
    composites = {name: build_composite(ee, cfg["bands"])
                  for name, cfg in STREAMS.items() if name in args.streams}
    manifest_rows = []
    if MANIFEST_PATH.exists():
        manifest_rows = pd.read_csv(MANIFEST_PATH).to_dict("records")
    for stream_name in args.streams:
        cfg = STREAMS[stream_name]
        out_dir = OUT_DIR / stream_name
        out_dir.mkdir(parents=True, exist_ok=True)
        image = composites[stream_name]
        print(f"=== {stream_name} ({len(cfg['bands'])} bands, "
              f"{cfg['footprint_m']}m footprint, {cfg['px']}x{cfg['px']}px) ===")
        for i, row in locations.iterrows():
            out_path = out_dir / f"{row['station_code']}.npy"
            if out_path.exists() and not args.force:
                continue
            arr = fetch_patch(ee, image, row["lat"], row["lon"],
                              cfg["footprint_m"], cfg["px"], cfg["bands"])
            if arr is None:
                manifest_rows.append({"station_code": row["station_code"], "stream": stream_name,
                                      "status": "failed", "nodata_frac": None})
                continue
            frac = nodata_fraction(arr)
            if frac > MAX_NODATA_FRAC:
                manifest_rows.append({"station_code": row["station_code"], "stream": stream_name,
                                      "status": "corrupted", "nodata_frac": frac})
                continue
            np.save(out_path, arr)
            manifest_rows.append({"station_code": row["station_code"], "stream": stream_name,
                                  "status": "ok", "nodata_frac": frac})
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(locations)} processed")
                pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
        pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
        ok = sum(1 for r in manifest_rows if r["stream"] == stream_name and r["status"] == "ok")
        print(f"  {stream_name}: {ok}/{len(locations)} patches saved\n")
    print(f"Done. Manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
