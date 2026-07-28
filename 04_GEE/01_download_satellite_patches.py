"""
Downloads Sentinel-2 patches per PM sensor location for the CNN's two input
streams: high-res multispectral (local detail) and low-res multispectral (wide
context).

Multispectral bands are the 10/13 the BigEarthNet models actually use -- B01
(coastal aerosol), B09 (water vapour), and B10 (cirrus) are all 60m atmospheric-
correction bands, not surface features, so they're excluded.

Cloud Filtering:
1st step:
CLOUDY_PIXEL_PERCENTAGE < 20 throws out any satellite pass where more than 20% of the scene was cloudy
2nd Step:
pixel-level median. For each sensor location, Sentinel-2 flew 70 times in 2024. For every single pixel, we take the median value across those 40 images. 

Each resulting patch is additionally
checked for nodata/masked fraction after download; 

COORDINATE HANDLING: sensor coordinates are WGS84 (EPSG:4326) lat/lon. Patch
grids are requested in EPSG:3857 so lon/lat are
converted to exact Web Mercator x/y via the closed-form projection formula
before building the grid.

High-res are 120x120px 10m/p (1.2x1.2km).
low-res are 60x60 200m/p (12x12km)

skips a (location, stream) pair if its .npy already exists, unless --force. 

Input:  client_id_GEE.txt           (GEE service account JSON key, if used)
        data/processed/hourly/pm/nodes/*.parquet   (unique sensor locations)
Output: data/processed/satellite/<stream>/<location>.npy
        data/processed/satellite/manifest.csv      (location, stream, status, nodata_frac)
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
NODES_DIR = PROC / "hourly" / "pm" / "nodes"
OUT_DIR = PROC / "satellite"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
CREDENTIALS_PATH = BASE_DIR / "client_id_GEE.txt"  # only read if it's a service account JSON key
PROJECT_ID = "air-pollution-501614"  # GCP project Earth Engine usage is billed/scoped to
YEAR = 2024
DATE_FROM, DATE_TO = f"{YEAR}-01-01", f"{YEAR}-12-31"
MAX_CLOUD_PCT = 20
BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}
# fraction of masked/nodata pixels above which a patch is dropped as corrupted
MAX_NODATA_FRAC = 0.10
# 10/13 bands: excludes B1 (coastal aerosol), B9 (water vapour), B10 (cirrus) --
# all 60m atmospheric-correction bands, not surface features. 
MULTISPECTRAL_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# indices of B4 (red), B3 (green), B2 (blue) within MULTISPECTRAL_BANDS, for
# slicing an RGB preview out of the multispectral array -- no separate download
RGB_SLICE_INDEX = [
    MULTISPECTRAL_BANDS.index("B4"),
    MULTISPECTRAL_BANDS.index("B3"),
    MULTISPECTRAL_BANDS.index("B2"),
]
# physical footprint in meters, output array size in pixels, per stream.
# Two streams high/low:
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
    """
    Only the sensors that survived the annual completeness filter (>=182 valid
    days, i.e. the 50% threshold) 
    """
    annual_files = list((PROC / "corrected" / "fold").glob("*/annual/*.csv"))
    if not annual_files:
        # single-file calibration fallback
        annual_files = list((PROC / "corrected" / "annual").glob("*.csv"))
    if not annual_files:
        raise FileNotFoundError(
            "no calibrated annual files found under corrected/fold/*/annual/ "
            "or corrected/annual/ -- run the calibration script first")
    df = pd.concat([pd.read_csv(f) for f in annual_files], ignore_index=True)
    df = df.drop_duplicates(subset="location")
    print(f"{len(df)} calibrated sensors (union across {len(annual_files)} annual file(s))")
    return df[["location", "lat", "lon"]].reset_index(drop=True)
def build_composite(ee, bands):
    """Cloud-filtered, year-long median composite over Germany. Built once and
    reused for every point 
    """
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
    """
    Pulls one patch as a numpy array via computePixels. Grid and clip region
    are both built in exact EPSG:3857 meters, so scaleX/scaleY are true meters-per-pixel and the patch comes out
    square and correctly sized. Returns array shaped (px, px, len(bands)
    """
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
                       help="Only process the first N locations (for testing)")
    args = parser.parse_args()
    ee = init_ee()
    locations = load_locations()
    if args.limit:
        locations = locations.head(args.limit)
    print(f"{len(locations)} sensor locations to process\n")
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
            out_path = out_dir / f"{row['location']}.npy"
            if out_path.exists() and not args.force:
                continue
            arr = fetch_patch(ee, image, row["lat"], row["lon"],
                              cfg["footprint_m"], cfg["px"], cfg["bands"])
            if arr is None:
                manifest_rows.append({"location": row["location"], "stream": stream_name,
                                      "status": "failed", "nodata_frac": None})
                continue
            frac = nodata_fraction(arr)
            if frac > MAX_NODATA_FRAC:
                manifest_rows.append({"location": row["location"], "stream": stream_name,
                                      "status": "corrupted", "nodata_frac": frac})
                continue
            np.save(out_path, arr)
            manifest_rows.append({"location": row["location"], "stream": stream_name,
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
