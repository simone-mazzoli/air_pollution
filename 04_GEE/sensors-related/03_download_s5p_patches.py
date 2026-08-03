"""
Downloads Sentinel-5P (TROPOMI) atmospheric-composition patches per PM sensor
location, as extra wide-context input streams for the CNN. Generalized over
multiple S5P products via --product, each written to its own stream folder so the
"with vs without" ablations stay clean and the S2 pipeline is untouched.

PRODUCTS (physical relevance to PM):
  no2  -- tropospheric NO2 column: combustion tracer, co-varies with PM sources.
          This is the stream the Scheibenreif et al. paper used.
  aer  -- UV Aerosol Index: a DIRECT satellite measure of absorbing aerosols /
          particulates, the most physically relevant S5P signal for PM itself.
  co   -- carbon monoxide column: combustion tracer, co-varies with PM.

FORMAT: S5P native resolution is ~7 km, and the tropospheric column is spatially
near-constant over a sensor footprint (measured std/mean < 0.4% across a patch). So
its predictive content is essentially ONE column value, not spatial texture. We fetch
a small 3x3 grid at ~7 km/px (21 km footprint) so every pixel is REAL data, capturing
the local value at ~576x less transfer than an upsampled 120x120 patch would cost. The
paper upsampled S5P onto the 120x120 S2 grid for architectural convenience, but that
adds only interpolated pixels, not information; 5x5-native preserves the identical
signal efficiently.

COMPOSITE: quality/cloud-filtered annual mean. The L3 OFFL products are already
ESA-quality-screened (no per-pixel qa band), so we simply average the year.

Input:  client_id_GEE.txt
        data/processed/corrected/fold/*/annual/*.csv   (calibrated sensor locations)
Output: data/processed/satellite/<stream>/<location>.npy   (stream per product)
        data/processed/satellite/manifest_<product>.csv
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
OUT_DIR = PROC / "satellite"
CREDENTIALS_PATH = BASE_DIR / "client_id_GEE.txt"
PROJECT_ID = "air-pollution-501614"
YEAR = 2024
DATE_FROM, DATE_TO = f"{YEAR}-01-01", f"{YEAR}-12-31"

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

# product -> (GEE collection, band, output-stream name). All L3 OFFL, already
# quality-screened (no qa_value band to mask).
PRODUCTS = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density", "no2_tropomi"),
    "aer": ("COPERNICUS/S5P/OFFL/L3_AER_AI",
            "absorbing_aerosol_index", "aer_tropomi"),
    "co":  ("COPERNICUS/S5P/OFFL/L3_CO",
            "CO_column_number_density", "co_tropomi"),
}

# EFFICIENT format: S5P native resolution is ~7 km, so its predictive content over a
# sensor is essentially ONE column value, not spatial texture. Downloading 120x120
# (14,400 px) upsampled from ~1 real value wastes ~576x the transfer for no extra
# information. We fetch a small 3x3 grid at ~7 km/px (21 km footprint) so every pixel
# is real data and the value is captured, at a tiny fraction of the download cost.
FOOTPRINT_M = 21000
PX = 3

MAX_NODATA_FRAC = 0.30
RETRY_DELAY_SECONDS = 5
MAX_RETRIES = 3
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
            credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=raw)
            ee.Initialize(credentials)
            print(f"Authenticated as service account {key['client_email']}")
            return ee
        except (json.JSONDecodeError, KeyError):
            pass
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print(f"Authenticated via OAuth, project={PROJECT_ID}")
    return ee


def load_locations():
    annual_files = list((PROC / "corrected" / "fold").glob("*/annual/*.csv"))
    if not annual_files:
        annual_files = list((PROC / "corrected" / "annual").glob("*.csv"))
    if not annual_files:
        raise FileNotFoundError(
            "no calibrated annual files found -- run the calibration script first")
    df = pd.concat([pd.read_csv(f) for f in annual_files], ignore_index=True)
    df = df.drop_duplicates(subset="location")
    print(f"{len(df)} calibrated sensors (union across {len(annual_files)} annual file(s))")
    return df[["location", "lat", "lon"]].reset_index(drop=True)


def build_composite(ee, collection, band):
    """Annual mean of the chosen S5P band over Germany. L3 products are already
    ESA-quality-screened, so a plain temporal mean is the standard column summary."""
    region = ee.Geometry.Rectangle([
        BBOX_GERMANY["lon_min"], BBOX_GERMANY["lat_min"],
        BBOX_GERMANY["lon_max"], BBOX_GERMANY["lat_max"],
    ])
    coll = (
        ee.ImageCollection(collection)
        .filterDate(DATE_FROM, DATE_TO)
        .filterBounds(region)
        .select(band)
    )
    return coll.mean()


def fetch_patch(ee, image, band, lat, lon):
    x, y = lonlat_to_webmercator(lon, lat)
    half = FOOTPRINT_M / 2
    region = ee.Geometry.Rectangle(
        [x - half, y - half, x + half, y + half], proj="EPSG:3857", geodesic=False
    )
    request = {
        "expression": image.clip(region),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": PX, "height": PX},
            "affineTransform": {
                "scaleX": FOOTPRINT_M / PX, "scaleY": -FOOTPRINT_M / PX,
                "translateX": x - half, "translateY": y + half,
            },
            "crsCode": "EPSG:3857",
        },
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = ee.data.computePixels(request)
            arr = np.stack([raw[band]], axis=-1).astype("float32")
            return arr
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    return None


def nodata_fraction(arr):
    return float(np.mean(~np.isfinite(arr) | (arr <= 0)))


def run_product(ee, product, locations, force):
    collection, band, stream = PRODUCTS[product]
    manifest_path = OUT_DIR / f"manifest_{product}.csv"
    out_dir = OUT_DIR / stream
    out_dir.mkdir(parents=True, exist_ok=True)

    image = build_composite(ee, collection, band)
    manifest_rows = []
    if manifest_path.exists():
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")

    mpx = FOOTPRINT_M / PX
    print(f"=== {stream} ({product.upper()}, {FOOTPRINT_M/1000:.1f}km footprint, "
          f"{PX}x{PX}px @ {mpx:.1f}m/px) ===")
    fetched = 0
    skipped = 0
    for i, row in locations.iterrows():
        out_path = out_dir / f"{row['location']}.npy"
        if out_path.exists() and not force:
            skipped += 1
            if skipped % 500 == 0:
                print(f"    ...skipped {skipped} already-downloaded")
            continue
        arr = fetch_patch(ee, image, band, row["lat"], row["lon"])
        if arr is None:
            manifest_rows.append({"location": row["location"], "stream": stream,
                                  "status": "failed", "nodata_frac": None})
            continue
        frac = nodata_fraction(arr)
        # AER_AI can be legitimately negative (index, not a concentration); for it,
        # only non-finite counts as nodata.
        if product == "aer":
            frac = float(np.mean(~np.isfinite(arr)))
        if frac > MAX_NODATA_FRAC:
            manifest_rows.append({"location": row["location"], "stream": stream,
                                  "status": "corrupted", "nodata_frac": frac})
            continue
        np.save(out_path, arr)
        manifest_rows.append({"location": row["location"], "stream": stream,
                              "status": "ok", "nodata_frac": frac})
        fetched += 1
        if fetched % 25 == 0:
            print(f"    {fetched} fetched ({skipped} skipped) of {len(locations)}")
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    ok = sum(1 for r in manifest_rows if r["stream"] == stream and r["status"] == "ok")
    print(f"  {stream}: {ok}/{len(locations)} patches saved  (manifest {manifest_path.name})\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", nargs="+", default=["no2"],
                        choices=list(PRODUCTS.keys()),
                        help="which S5P product(s) to download (e.g. --product no2 aer co)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ee = init_ee()
    locations = load_locations()
    if args.limit:
        locations = locations.head(args.limit)
    print(f"{len(locations)} sensor locations, products: {args.product}\n")

    for product in args.product:
        run_product(ee, product, locations, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
