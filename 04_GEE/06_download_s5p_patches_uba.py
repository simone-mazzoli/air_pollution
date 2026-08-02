"""
Downloads Sentinel-5P (TROPOMI) atmospheric-composition patches per UBA reference
station, as extra wide-context input streams for the CNN -- same products/logic as
download_s5p_patches.py, just for UBA stations instead of low-cost sensors, same as
download_satellite_patches_uba.py mirrors download_satellite_patches.py.

PRODUCTS (physical relevance to PM):
  no2  -- tropospheric NO2 column: combustion tracer, co-varies with PM sources.
  aer  -- UV Aerosol Index: a DIRECT satellite measure of absorbing aerosols /
          particulates, the most physically relevant S5P signal for PM itself.
  co   -- carbon monoxide column: combustion tracer, co-varies with PM.

FORMAT: same 3x3 grid at ~7 km/px (21 km footprint) as the sensor version -- S5P's
predictive content over any single point is essentially one column value, not
spatial texture, so a small native-resolution grid captures it without wasting
transfer on interpolated pixels.

COMPOSITE: quality/cloud-filtered annual mean, same as the sensor version -- L3
OFFL products are already ESA-quality-screened.

Station coordinates come from the same UBA daily file
calibrate_pm_leave_one_fold_out.py and download_satellite_patches_uba.py read
(station_code, lat, lon columns) -- not from the low-cost sensor annual files.

Input:  client_id_GEE.txt
        data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
Output: data/processed/satellite_uba/<stream>/<station_code>.npy   (stream per product)
        data/processed/satellite_uba/manifest_<product>.csv
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
CREDENTIALS_PATH = BASE_DIR / "client_id_GEE.txt"
PROJECT_ID = "air-pollution-501614"
YEAR = 2024
DATE_FROM, DATE_TO = f"{YEAR}-01-01", f"{YEAR}-12-31"

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}

# product -> (GEE collection, band, output-stream name). All L3 OFFL, already
# quality-screened (no qa_value band to mask). Same as download_s5p_patches.py.
PRODUCTS = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density", "no2_tropomi"),
    "aer": ("COPERNICUS/S5P/OFFL/L3_AER_AI",
            "absorbing_aerosol_index", "aer_tropomi"),
    "co":  ("COPERNICUS/S5P/OFFL/L3_CO",
            "CO_column_number_density", "co_tropomi"),
}

# same reasoning as download_s5p_patches.py: S5P's ~7km native resolution means a
# small native-res grid captures the real signal without upsampling waste.
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
    """UBA reference station coordinates -- same source file
    calibrate_pm_leave_one_fold_out.py's load_uba() and
    download_satellite_patches_uba.py read (station_code, lat, lon)."""
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
        out_path = out_dir / f"{row['station_code']}.npy"
        if out_path.exists() and not force:
            skipped += 1
            if skipped % 500 == 0:
                print(f"    ...skipped {skipped} already-downloaded")
            continue
        arr = fetch_patch(ee, image, band, row["lat"], row["lon"])
        if arr is None:
            manifest_rows.append({"station_code": row["station_code"], "stream": stream,
                                  "status": "failed", "nodata_frac": None})
            continue
        frac = nodata_fraction(arr)
        # AER_AI can be legitimately negative (index, not a concentration); for it,
        # only non-finite counts as nodata.
        if product == "aer":
            frac = float(np.mean(~np.isfinite(arr)))
        if frac > MAX_NODATA_FRAC:
            manifest_rows.append({"station_code": row["station_code"], "stream": stream,
                                  "status": "corrupted", "nodata_frac": frac})
            continue
        np.save(out_path, arr)
        manifest_rows.append({"station_code": row["station_code"], "stream": stream,
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
    print(f"{len(locations)} UBA stations, products: {args.product}\n")

    for product in args.product:
        run_product(ee, product, locations, args.force)
    print("Done.")


if __name__ == "__main__":
    main()
