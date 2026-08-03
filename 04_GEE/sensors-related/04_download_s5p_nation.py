"""
Wide-context S5P streams WITHOUT per-sensor downloads.

The wide window (~105 km at ~7 km/px) overlaps almost completely between
neighbouring sensors, so per-sensor fetching would download the same pixels
thousands of times. Instead: fetch ONE Germany-wide annual raster per product
(a single computePixels call of ~170x200 px -- less transfer than a few dozen
of the existing 3x3 fetches), then crop each sensor's window locally.

Two phases, resumable independently:
  --download   fetch + save the national raster(s):
                 data/processed/satellite/national_<product>.npy
                 data/processed/satellite/national_<product>.json   (grid meta)
  --crop       crop per-sensor windows from the saved raster(s):
                 data/processed/satellite/<stream>_wide/<location>.npy
               (matches what 03_train_resnet.py --s5p-wide expects)

Re-cropping at a different --px never re-downloads. Windows extending past the
raster edge are NaN-padded; the training script z-scores with nan->0, and the
raster is buffered 150 km beyond Germany's bbox so this only affects a handful
of border sensors anyway.

Usage:
  python3 05_s5p_wide.py --product no2 aer co --download --crop
  python3 05_s5p_wide.py --product no2 aer co --crop --px 21   # re-crop only
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

# Germany bbox + buffer so border sensors get full windows
BBOX = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}
BUFFER_M = 150_000

PRODUCTS = {
    "no2": ("COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density", "no2_tropomi"),
    "aer": ("COPERNICUS/S5P/OFFL/L3_AER_AI",
            "absorbing_aerosol_index", "aer_tropomi"),
    "co":  ("COPERNICUS/S5P/OFFL/L3_CO",
            "CO_column_number_density", "co_tropomi"),
}

SCALE_M = 7000            # ~S5P native resolution
DEFAULT_PX = 15           # 15 px * 7 km = 105 km window
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
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
        raise FileNotFoundError("no calibrated annual files found -- run calibration first")
    df = pd.concat([pd.read_csv(f) for f in annual_files], ignore_index=True)
    df = df.drop_duplicates(subset="location")
    print(f"{len(df)} calibrated sensors (union across {len(annual_files)} annual file(s))")
    return df[["location", "lat", "lon"]].reset_index(drop=True)


def national_grid():
    """Web-mercator grid covering buffered Germany at SCALE_M m/px."""
    x0, y0 = lonlat_to_webmercator(BBOX["lon_min"], BBOX["lat_min"])
    x1, y1 = lonlat_to_webmercator(BBOX["lon_max"], BBOX["lat_max"])
    x_min, x_max = x0 - BUFFER_M, x1 + BUFFER_M
    y_min, y_max = y0 - BUFFER_M, y1 + BUFFER_M
    width = int(math.ceil((x_max - x_min) / SCALE_M))
    height = int(math.ceil((y_max - y_min) / SCALE_M))
    return {"x_min": x_min, "y_max": y_max, "scale": SCALE_M,
            "width": width, "height": height}


def download_national(ee, product):
    collection, band, stream = PRODUCTS[product]
    g = national_grid()
    region = ee.Geometry.Rectangle(
        [g["x_min"], g["y_max"] - g["height"] * g["scale"],
         g["x_min"] + g["width"] * g["scale"], g["y_max"]],
        proj="EPSG:3857", geodesic=False)
    image = (ee.ImageCollection(collection)
             .filterDate(DATE_FROM, DATE_TO)
             .select(band)
             .mean())
    request = {
        "expression": image.clip(region),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": g["width"], "height": g["height"]},
            "affineTransform": {
                "scaleX": g["scale"], "scaleY": -g["scale"],
                "translateX": g["x_min"], "translateY": g["y_max"],
            },
            "crsCode": "EPSG:3857",
        },
    }
    print(f"=== national raster: {stream} ({g['width']}x{g['height']} px "
          f"@ {g['scale']/1000:.0f} km/px, one request) ===")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = ee.data.computePixels(request)
            arr = np.asarray(raw[band], dtype="float32")
            break
        except Exception as exc:
            print(f"    attempt {attempt} failed: {exc}")
            if attempt == MAX_RETRIES:
                return False
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    finite = np.isfinite(arr)
    print(f"    shape {arr.shape}, finite {finite.mean():.1%}, "
          f"range [{np.nanmin(arr):.3g}, {np.nanmax(arr):.3g}]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"national_{product}.npy", arr)
    (OUT_DIR / f"national_{product}.json").write_text(json.dumps(g, indent=2))
    print(f"    saved national_{product}.npy + .json")
    return True


def crop_windows(product, locations, px):
    _, _, stream = PRODUCTS[product]
    rast_path = OUT_DIR / f"national_{product}.npy"
    meta_path = OUT_DIR / f"national_{product}.json"
    if not (rast_path.exists() and meta_path.exists()):
        print(f"  {product}: national raster missing -- run with --download first")
        return
    arr = np.load(rast_path)
    g = json.loads(meta_path.read_text())
    out_dir = OUT_DIR / f"{stream}_wide"
    out_dir.mkdir(parents=True, exist_ok=True)
    half = px // 2
    H, W = arr.shape
    n_pad = 0
    for _, row in locations.iterrows():
        x, y = lonlat_to_webmercator(row["lon"], row["lat"])
        c = int(round((x - g["x_min"]) / g["scale"] - 0.5))
        r = int(round((g["y_max"] - y) / g["scale"] - 0.5))
        win = np.full((px, px), np.nan, dtype="float32")
        r0, r1 = r - half, r - half + px
        c0, c1 = c - half, c - half + px
        sr0, sr1 = max(r0, 0), min(r1, H)
        sc0, sc1 = max(c0, 0), min(c1, W)
        if sr0 < sr1 and sc0 < sc1:
            win[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = arr[sr0:sr1, sc0:sc1]
        else:
            print(f"    WARNING: {row['location']} entirely outside raster")
        if not np.isfinite(win).all():
            n_pad += 1
        np.save(out_dir / f"{row['location']}.npy", win)
    print(f"  {stream}_wide: {len(locations)} windows of {px}x{px} px "
          f"({px * g['scale'] / 1000:.0f} km), {n_pad} with NaN padding/holes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", nargs="+", default=["no2"], choices=list(PRODUCTS.keys()))
    ap.add_argument("--download", action="store_true", help="fetch national raster(s)")
    ap.add_argument("--crop", action="store_true", help="crop per-sensor windows")
    ap.add_argument("--px", type=int, default=DEFAULT_PX,
                    help=f"window size in pixels (default {DEFAULT_PX} = "
                         f"{DEFAULT_PX * SCALE_M / 1000:.0f} km)")
    args = ap.parse_args()
    if not (args.download or args.crop):
        args.download = args.crop = True

    locations = load_locations() if args.crop else None
    ee = init_ee() if args.download else None

    for product in args.product:
        if args.download:
            ok = download_national(ee, product)
            if not ok:
                print(f"  {product}: download FAILED, skipping crop")
                continue
        if args.crop:
            crop_windows(product, locations, args.px)
    print("Done.")


if __name__ == "__main__":
    main()
