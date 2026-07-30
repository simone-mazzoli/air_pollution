"""
Downloads the official BKG VG250 administrative boundaries (Bundeslaender +
Kreise) and caches the Land and Kreis layers as GeoJSON, reprojected to
WGS84 (EPSG:4326) to match the sensor lat/lon convention used everywhere
else in this pipeline.

Source: BKG (Bundesamt fuer Kartographie und Geodaesie), VG250 1:250 000,
Stand 01.01., native CRS ETRS89 / UTM zone 32N (EPSG:25832):
https://gdz.bkg.bund.de/index.php/default/verwaltungsgebiete-1-250-000-stand-01-01-vg250-01-01.html

The VG250_LAN and VG250_KRS layers each carry several rows per unit via the
GF ("Geofaktor") column -- coastal/water-area duplicates of the same unit.
GF == 4 keeps exactly one polygon per unit: 16 for Land, 401 for Kreis.

Kreis boundaries are what district-level socioeconomic data (Destatis/INKAR)
needs to be spatially joined against for the exposure analysis -- AGS
(Amtlicher Gemeindeschluessel) is the key both datasets share, so INKAR
tables can be merged on AGS directly without any coordinate matching.

Input:  none (downloads from BKG)
Output: data/raw/vg250/vg250_01-01.utm32s.shape.ebenen.zip  (cached download)
        data/processed/admin_boundaries/vg250_land.geojson  (16 Bundeslaender, EPSG:4326)
        data/processed/admin_boundaries/vg250_kreis.geojson (401 Kreise, EPSG:4326)
"""

import zipfile
from pathlib import Path

import geopandas as gpd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "vg250"
OUT_DIR = BASE_DIR / "data" / "processed" / "admin_boundaries"

VG250_URL = (
    "https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_0101/aktuell/"
    "vg250_01-01.utm32s.shape.ebenen.zip"
)
ZIP_PATH = RAW_DIR / "vg250_01-01.utm32s.shape.ebenen.zip"
EXTRACT_DIR = RAW_DIR / "vg250_ebenen_0101"

NATIVE_CRS = "EPSG:25832"  # ETRS89 / UTM zone 32N, as shipped by BKG
TARGET_CRS = "EPSG:4326"  # WGS84 -- matches sensor lat/lon across this pipeline

KEEP_COLS = ["AGS", "GEN", "BEZ", "NUTS", "geometry"]

LAYERS = {
    "VG250_LAN": OUT_DIR / "vg250_land.geojson",
    "VG250_KRS": OUT_DIR / "vg250_kreis.geojson",
}


def download_vg250() -> None:
    if ZIP_PATH.exists():
        print(f"Using cached download: {ZIP_PATH}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VG250 from BKG ({VG250_URL})...")
    resp = requests.get(VG250_URL, timeout=120)
    resp.raise_for_status()
    ZIP_PATH.write_bytes(resp.content)
    print(f"  saved {len(resp.content) / 1024**2:.1f} MiB -> {ZIP_PATH}")


def extract_vg250() -> None:
    if EXTRACT_DIR.exists():
        print(f"Already extracted: {EXTRACT_DIR}")
        return
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)


def process_layer(shp_name: str, out_path: Path) -> None:
    # the .cpg sidecar claims UTF-8 but the .dbf name fields are actually
    # Latin-1 (e.g. "Luebeck" comes out mojibake otherwise) -- override it
    gdf = gpd.read_file(EXTRACT_DIR / f"{shp_name}.shp", encoding="latin1")

    if gdf.crs is None:
        gdf = gdf.set_crs(NATIVE_CRS)
    assert gdf.crs.to_epsg() == 25832, f"unexpected CRS in {shp_name}: {gdf.crs}"

    gdf = gdf[gdf["GF"] == 4].copy()
    gdf = gdf.to_crs(TARGET_CRS)[KEEP_COLS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # GeoJSON driver refuses to overwrite in place
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"  {shp_name}: {len(gdf)} features -> {out_path}")


def main() -> None:
    download_vg250()
    extract_vg250()

    print(f"\nExtracting layers (native {NATIVE_CRS} -> {TARGET_CRS}):")
    for shp_name, out_path in LAYERS.items():
        process_layer(shp_name, out_path)

    print(
        "\nDone. Join key for district-level socioeconomic data "
        "(Destatis/INKAR): AGS (Amtlicher Gemeindeschluessel)."
    )


if __name__ == "__main__":
    main()