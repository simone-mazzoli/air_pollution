"""
Download satellite patches on a dense grid over the sealed-test German Laender
(east + north), for continuous pollution-map inference. Reuses the exact GEE
fetch logic from 00_download_patches.py.
Streams needed for inference (matches final_model.pt's cfg): high_res, low_res,
no2, co, aer_wide, dem. 

Grid: 10 km spacing over the union of the sealed-test
Laender, built from FAO/GAUL level-1 admin boundaries and filtered to points
that actually fall inside those Laender polygons (not just their bounding box).
10 km spacing: no2/co (S5P, ~7 km native pixels, 35 km footprint) and
aer_wide (~7 km native pixels, 217 km footprint) barely change between points
closer together than their native resolution. most of the map's fine spatial
detail will come from the 10 m S2 high-res stream and the DEM.
Ca. 170,000 km^2 of target held-out region

To run:
python3 02_download_dense_grid_patches.py --stream high_res
output:
dl_project/data/processed/satellite_grid/high_res_multispec/g000000.npy
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
DL_MODULE = Path(__file__).resolve().parent / "01_download_eea_patches.py" #to make sure its the same process
spec = importlib.util.spec_from_file_location("download_patches", DL_MODULE)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
GRID_STREAMS = ["high_res", "low_res", "no2", "co", "aer_wide", "dem"]
GRID_SPACING_KM = 10.0
OUT_GRID_CSV = D.PROC / "eea" / "grid_points.csv"
GRID_SAT_DIR = D.PROC / "satellite_grid"   # mirrors satellite_eea/, keyed by grid_id
# station_land.csv / 00_assign_folds.py spelling -> candidate FAO/GAUL ADM1_NAME
# spellings to try. GAUL's exact English/German mix varies by dataset version, so
# fetch_land_boundaries() prints what it actually has and what matched, for a
# sanity check.
TARGET_LAENDER = {
    "Brandenburg": ["Brandenburg"],
    "Mecklenburg-Vorpommern": ["Mecklenburg-Vorpommern", "Mecklenburg-West Pomerania"],
    "Sachsen": ["Sachsen", "Saxony"],
    "Sachsen-Anhalt": ["Sachsen-Anhalt", "Saxony-Anhalt"],
    "Thueringen": ["Thueringen", "Thuringia", "Thüringen"],
    "Berlin": ["Berlin"],
    "Hamburg": ["Hamburg"],
    "Bremen": ["Bremen"],
    "Niedersachsen": ["Niedersachsen", "Lower Saxony"],
    "Schleswig-Holstein": ["Schleswig-Holstein"],
}
def fetch_land_boundaries(ee):
    """FAO/GAUL level-1 polygons for the sealed-test Laender."""
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(
        ee.Filter.eq("ADM0_NAME", "Germany"))
    all_names = gaul.aggregate_array("ADM1_NAME").getInfo()
    print(f"GAUL Germany ADM1 names available: {sorted(all_names)}")
    matched = {}
    for land, variants in TARGET_LAENDER.items():
        hit = next((v for v in variants if v in all_names), None)
        if hit is None:
            print(f"  !! no GAUL match for {land} (tried {variants}) -- skipped, "
                  f"check spelling against the list above")
            continue
        matched[land] = hit
    geoms = {}
    for land, gaul_name in matched.items():
        geom = gaul.filter(ee.Filter.eq("ADM1_NAME", gaul_name)).geometry()
        geoms[land] = geom.getInfo()
    return geoms
def build_grid(land_geoms, spacing_km):
    """Regular lat/lon grid at ~spacing_km, kept only where it falls inside an
    actual Land polygon (shapely point-in-polygon, not just the bbox)."""
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    polys = {land: shape(g) for land, g in land_geoms.items()}
    union = unary_union(list(polys.values()))
    lon_min, lat_min, lon_max, lat_max = union.bounds
    lat0 = (lat_min + lat_max) / 2
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat0))
    dlat = spacing_km / km_per_deg_lat
    dlon = spacing_km / km_per_deg_lon
    lats = np.arange(lat_min, lat_max + dlat, dlat)
    lons = np.arange(lon_min, lon_max + dlon, dlon)
    rows, gid = [], 0
    for lat in lats:
        for lon in lons:
            pt = Point(lon, lat)
            for land, poly in polys.items():
                if poly.contains(pt):
                    rows.append({"grid_id": f"g{gid:06d}", "lat": lat, "lon": lon, "land": land})
                    gid += 1
                    break
    df = pd.DataFrame(rows)
    print(f"grid: {len(df)} points at ~{spacing_km:g} km spacing across "
          f"{df['land'].nunique()} Laender")
    for land, n in df["land"].value_counts().items():
        print(f"    {land:<22} {n:>5}")
    return df
def run_stream(ee, stream, points, force, coarse_cache_km):
    cfg = D.STREAMS[stream]
    out_dir = GRID_SAT_DIR / cfg["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = GRID_SAT_DIR / f"manifest_{stream}.csv"
    rows = pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
    image = D.build_composite(ee, cfg)
    use_cache = coarse_cache_km > 0 and cfg["footprint_m"] >= 30000
    cache = {}
    if use_cache:
        print(f"  (coarse cache on: reusing patches within ~{coarse_cache_km:g} km "
              f"for this large-footprint stream)")
    print(f"=== grid/{stream} -> {cfg['folder']} "
          f"({cfg['px']}x{cfg['px']}px, {cfg['footprint_m']/1000:.0f} km) ===")
    n = len(points)
    fetched = skipped = failed = reused = 0
    for i, row in points.iterrows():
        gid = row["grid_id"]
        out_path = out_dir / f"{gid}.npy"
        if out_path.exists() and not force:
            skipped += 1
            continue
        arr, cache_key = None, None
        if use_cache:
            cell = coarse_cache_km / 111.32
            cache_key = (round(row["lat"] / cell), round(row["lon"] / cell))
            if cache_key in cache:
                arr = cache[cache_key]
                reused += 1
        if arr is None:
            arr = D.fetch_patch(ee, image, cfg, row["lat"], row["lon"])
            if use_cache and arr is not None:
                cache[cache_key] = arr
        if arr is None:
            rows.append({"grid_id": gid, "stream": stream, "status": "failed", "nodata_frac": None})
            failed += 1
            print(f"  [{i+1}/{n}] {gid}: FAILED")
            continue
        frac = D.nodata_fraction(arr, cfg)
        if frac > cfg["max_nodata"]:
            rows.append({"grid_id": gid, "stream": stream, "status": "corrupted", "nodata_frac": frac})
            print(f"  [{i+1}/{n}] {gid}: dropped (nodata {frac:.0%})")
            continue
        np.save(out_path, arr)
        rows.append({"grid_id": gid, "stream": stream, "status": "ok", "nodata_frac": frac})
        fetched += 1
        print(f"  [{i+1}/{n}] {gid}: ok (nodata {frac:.0%})")
        if fetched % 20 == 0:
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            print(f"    -- progress: {fetched} fetched ({reused} reused), "
                  f"{skipped} skipped, {failed} failed --")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    ok = sum(1 for r in rows if r["stream"] == stream and r["status"] == "ok")
    print(f"  {stream}: {ok} patches on disk "
          f"({fetched} new [{reused} reused], {skipped} already there, {failed} failed)\n")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", nargs="+", default=GRID_STREAMS, choices=list(D.STREAMS))
    args = ap.parse_args()
    ee = D.init_ee()
    if OUT_GRID_CSV.exists():
        points = pd.read_csv(OUT_GRID_CSV, dtype={"grid_id": str})
        print(f"loaded existing grid: {len(points)} points from {OUT_GRID_CSV}")
    else:
        land_geoms = fetch_land_boundaries(ee)
        points = build_grid(land_geoms, GRID_SPACING_KM)
        OUT_GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
        points.to_csv(OUT_GRID_CSV, index=False)
        print(f"saved {OUT_GRID_CSV}")
    for stream in args.stream:
        run_stream(ee, stream, points, False, 0.0)
    print("Done.")
if __name__ == "__main__":
    main()
