"""
sensor coverage check for each fold to decide which state should be our test

Input:  data/processed/germany_states.geojson
        data/processed/sensor_land.csv       (from 02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py)
Output: printed comparison table, no file saved
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PROC = BASE_DIR / "data" / "processed"
GEOJSON_PATH = PROC / "germany_states.geojson"
SENSOR_LAND_PATH = PROC / "sensor_land.csv"
CORR_FOLD_DIR = PROC / "corrected" / "fold"

METRIC_CRS = "EPSG:3035"  # ETRS89-LAEA Europe -- equal-area, correct for this
RADII_KM = [10, 15]

# These are the fold groups the calibration was actually fit against.
LAND_TO_FOLD = {
    "Berlin": "Berlin-Brandenburg", "Brandenburg": "Berlin-Brandenburg",
    "Bremen": "Bremen-Niedersachsen", "Niedersachsen": "Bremen-Niedersachsen",
    "Hamburg": "Hamburg-Schleswig-Holstein", "Schleswig-Holstein": "Hamburg-Schleswig-Holstein",
    "Saarland": "Saarland-Rheinland-Pfalz", "Rheinland-Pfalz": "Saarland-Rheinland-Pfalz",
    "Baden-Wuerttemberg": "Baden-Wuerttemberg", "Bayern": "Bayern", "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen", "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt", "Thueringen": "Thueringen",
}

# fold-group name -> which raw Land name(s) from the geojson make it up.
# Derived from LAND_TO_FOLD (inverted) so all 12 folds are covered automatically.
CANDIDATES = {}
for land, fold in LAND_TO_FOLD.items():
    CANDIDATES.setdefault(fold, []).append(land)


def _canon(s):
    # normalize Land names so ASCII keys (Baden-Wuerttemberg, Thueringen) match the
    # geojson's native spellings (Baden-Wuerttemberg, Thueringen) -- umlauts, ss, and
    # separators are folded to a common form on both sides before comparing
    return (str(s).lower()
            .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            .replace("-", " ").replace("_", " ").strip())


def load_land_polygons():
    gdf = gpd.read_file(GEOJSON_PATH)
    if gdf.crs is None:
        raise ValueError(f"{GEOJSON_PATH.name} has no CRS set -- fix this first")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    name_candidates = [c for c in gdf.columns if c.lower() in
                      ("name", "name_1", "gen", "land", "bundesland", "nuts_name")]
    if not name_candidates:
        raise ValueError(f"Could not find a Land-name column in {list(gdf.columns)}")
    name_col = name_candidates[0]
    print(f"Using '{name_col}' as the Land name column")
    print(f"Available Land names: {sorted(gdf[name_col].unique())}\n")

    return gdf, name_col


def load_sensors_with_fold():
    if not SENSOR_LAND_PATH.exists():
        raise FileNotFoundError(
            f"missing {SENSOR_LAND_PATH} -- run "
            "02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py first"
        )
    sensors = pd.read_csv(SENSOR_LAND_PATH)
    if "land" not in sensors.columns:
        raise KeyError(
            f"{SENSOR_LAND_PATH} has no 'land' column (found: {list(sensors.columns)}) "
            "-- check 02_scripts_cleaning/sensors-related/05_resolve_sensor_land.py output"
        )

    sensors["fold"] = sensors["land"].map(LAND_TO_FOLD)
    unmapped = sensors.loc[sensors["fold"].isna(), "land"].unique()
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} Land value(s) in {SENSOR_LAND_PATH.name} "
              f"don't match any key in LAND_TO_FOLD: {list(unmapped)} -- those "
              f"sensors will be excluded from every candidate's count below")

    if "lat" not in sensors.columns or "lon" not in sensors.columns:
        # sensor_land.csv may only have location/land, so pull coordinates from
        # the same source it reads from. Lat/lon are
        # identical across every fold's annual file for a given location, so any
        # one fold's file has everything needed here.
        annual_files = sorted(CORR_FOLD_DIR.glob("*/annual/*.csv"))
        if not annual_files:
            raise FileNotFoundError(
                f"{SENSOR_LAND_PATH} has no lat/lon and no fold annual files "
                f"found under {CORR_FOLD_DIR}/*/annual/*.csv to pull them from"
            )
        coords = pd.read_csv(annual_files[0])[["location", "lat", "lon"]]
        before = len(sensors)
        sensors = sensors.merge(coords, on="location", how="left")
        missing_coords = sensors["lat"].isna().sum()
        if missing_coords:
            print(f"WARNING: {missing_coords}/{before} sensor(s) had no matching "
                  f"lat/lon in {annual_files[0].name} -- excluded from coverage math")
        sensors = sensors.dropna(subset=["lat", "lon"])

    return sensors


def coverage_for_candidate(fold_name, land_names, land_gdf, name_col, sensors):
    # dissolve the (possibly multiple) states into one polygon, reproject to
    # the equal-area CRS for correct buffering/area math
    wanted = {_canon(n) for n in land_names}
    polys = land_gdf[land_gdf[name_col].map(_canon).isin(wanted)]
    if polys.empty:
        print(f"  WARNING: no polygon(s) matched {land_names} in the geojson -- "
              f"check spelling against the available names printed above")
        return None
    land_poly = polys.to_crs(METRIC_CRS).union_all()
    land_area_km2 = land_poly.area / 1e6

    fold_sensors = sensors[sensors["fold"] == fold_name]
    n_sensors = len(fold_sensors)
    if n_sensors == 0:
        print(f"  WARNING: 0 sensors found with fold == '{fold_name}' in "
              f"sensor_land.csv -- check the fold name matches exactly")
        return None

    points = gpd.GeoDataFrame(
        fold_sensors,
        geometry=gpd.points_from_xy(fold_sensors["lon"], fold_sensors["lat"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    row = {
        "fold": fold_name,
        "land_area_km2": land_area_km2,
        "n_sensors": n_sensors,
        "density_per_1000km2": n_sensors / (land_area_km2 / 1000),
    }

    for r_km in RADII_KM:
        buffered = points.geometry.buffer(r_km * 1000)
        covered = buffered.union_all().intersection(land_poly)
        covered_km2 = covered.area / 1e6
        coverage_pct = 100.0 * covered_km2 / land_area_km2
        row[f"coverage_pct_{r_km}km"] = coverage_pct
        row[f"empty_pct_{r_km}km"] = 100.0 - coverage_pct

    return row


def main():
    land_gdf, name_col = load_land_polygons()
    sensors = load_sensors_with_fold()

    print(f"Checking all {len(CANDIDATES)} fold groups: {sorted(CANDIDATES)}\n")

    rows = []
    for fold_name, land_names in CANDIDATES.items():
        print(f"=== {fold_name} ({'+'.join(land_names)}) ===")
        row = coverage_for_candidate(fold_name, land_names, land_gdf, name_col, sensors)
        if row:
            rows.append(row)
            print(f"  area: {row['land_area_km2']:,.0f} km2  "
                  f"sensors: {row['n_sensors']}  "
                  f"density: {row['density_per_1000km2']:.2f} / 1000km2")
            for r_km in RADII_KM:
                print(f"  @{r_km}km buffer: "
                      f"{row[f'coverage_pct_{r_km}km']:.1f}% covered, "
                      f"{row[f'empty_pct_{r_km}km']:.1f}% empty")
        print()

    if not rows:
        print("Nothing computed -- check the warnings above.")
        return

    df = pd.DataFrame(rows).set_index("fold").sort_values("empty_pct_10km", ascending=False)
    print("=" * 70)
    print("Summary (sorted by empty_pct_10km, sparsest first):")
    print(df.round(1).to_string())

    print("\nGuidance: the assignment wants 'sparse but still enough sensors "
          "to evaluate on'. There's no universal magic ratio -- pick the "
          "candidate with the HIGHEST empty_pct (genuinely under-covered, a "
          "real spatial-extrapolation test), as long as n_sensors stays above "
          "a sane floor for a stable evaluation metric (aim for at least "
          "several dozen to ~100 sensors so the held-out RMSE/R2 isn't just "
          "noise from a handful of points). Also worth checking whether a "
          "candidate's sensors cluster around one feature (a coastline, a "
          "single city) rather than spreading across the empty area -- a high "
          "empty_pct can be misleading if the sensors you DO have all sit in "
          "one corner of the Land.")


if __name__ == "__main__":
    main()
