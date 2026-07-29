"""
Assigns each low-cost PM sensor location to its German state via a
point-in-polygon spatial join against data/processed/germany_states.geojson.

Sensor coordinates are read from a single fold's annual CSV. .

this scans every non-geometry column and picks whichever one has values
that match the most of the 16 known Land names. If no column matches,
it prints every column and its unique values and stops. TO SIMPLIFY

Land name spellings in the output match LAND_TO_FOLD in
calibrate_pm_leave_one_fold_out.py, so this file's "land" column can be mapped
through LAND_TO_FOLD directly without a second renaming step.

a point sitting exactly on a shared border between
two Land polygons can match BOTH under predicate="within" (floating-point
precision at the shared edge), so sjoin can return more than one row for the
same location.

Input:  data/processed/corrected/fold/<any fold>/annual/<YEAR>.csv  (location, lat, lon)
        data/processed/germany_states.geojson  (Land polygons)
Output: data/processed/sensor_land.csv   (location, land)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
FOLD_ANNUAL_DIR = PROC / "corrected" / "fold"
LAND_BOUNDARIES_PATH = PROC / "germany_states.geojson"
OUT_PATH = PROC / "sensor_land.csv"

# ASCII spellings matching LAND_TO_FOLD in calibrate_pm_leave_one_fold_out.py.
EXPECTED_LAENDER = {
    "Baden-Wuerttemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
    "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen",
    "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt",
    "Schleswig-Holstein", "Thueringen",
}


def transliterate_umlauts(s):
    """Same ASCII scheme LAND_TO_FOLD's keys already use -- undoes German
    umlauts/ß so "Baden-Württemberg" and "Baden-Wuerttemberg" become the same
    string, regardless of which spelling germany_states.geojson happens to use."""
    return (str(s)
            .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            .replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
            .replace("ß", "ss"))


def detect_name_column(gdf):
    """Scans every non-geometry column, transliterates its values, and scores
    how many match EXPECTED_LAENDER -- returns (column, score, transliterated
    values) for the best-scoring column, without assuming any particular
    column name up front."""
    best_col, best_score, best_values = None, -1, None
    for col in gdf.columns:
        if col == "geometry":
            continue
        values = gdf[col].dropna().unique().tolist()
        if not (10 <= len(values) <= 20):
            continue
        transliterated = {transliterate_umlauts(v) for v in values}
        score = len(transliterated & EXPECTED_LAENDER)
        if score > best_score:
            best_col, best_score, best_values = col, score, transliterated
    return best_col, best_score, best_values


def load_sensor_coords():
    annual_files = sorted(FOLD_ANNUAL_DIR.glob("*/annual/*.csv"))
    if not annual_files:
        raise FileNotFoundError(
            f"no fold annual files found under {FOLD_ANNUAL_DIR}/*/annual/*.csv -- "
            f"run the calibration script first")
    df = pd.read_csv(annual_files[0])
    missing_cols = {"location", "lat", "lon"} - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"{annual_files[0]} is missing expected column(s) {missing_cols}")
    return df[["location", "lat", "lon"]].drop_duplicates(subset="location")


def load_land_polygons(boundaries_path, name_column=None):
    if not boundaries_path.exists():
        raise FileNotFoundError(f"{boundaries_path} not found")
    gdf = gpd.read_file(boundaries_path)

    if name_column is None:
        name_column, score, transliterated = detect_name_column(gdf)
        if name_column is None or score < 14:
            print("Could not confidently auto-detect the Land-name column.")
            print(f"Columns in {boundaries_path}: {list(gdf.columns)}")
            for col in gdf.columns:
                if col == "geometry":
                    continue
                print(f"  {col}: {gdf[col].dropna().unique().tolist()[:20]}")
            raise KeyError(
                f"best guess was column {name_column!r} matching only "
                f"{score}/16 known Laender -- rerun with "
                f"--name-column <the right one> using the columns printed above")
        print(f"auto-detected name column: {name_column!r} "
              f"({score}/16 known Laender matched)")

    if name_column not in gdf.columns:
        raise KeyError(
            f"'{name_column}' not in {boundaries_path}'s columns "
            f"({list(gdf.columns)})")

    gdf = gdf.to_crs(epsg=4326)  # match sensor lat/lon, which are WGS84
    gdf["land"] = gdf[name_column].map(transliterate_umlauts)
    unmapped = sorted(set(gdf["land"]) - EXPECTED_LAENDER)
    if unmapped:
        print(f"WARNING: {len(unmapped)} value(s) in column {name_column!r} "
              f"don't match any of the 16 known Laender after transliteration: "
              f"{unmapped} -- check this is really the Land-name column")

    return gdf[["land", "geometry"]]


def assign_land(sensors, land_polygons):
    points = gpd.GeoDataFrame(
        sensors,
        geometry=gpd.points_from_xy(sensors["lon"], sensors["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, land_polygons, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"])

    # a point exactly on a shared border can match >1 polygon -- collapse back
    # to one row per location before anything else touches this, so downstream
    # code never has to reason about row-count/position mismatches against
    # `points` again
    joined = joined.drop_duplicates(subset="location", keep="first")

    unmatched_mask = joined["land"].isna()
    n_unmatched = int(unmatched_mask.sum())
    if n_unmatched:
        print(f"{n_unmatched} sensor(s) didn't fall strictly within any Land "
              f"polygon (border/coastline/foreign points) -- falling back to nearest "
              f"polygon for those")
        unmatched_locations = joined.loc[unmatched_mask, "location"]
        # select by location VALUE, not by row position/index -- points and
        # joined are never assumed to share row alignment anywhere in this file
        unmatched_points = points[points["location"].isin(unmatched_locations)][
            ["location", "geometry"]
        ]
        nearest = gpd.sjoin_nearest(unmatched_points, land_polygons, how="left")
        nearest = nearest.drop_duplicates(subset="location")
        land_by_location = nearest.set_index("location")["land"]
        joined.loc[unmatched_mask, "land"] = (
            joined.loc[unmatched_mask, "location"].map(land_by_location)
        )

    still_unmatched = joined["land"].isna().sum()
    if still_unmatched:
        print(f"WARNING: {still_unmatched} sensor(s) still unmatched after "
              f"nearest-polygon fallback -- check their lat/lon are within "
              f"Germany's bounding box")

    # final safety check: every input location should appear exactly once
    result = joined[["location", "land"]].reset_index(drop=True)
    if len(result) != len(sensors) or result["location"].duplicated().any():
        raise AssertionError(
            f"row-count/uniqueness mismatch after join: {len(result)} output "
            f"rows vs {len(sensors)} input sensors -- do not trust this output, "
            f"something upstream still isn't 1-row-per-location"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", default=None,
                       help="override the germany_states.geojson path")
    parser.add_argument("--name-column", default=None,
                       help="skip auto-detection, use this column directly")
    args = parser.parse_args()

    boundaries_path = Path(args.boundaries) if args.boundaries else LAND_BOUNDARIES_PATH

    sensors = load_sensor_coords()
    print(f"{len(sensors)} sensor locations to assign")

    land_polygons = load_land_polygons(boundaries_path, args.name_column)
    print(f"{len(land_polygons)} Land polygons loaded from {boundaries_path}")

    result = assign_land(sensors, land_polygons)
    result.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
    print(result["land"].value_counts().to_string())


if __name__ == "__main__":
    main()
