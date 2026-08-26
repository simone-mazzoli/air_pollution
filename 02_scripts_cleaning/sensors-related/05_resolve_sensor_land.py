"""
Assign SDS011 low-cost PM sensor locations to German Laender.

The PM cleaning stage keeps sensors inside a broad Germany bounding box. That
rectangle catches all of Germany, but it also includes parts of neighboring
countries. This resolver therefore does not assign every point to the nearest
German state. It uses three explicit categories:

1. direct: the point is covered by a German Land polygon
2. boundary_fallback: the point is just outside a polygon, within a documented
   distance tolerance in projected meters
3. outside_germany: the point is farther away and is excluded from
   sensor_land.csv

Input:
    data/processed/hourly/pm/nodes/sds011_<YYYY-MM>.parquet
    data/processed/germany_states.geojson

Outputs:
    data/processed/sensor_land.csv
    data/processed/sensor_land_assignment_diagnostics.csv
    data/processed/sensor_land_assignment_summary.json
    data/processed/plots/sensor_land_assignment_diagnostic.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROC = BASE_DIR / "data" / "processed"
LAND_BOUNDARIES_PATH = PROC / "germany_states.geojson"
OUT_PATH = PROC / "sensor_land.csv"
DIAGNOSTICS_PATH = PROC / "sensor_land_assignment_diagnostics.csv"
SUMMARY_PATH = PROC / "sensor_land_assignment_summary.json"
PLOT_PATH = PROC / "plots" / "sensor_land_assignment_diagnostic.png"

SENSOR_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3035"  # ETRS89 / LAEA Europe, meters.
DEFAULT_BOUNDARY_FALLBACK_KM = 1.0
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

EXPECTED_LAENDER = {
    "Baden-Wuerttemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thueringen",
}

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}


def transliterate_umlauts(value: object) -> str:
    return (
        str(value)
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )


def detect_name_column(gdf: gpd.GeoDataFrame) -> str:
    best_col, best_score = None, -1
    for col in gdf.columns:
        if col == "geometry":
            continue
        values = gdf[col].dropna().unique().tolist()
        if not (10 <= len(values) <= 20):
            continue
        score = len({transliterate_umlauts(v) for v in values} & EXPECTED_LAENDER)
        if score > best_score:
            best_col, best_score = col, score
    if best_col is None or best_score < 14:
        raise KeyError(
            "Could not confidently detect the Land-name column in the boundary "
            f"file. Best candidate={best_col!r}, score={best_score}/16."
        )
    print(f"auto-detected name column: {best_col!r} ({best_score}/16 matched)")
    return best_col


def discover_months(processed_dir: Path) -> list[str]:
    node_dir = processed_dir / "hourly" / "pm" / "nodes"
    return sorted(
        p.stem.removeprefix("sds011_")
        for p in node_dir.glob("sds011_*.parquet")
        if MONTH_RE.match(p.stem.removeprefix("sds011_"))
    )


def load_sensor_coords(processed_dir: Path, months: list[str]) -> tuple[pd.DataFrame, dict]:
    parts = []
    for month in months:
        path = processed_dir / "hourly" / "pm" / "nodes" / f"sds011_{month}.parquet"
        if not path.exists():
            print(f"WARNING: missing node coordinates for {month}: {path}")
            continue
        part = pd.read_parquet(path, columns=["location", "lat", "lon"])
        part["month"] = month
        parts.append(part)
    if not parts:
        raise FileNotFoundError("No SDS011 node coordinate files found.")

    raw = pd.concat(parts, ignore_index=True)
    for col in ["location", "lat", "lon"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    malformed = raw[raw[["location", "lat", "lon"]].isna().any(axis=1)]
    raw = raw.dropna(subset=["location", "lat", "lon"]).copy()
    raw["location"] = raw["location"].astype("int64")

    unique_coords = raw.drop_duplicates(["location", "lat", "lon"])
    coordinate_pairs = unique_coords.groupby("location").size()
    coords = (
        raw.groupby("location", as_index=False)
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            months_seen=("month", "nunique"),
        )
        .merge(
            coordinate_pairs.rename("coordinate_pairs").reset_index(),
            on="location",
            how="left",
        )
    )
    in_bbox = coords["lat"].between(BBOX_GERMANY["lat_min"], BBOX_GERMANY["lat_max"]) & coords[
        "lon"
    ].between(BBOX_GERMANY["lon_min"], BBOX_GERMANY["lon_max"])

    stats = {
        "raw_node_rows": int(len(raw) + len(malformed)),
        "malformed_coordinate_rows": int(len(malformed)),
        "unique_locations": int(coords["location"].nunique()),
        "locations_with_duplicate_coordinate_pairs": int((coordinate_pairs > 1).sum()),
        "locations_inside_cleaning_bbox": int(in_bbox.sum()),
        "locations_outside_cleaning_bbox": int((~in_bbox).sum()),
        "lat_min": float(coords["lat"].min()),
        "lat_max": float(coords["lat"].max()),
        "lon_min": float(coords["lon"].min()),
        "lon_max": float(coords["lon"].max()),
    }
    return coords, stats


def load_land_polygons(boundaries_path: Path, name_column: str | None) -> gpd.GeoDataFrame:
    if not boundaries_path.exists():
        raise FileNotFoundError(f"{boundaries_path} not found")
    gdf = gpd.read_file(boundaries_path)
    source_crs = str(gdf.crs)
    gdf = gdf.to_crs(SENSOR_CRS)
    name_column = name_column or detect_name_column(gdf)
    if name_column not in gdf.columns:
        raise KeyError(f"{name_column!r} is not a column in {boundaries_path}")

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        print(f"repairing {invalid} invalid boundary geometries with buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf["land"] = gdf[name_column].map(transliterate_umlauts)
    unmapped = sorted(set(gdf["land"]) - EXPECTED_LAENDER)
    if unmapped:
        raise ValueError(f"Unexpected Land names after transliteration: {unmapped}")

    out = gdf[["land", "geometry"]].copy()
    out.attrs["source_crs"] = source_crs
    out.attrs["valid_after_load"] = bool(out.geometry.is_valid.all())
    return out


def assign_land(
    sensors: pd.DataFrame,
    land_polygons: gpd.GeoDataFrame,
    boundary_fallback_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    points = gpd.GeoDataFrame(
        sensors.copy(),
        geometry=gpd.points_from_xy(sensors["lon"], sensors["lat"]),
        crs=SENSOR_CRS,
    )

    direct = gpd.sjoin(points, land_polygons, how="left", predicate="covered_by")
    direct = direct.drop(columns=["index_right"]).drop_duplicates("location")
    direct["assignment_method"] = np.where(direct["land"].notna(), "direct", "unassigned")

    unmatched_locations = direct.loc[direct["land"].isna(), "location"]
    unmatched = points[points["location"].isin(unmatched_locations)].copy()

    diagnostics = direct.drop(columns="geometry").copy()
    diagnostics["nearest_land"] = diagnostics["land"]
    diagnostics["distance_to_state_km"] = 0.0

    if not unmatched.empty:
        nearest = gpd.sjoin_nearest(
            unmatched.to_crs(DISTANCE_CRS),
            land_polygons.to_crs(DISTANCE_CRS),
            how="left",
            distance_col="distance_m",
        )
        nearest = nearest.drop_duplicates("location")
        nearest = nearest[["location", "land", "distance_m"]].rename(
            columns={"land": "nearest_land"}
        )
        nearest["distance_to_state_km"] = nearest["distance_m"] / 1000.0

        diagnostics = diagnostics.drop(columns=["nearest_land", "distance_to_state_km"]).merge(
            nearest[["location", "nearest_land", "distance_to_state_km"]],
            on="location",
            how="left",
        )
        fallback_mask = (
            diagnostics["assignment_method"].eq("unassigned")
            & diagnostics["distance_to_state_km"].le(boundary_fallback_km)
        )
        outside_mask = diagnostics["assignment_method"].eq("unassigned") & ~fallback_mask
        diagnostics.loc[fallback_mask, "land"] = diagnostics.loc[
            fallback_mask, "nearest_land"
        ]
        diagnostics.loc[fallback_mask, "assignment_method"] = "boundary_fallback"
        diagnostics.loc[outside_mask, "assignment_method"] = "outside_germany"

    assigned = diagnostics[diagnostics["assignment_method"].ne("outside_germany")].copy()
    assigned = assigned[["location", "land", "assignment_method", "distance_to_state_km"]]
    assigned["distance_to_state_km"] = assigned["distance_to_state_km"].fillna(0.0)

    if assigned["location"].duplicated().any():
        raise AssertionError("duplicate locations after Land assignment")
    if assigned["land"].isna().any():
        raise AssertionError("assigned rows still contain missing Land values")

    summary = {
        "sensor_crs": SENSOR_CRS,
        "distance_crs": DISTANCE_CRS,
        "spatial_predicate": "covered_by",
        "boundary_fallback_km": float(boundary_fallback_km),
        "total_locations": int(len(sensors)),
        "direct_polygon_assignments": int((diagnostics["assignment_method"] == "direct").sum()),
        "accepted_boundary_fallbacks": int(
            (diagnostics["assignment_method"] == "boundary_fallback").sum()
        ),
        "excluded_outside_germany": int(
            (diagnostics["assignment_method"] == "outside_germany").sum()
        ),
        "output_rows": int(len(assigned)),
    }
    return assigned, diagnostics, summary


def distance_bucket_counts(diagnostics: pd.DataFrame) -> dict[str, int]:
    fallback = diagnostics[diagnostics["assignment_method"].ne("direct")].copy()
    bins = [0, 0.1, 0.5, 1.0, 5.0, 10.0, np.inf]
    labels = ["<=100m", "100-500m", "500m-1km", "1-5km", "5-10km", ">10km"]
    bucket = pd.cut(
        fallback["distance_to_state_km"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    counts = bucket.value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def write_plot(
    diagnostics: pd.DataFrame,
    land_polygons: gpd.GeoDataFrame,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping diagnostic plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        diagnostics,
        geometry=gpd.points_from_xy(diagnostics["lon"], diagnostics["lat"]),
        crs=SENSOR_CRS,
    ).to_crs(DISTANCE_CRS)
    states = land_polygons.to_crs(DISTANCE_CRS)

    fig, ax = plt.subplots(figsize=(8, 9))
    states.boundary.plot(ax=ax, linewidth=0.6, color="black")
    styles = {
        "direct": {"color": "#2f7ed8", "s": 5, "alpha": 0.45, "label": "direct"},
        "boundary_fallback": {
            "color": "#f28e2b",
            "s": 14,
            "alpha": 0.85,
            "label": "boundary fallback",
        },
        "outside_germany": {
            "color": "#d62728",
            "s": 8,
            "alpha": 0.55,
            "label": "excluded outside Germany",
        },
    }
    for method, style in styles.items():
        subset = gdf[gdf["assignment_method"] == method]
        if not subset.empty:
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                marker=".",
                c=style["color"],
                s=style["s"],
                alpha=style["alpha"],
                label=style["label"],
            )
    ax.set_axis_off()
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("SDS011 sensor-to-Land assignment diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(PROC))
    parser.add_argument("--boundaries", default=str(LAND_BOUNDARIES_PATH))
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--months", nargs="+", default=None)
    parser.add_argument(
        "--boundary-fallback-km",
        type=float,
        default=DEFAULT_BOUNDARY_FALLBACK_KM,
        help=(
            "Maximum distance from a German Land polygon for accepting a "
            "near-boundary fallback. Farther points are excluded."
        ),
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    months = args.months or discover_months(processed_dir)
    if not months:
        raise FileNotFoundError("No requested or discoverable SDS011 node months.")

    sensors, coord_stats = load_sensor_coords(processed_dir, months)
    print(f"{len(sensors)} SDS011 sensor locations loaded from {len(months)} month(s)")

    land_polygons = load_land_polygons(Path(args.boundaries), args.name_column)
    print(f"{len(land_polygons)} Land polygons loaded from {args.boundaries}")

    assigned, diagnostics, summary = assign_land(
        sensors, land_polygons, args.boundary_fallback_km
    )
    summary.update(
        {
            "source_months": months,
            "coordinate_stats": coord_stats,
            "boundary_file": str(Path(args.boundaries).resolve()),
            "boundary_source_crs": land_polygons.attrs.get("source_crs"),
            "boundary_valid_after_load": land_polygons.attrs.get("valid_after_load"),
            "fallback_distance_buckets": distance_bucket_counts(diagnostics),
            "nearest_land_counts_for_uncovered": {
                str(k): int(v)
                for k, v in diagnostics.loc[
                    diagnostics["assignment_method"].ne("direct"), "nearest_land"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
            "assigned_land_counts": {
                str(k): int(v) for k, v in assigned["land"].value_counts().sort_index().items()
            },
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(OUT_PATH, index=False)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    write_json(SUMMARY_PATH, summary)
    write_plot(diagnostics, land_polygons, PLOT_PATH)

    print(f"saved assignments -> {OUT_PATH}")
    print(f"saved diagnostics -> {DIAGNOSTICS_PATH}")
    print(f"saved summary -> {SUMMARY_PATH}")
    print(f"saved plot -> {PLOT_PATH}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
