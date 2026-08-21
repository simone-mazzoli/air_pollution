"""
Aggregate dense grid predictions to German Kreise and join them to the
Kreis-level socioeconomic table.

This is the bridge between the model output and the environmental-justice
analysis.

Inputs
------
grid_results/cnn_deep_wide_grid_predictions.csv
08_kreislevel_data/socioeconomic_kreis_2024.csv

Kreis geometry is loaded from:
1) data/processed/admin_boundaries/vg250_kreis.geojson, if available
2) the cached BKG VG250 GeoPackage created by the socioeconomic builder

Outputs
-------
08_kreislevel_data/kreis_pollution_exposure.csv
08_kreislevel_data/kreis_exposure_socioeconomic.csv
08_kreislevel_data/figures/pollution_inequality/01_mean_pm25_by_kreis.png

Method
------
Each dense-grid point is assigned to the Kreis polygon containing its centre.
The main exposure measure is the arithmetic mean of predicted PM2.5 across
grid points in that Kreis. With a regular grid, this is an approximate
area-average of modeled ambient PM2.5.

This is NOT population-weighted exposure.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "This script needs geopandas for the spatial join.\n"
        "Install it once in the project venv with:\n"
        "  pip install geopandas"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GRID = ROOT / "grid_results" / "cnn_deep_wide_grid_predictions.csv"
DEFAULT_SOCIO = ROOT / "08_kreislevel_data" / "socioeconomic_kreis_2024.csv"

PROJECT_KREIS_GEOJSON = (
    ROOT / "data" / "processed" / "admin_boundaries" / "vg250_kreis.geojson"
)
BKG_CACHE_DIR = ROOT / "data" / "raw" / "socioeconomic_auto" / "bkg"

OUT_DIR = ROOT / "08_kreislevel_data"
DEFAULT_EXPOSURE_OUT = OUT_DIR / "kreis_pollution_exposure.csv"
DEFAULT_COMBINED_OUT = OUT_DIR / "kreis_exposure_socioeconomic.csv"
DEFAULT_FIGURE = (
    OUT_DIR
    / "figures"
    / "pollution_inequality"
    / "01_mean_pm25_by_kreis.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate dense pollution predictions to Kreis level."
    )
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--socio", type=Path, default=DEFAULT_SOCIO)
    parser.add_argument(
        "--pollutant",
        default="pm25",
        help="Pollutant suffix used in the prediction column, e.g. pm25",
    )
    parser.add_argument("--exposure-out", type=Path, default=DEFAULT_EXPOSURE_OUT)
    parser.add_argument("--combined-out", type=Path, default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def normalize_ags(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
        .str[:5]
    )


def find_bkg_gpkg() -> tuple[Path, str]:
    gpkg_files = sorted(BKG_CACHE_DIR.rglob("*.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError(
            "No Kreis GeoJSON and no cached BKG GeoPackage found. "
            "Run build_socioeconomic_kreis_2024.py first."
        )

    for gpkg in gpkg_files:
        con = sqlite3.connect(str(gpkg))
        try:
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
        finally:
            con.close()

        preferred = [
            table
            for table in tables
            if "krs" in str(table).lower() or "kreis" in str(table).lower()
        ]
        if preferred:
            return gpkg, preferred[0]

    raise RuntimeError(
        "Found a cached BKG GeoPackage but could not identify its Kreis layer."
    )


def load_kreis_geometry() -> gpd.GeoDataFrame:
    if PROJECT_KREIS_GEOJSON.exists():
        print(f"Using Kreis geometry: {PROJECT_KREIS_GEOJSON}")
        kreise = gpd.read_file(PROJECT_KREIS_GEOJSON)
    else:
        gpkg, layer = find_bkg_gpkg()
        print(f"Using Kreis geometry: {gpkg} | layer={layer}")
        kreise = gpd.read_file(gpkg, layer=layer)

    rename = {}
    if "GEN" in kreise.columns and "Name" not in kreise.columns:
        rename["GEN"] = "Name"
    if rename:
        kreise = kreise.rename(columns=rename)

    required = {"AGS", "geometry"}
    missing = required - set(kreise.columns)
    if missing:
        raise ValueError(
            f"Kreis geometry is missing required columns: {sorted(missing)}"
        )

    if "Name" not in kreise.columns:
        kreise["Name"] = kreise["AGS"].astype(str)

    keep = ["AGS", "Name", "geometry"]
    if "NUTS" in kreise.columns:
        keep.insert(2, "NUTS")

    kreise = kreise[keep].copy()
    kreise["AGS"] = normalize_ags(kreise["AGS"])
    kreise = kreise.dropna(subset=["AGS", "geometry"]).drop_duplicates("AGS")

    if kreise.crs is None:
        raise ValueError("Kreis geometry has no CRS; cannot safely spatially join.")
    kreise = kreise.to_crs(4326)

    print(f"Loaded {len(kreise)} Kreis polygons")
    return kreise


def load_grid(path: Path, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Grid prediction file not found: {path}")

    grid = pd.read_csv(path, dtype={"grid_id": str})
    required = {"grid_id", "lat", "lon", pred_col}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    for col in ["lat", "lon", pred_col]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")

    grid = grid.dropna(subset=["lat", "lon", pred_col]).copy()

    # If a land flag is present, remove explicit non-land points.
    if "land" in grid.columns:
        land_text = grid["land"].astype(str).str.lower()
        explicit_false = land_text.isin({"false", "0", "no"})
        if explicit_false.any():
            print(f"Dropping {int(explicit_false.sum())} grid points marked non-land")
            grid = grid.loc[~explicit_false].copy()

    print(f"Loaded {len(grid)} usable grid predictions")
    return grid


def assign_grid_to_kreise(
    grid: pd.DataFrame,
    kreise: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        grid.copy(),
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        kreise[["AGS", "Name", "geometry"]],
        how="left",
        predicate="within",
    )

    # A point exactly on a polygon boundary can be unmatched by `within`.
    # Retry only those points with `intersects`.
    unmatched = joined["AGS"].isna()
    if unmatched.any():
        retry = gpd.sjoin(
            points.loc[unmatched, points.columns],
            kreise[["AGS", "Name", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Shared-boundary points can intersect two Kreise. Keep one assignment
        # deterministically and report that this occurred.
        duplicated = retry.index.duplicated(keep=False)
        if duplicated.any():
            n_dup = int(pd.Index(retry.index[duplicated]).nunique())
            print(
                f"WARNING: {n_dup} boundary grid points intersect multiple Kreise; "
                "keeping the first deterministic match."
            )
            retry = retry[~retry.index.duplicated(keep="first")]

        joined.loc[retry.index, "AGS"] = retry["AGS"]
        joined.loc[retry.index, "Name"] = retry["Name"]

    unmatched_n = int(joined["AGS"].isna().sum())
    if unmatched_n:
        print(
            f"WARNING: {unmatched_n}/{len(joined)} grid points could not be "
            "assigned to a Kreis and will be excluded."
        )

    assigned = joined.dropna(subset=["AGS"]).copy()
    assigned["AGS"] = normalize_ags(assigned["AGS"])

    print(
        f"Assigned {len(assigned)}/{len(joined)} grid points to "
        f"{assigned['AGS'].nunique()} Kreise"
    )
    return assigned


def aggregate_exposure(
    assigned: gpd.GeoDataFrame,
    pred_col: str,
) -> pd.DataFrame:
    grouped = assigned.groupby(["AGS", "Name"], as_index=False)[pred_col]

    exposure = grouped.agg(
        n_grid_points="count",
        mean_pred="mean",
        median_pred="median",
        std_pred=lambda s: s.std(ddof=0),
        min_pred="min",
        max_pred="max",
    )

    rename = {
        "mean_pred": f"mean_{pred_col}",
        "median_pred": f"median_{pred_col}",
        "std_pred": f"std_{pred_col}",
        "min_pred": f"min_{pred_col}",
        "max_pred": f"max_{pred_col}",
    }
    exposure = exposure.rename(columns=rename)

    exposure = exposure.sort_values("AGS").reset_index(drop=True)

    print("\nGrid-point coverage per Kreis:")
    print(exposure["n_grid_points"].describe().to_string())
    return exposure


def join_socioeconomic(
    exposure: pd.DataFrame,
    socio_path: Path,
) -> pd.DataFrame:
    if not socio_path.exists():
        raise FileNotFoundError(f"Socioeconomic table not found: {socio_path}")

    socio = pd.read_csv(socio_path, dtype={"AGS": str})
    socio["AGS"] = normalize_ags(socio["AGS"])

    # Avoid duplicate Name columns; keep the socioeconomic table's official name.
    exp = exposure.drop(columns=["Name"], errors="ignore")

    combined = socio.merge(
        exp,
        on="AGS",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"\nCombined analysis table: {len(combined)} Kreise "
        f"(out of {len(socio)} socioeconomic Kreise)"
    )
    return combined


def save_kreis_map(
    kreise: gpd.GeoDataFrame,
    exposure: pd.DataFrame,
    value_col: str,
    out_path: Path,
) -> None:
    mapped = kreise.merge(
        exposure[["AGS", value_col]],
        on="AGS",
        how="left",
        validate="one_to_one",
    )

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw all German Kreise faintly for geographic context.
    mapped.plot(
        ax=ax,
        facecolor="0.94",
        edgecolor="white",
        linewidth=0.35,
    )

    modeled = mapped[mapped[value_col].notna()]
    if modeled.empty:
        raise ValueError("No Kreis exposure values available for mapping.")

    modeled.plot(
        ax=ax,
        column=value_col,
        cmap="viridis",
        legend=True,
        edgecolor="white",
        linewidth=0.45,
        legend_kwds={
            "label": "Mean predicted PM2.5 [µg/m³]",
            "shrink": 0.75,
        },
    )

    ax.set_title("Mean modeled PM2.5 by Kreis")
    ax.set_axis_off()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pred_col = f"pred_{args.pollutant}"

    grid = load_grid(args.grid, pred_col)
    kreise = load_kreis_geometry()
    assigned = assign_grid_to_kreise(grid, kreise)

    exposure = aggregate_exposure(assigned, pred_col)
    args.exposure_out.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_csv(args.exposure_out, index=False)

    combined = join_socioeconomic(exposure, args.socio)
    args.combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.combined_out, index=False)

    mean_col = f"mean_{pred_col}"
    save_kreis_map(kreise, exposure, mean_col, args.figure)

    print("\nSaved:")
    print(f"  {args.exposure_out}")
    print(f"  {args.combined_out}")
    print(f"  {args.figure}")


if __name__ == "__main__":
    main()