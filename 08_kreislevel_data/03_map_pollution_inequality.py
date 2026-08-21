#!/usr/bin/env python3
"""
Create a map series for the Kreis-level pollution + socioeconomic analysis.

The script reads the already combined Kreis-level table created by
02_build_kreis_exposure.py and produces only choropleth maps.

The first map is mean predicted PM2.5. The remaining maps show the selected
socioeconomic and demographic variables for the same set of covered Kreise.

Inputs
------
08_kreislevel_data/kreis_exposure_socioeconomic.csv

Kreis geometry
--------------
1) data/processed/admin_boundaries/vg250_kreis.geojson, if available
2) otherwise the cached BKG VG250 GeoPackage created by the socioeconomic builder

Outputs
-------
08_kreislevel_data/figures/pollution_inequality_maps/
    01_mean_pm25.png
    02_disposable_income.png
    03_unemployment.png
    04_no_vocational_qualification.png
    05_university_degree.png
    06_immigration_history.png
    07_population_density.png
    08_share_65plus.png

Only Kreise contained in the combined analysis table are colored.
Other German Kreise are shown in light grey for geographic context.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:
    raise SystemExit(
        "This script needs geopandas.\n"
        "Install it once in the project environment with:\n"
        "  pip install geopandas"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT
    / "08_kreislevel_data"
    / "kreis_exposure_socioeconomic.csv"
)

PROJECT_KREIS_GEOJSON = (
    ROOT
    / "data"
    / "processed"
    / "admin_boundaries"
    / "vg250_kreis.geojson"
)

BKG_CACHE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "socioeconomic_auto"
    / "bkg"
)

DEFAULT_OUTDIR = (
    ROOT
    / "08_kreislevel_data"
    / "figures"
    / "pollution_inequality_maps"
)


MAPS = [
    {
        "column": "mean_pred_pm25",
        "filename": "01_mean_pm25.png",
        "title": "Mean predicted PM2.5 by Kreis",
        "legend": "Mean predicted PM2.5 [µg/m³]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "disposable_income_2023_eur_per_capita",
        "filename": "02_disposable_income.png",
        "title": "Disposable income per capita by Kreis (2023)",
        "legend": "Disposable income [EUR per capita]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "unemployment_rate_2024_pct",
        "filename": "03_unemployment.png",
        "title": "Unemployment rate by Kreis (2024)",
        "legend": "Unemployment rate [%]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "no_vocational_qualification_2022_pct",
        "filename": "04_no_vocational_qualification.png",
        "title": "Population without vocational qualification by Kreis (2022)",
        "legend": "No vocational qualification [%]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "university_degree_2022_pct",
        "filename": "05_university_degree.png",
        "title": "Population with university degree by Kreis (2022)",
        "legend": "University degree [%]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "immigration_history_2022_pct",
        "filename": "06_immigration_history.png",
        "title": "Population with immigration history by Kreis (2022)",
        "legend": "Immigration history [%]",
        "cmap": "viridis",
        "log_scale": False,
    },
    {
        "column": "population_density_2024_per_km2",
        "filename": "07_population_density.png",
        "title": "Population density by Kreis (2024)",
        "legend": "Population density [persons per km²] — log color scale",
        "cmap": "viridis",
        "log_scale": True,
    },
    {
        "column": "share_65plus_2024_pct",
        "filename": "08_share_65plus.png",
        "title": "Population aged 65+ by Kreis (2024)",
        "legend": "Population aged 65+ [%]",
        "cmap": "viridis",
        "log_scale": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Kreis-level pollution and socioeconomic choropleth maps."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
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

        candidates = [
            table
            for table in tables
            if "krs" in str(table).lower()
            or "kreis" in str(table).lower()
        ]
        if candidates:
            return gpkg, candidates[0]

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

    if "GEN" in kreise.columns and "Name" not in kreise.columns:
        kreise = kreise.rename(columns={"GEN": "Name"})

    required = {"AGS", "geometry"}
    missing = required - set(kreise.columns)
    if missing:
        raise ValueError(
            f"Kreis geometry is missing required columns: {sorted(missing)}"
        )

    if "Name" not in kreise.columns:
        kreise["Name"] = kreise["AGS"].astype(str)

    kreise = kreise[["AGS", "Name", "geometry"]].copy()
    kreise["AGS"] = normalize_ags(kreise["AGS"])
    kreise = kreise.dropna(subset=["AGS", "geometry"]).drop_duplicates("AGS")

    if kreise.crs is None:
        raise ValueError("Kreis geometry has no CRS.")

    # Work in the geometry's native projected CRS for plotting.
    print(f"Loaded {len(kreise)} Kreis polygons")
    return kreise


def load_analysis_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Combined Kreis analysis table not found: {path}\n"
            "Run 02_build_kreis_exposure.py first."
        )

    df = pd.read_csv(path, dtype={"AGS": str})
    df["AGS"] = normalize_ags(df["AGS"])

    required = {"AGS", *[spec["column"] for spec in MAPS]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Combined analysis table is missing required columns: {sorted(missing)}"
        )

    for spec in MAPS:
        col = spec["column"]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df)} covered Kreise from {path}")
    return df


def analysis_extent(
    merged: gpd.GeoDataFrame,
    covered_ags: set[str],
) -> tuple[float, float, float, float]:
    covered = merged[merged["AGS"].isin(covered_ags)]
    if covered.empty:
        raise ValueError("No covered Kreis geometries found after the AGS join.")

    xmin, ymin, xmax, ymax = covered.total_bounds
    xpad = max((xmax - xmin) * 0.06, 1.0)
    ypad = max((ymax - ymin) * 0.06, 1.0)

    return (
        xmin - xpad,
        xmax + xpad,
        ymin - ypad,
        ymax + ypad,
    )


def make_norm(values: pd.Series, log_scale: bool):
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean[np.isfinite(clean)]

    if log_scale:
        clean = clean[clean > 0]
        if clean.empty:
            raise ValueError("No positive values available for logarithmic map scale.")
        return LogNorm(
            vmin=float(clean.min()),
            vmax=float(clean.max()),
        )

    if clean.empty:
        raise ValueError("No finite values available for map scale.")

    vmin = float(clean.min())
    vmax = float(clean.max())

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    return Normalize(vmin=vmin, vmax=vmax)


def save_map(
    merged: gpd.GeoDataFrame,
    covered_ags: set[str],
    spec: dict,
    extent: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    column = spec["column"]

    # Restrict colored values to the Kreise present in the pollution-analysis
    # table, so every map describes exactly the same geographic sample.
    covered = merged[merged["AGS"].isin(covered_ags)].copy()
    available = covered[covered[column].notna()].copy()

    if available.empty:
        raise ValueError(f"No mapped values available for {column}")

    if spec["log_scale"]:
        available = available[available[column] > 0].copy()
        if available.empty:
            raise ValueError(f"No positive mapped values available for {column}")

    norm = make_norm(available[column], spec["log_scale"])

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # All Germany in light grey gives geographic context.
    merged.plot(
        ax=ax,
        facecolor="0.92",
        edgecolor="white",
        linewidth=0.35,
        zorder=1,
    )

    # Kreise included in the analysis but missing this particular variable are
    # slightly darker grey than the rest of Germany.
    missing = covered[covered[column].isna()]
    if not missing.empty:
        missing.plot(
            ax=ax,
            facecolor="0.78",
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )

    available.plot(
        ax=ax,
        column=column,
        cmap=spec["cmap"],
        norm=norm,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )

    sm = ScalarMappable(norm=norm, cmap=spec["cmap"])
    sm.set_array([])
    colorbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        shrink=0.78,
    )
    colorbar.set_label(spec["legend"])

    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(spec["title"], pad=12)
    ax.set_axis_off()

    n_available = int(available["AGS"].nunique())
    n_covered = len(covered_ags)

    ax.text(
        0.01,
        0.01,
        f"Colored Kreise: {n_available}/{n_covered} in modeled grid coverage",
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.35",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_path}")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    analysis = load_analysis_table(args.input)
    kreise = load_kreis_geometry()

    # Merge the analysis variables onto all 400 Kreis polygons.
    values = analysis.drop(columns=["Name", "NUTS"], errors="ignore")
    merged = kreise.merge(
        values,
        on="AGS",
        how="left",
        validate="one_to_one",
    )

    covered_ags = set(analysis["AGS"].dropna().unique())

    matched = int(merged["AGS"].isin(covered_ags).sum())
    if matched != len(covered_ags):
        print(
            f"WARNING: matched {matched}/{len(covered_ags)} covered AGS values "
            "to Kreis geometry."
        )

    extent = analysis_extent(merged, covered_ags)

    print(f"Creating {len(MAPS)} maps in {args.outdir}")

    for spec in MAPS:
        save_map(
            merged=merged,
            covered_ags=covered_ags,
            spec=spec,
            extent=extent,
            out_path=args.outdir / spec["filename"],
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
