"""
Plots low-qual PM sensors, UBA reference stations, and humidity sensors over a Germany state-border background for a given month.
Input:  data/processed/monthly_avg/all_pm_sensors/germany_monthly_avg_ALLPM_<MONTH>.csv
        data/processed/monthly_avg/uba/pm_reference_stations_<MONTH>.csv
        data/processed/monthly_avg/humidity/all_sensors/<MONTH>.csv
Output: data/processed/plots/sensors_map_static_<MONTH>.png
        data/processed/germany_states.geojson (cached background, fetched once)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

MONTH = "2024-01"  # swap for a year tag once yearly-avg files exist, rest of the script stays the same
BASE_DIR = Path(__file__).resolve().parent.parent
MONTHLY_DIR = BASE_DIR / "data" / "processed" / "monthly_avg"
OUT_DIR = BASE_DIR / "data" / "processed" / "plots"

PM_PATH = MONTHLY_DIR / "all_pm_sensors" / f"germany_monthly_avg_ALLPM_{MONTH}.csv"
UBA_PATH = MONTHLY_DIR / "uba" / f"pm_reference_stations_{MONTH}.csv"
HUMIDITY_PATH = MONTHLY_DIR / "humidity" / "all_sensors" / f"{MONTH}.csv"

GEOJSON_URL = "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json"
GEOJSON_CACHE = BASE_DIR / "data" / "processed" / "germany_states.geojson"

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}
MARGIN = 0.3


def load_state_borders() -> dict:
    if not GEOJSON_CACHE.exists():
        GEOJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(GEOJSON_URL, timeout=30)
        resp.raise_for_status()
        GEOJSON_CACHE.write_text(resp.text)
    return json.loads(GEOJSON_CACHE.read_text())


def draw_state_borders(ax, geojson: dict) -> None:
    patches = []
    for feature in geojson["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            outer_ring = polygon[0]
            patches.append(Polygon(outer_ring, closed=True))

    collection = PatchCollection(patches, facecolor="#f0f0f0", edgecolor="#999999", linewidths=0.6, zorder=0)
    ax.add_collection(collection)


def main() -> None:
    pm = pd.read_csv(PM_PATH)
    uba = pd.read_csv(UBA_PATH)
    humidity = pd.read_csv(HUMIDITY_PATH)

    geojson = load_state_borders()

    fig, ax = plt.subplots(figsize=(9, 11))
    draw_state_borders(ax, geojson)

    ax.scatter(humidity["lon"], humidity["lat"], s=12, c="tab:blue", alpha=0.5, linewidths=0,
               zorder=1, label=f"humidity sensors ({len(humidity)})")
    ax.scatter(pm["lon"], pm["lat"], s=8, c="tab:green", alpha=0.5, linewidths=0,
               zorder=2, label=f"low-qual PM sensors ({len(pm)})")
    ax.scatter(uba["lon"], uba["lat"], s=45, c="saddlebrown", marker="^", edgecolors="black",
               linewidths=0.6, alpha=0.9, zorder=3, label=f"UBA reference stations ({len(uba)})")

    ax.set_xlim(BBOX_GERMANY["lon_min"] - MARGIN, BBOX_GERMANY["lon_max"] + MARGIN)
    ax.set_ylim(BBOX_GERMANY["lat_min"] - MARGIN, BBOX_GERMANY["lat_max"] + MARGIN)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Sensor locations -- {MONTH}")
    ax.legend(loc="lower left", framealpha=0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sensors_map_static_{MONTH}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
