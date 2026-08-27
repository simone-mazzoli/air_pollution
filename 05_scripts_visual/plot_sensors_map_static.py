"""
Plots the EEA stations on a map of Europe.
Input:  data/processed/monthly_avg/humidity/all_sensors/<MONTH>.csv
Output: Air_pollution_report/Figures/generated/sensors_map_static.png
        data/processed/germany_states.geojson (cached background, fetched once)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import report_plot_style

report_plot_style.apply()

DATA_DIR = BASE_DIR / "data" / "processed" / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"
OUT_DIR = BASE_DIR / "Air_pollution_report" / "Figures" / "generated"

GEOJSON_URL = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson"
GEOJSON_CACHE = BASE_DIR / "data" / "processed" / "european_countries.geojson"

BBOX_EUROPE = {"lat_min": 34.0, "lat_max": 71.0, "lon_min": -25.0, "lon_max": 46.0}
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
    pm = pd.read_csv(DATA_DIR)

    geojson = load_state_borders()

    fig, ax = plt.subplots(figsize=(9, 11))
    draw_state_borders(ax, geojson)

    ax.scatter(pm["lon"], pm["lat"], s=8, c="tab:green", alpha=0.5, linewidths=0,
               zorder=2, label=f"EEA PM stations ({len(pm)})")

    ax.set_xlim(BBOX_EUROPE["lon_min"] - MARGIN, BBOX_EUROPE["lon_max"] + MARGIN)
    ax.set_ylim(BBOX_EUROPE["lat_min"] - MARGIN, BBOX_EUROPE["lat_max"] + MARGIN)
    ax.axis('off')
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sensors_map_static.png"
    report_plot_style.savefig(fig, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
