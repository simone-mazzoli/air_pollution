"""
Plots low-qual PM sensors, UBA reference stations, and humidity sensors on an interactive Leaflet map for a given month.
Input:  data/processed/monthly_avg/all_pm_sensors/germany_monthly_avg_ALLPM_<MONTH>.csv
        data/processed/monthly_avg/uba/pm_reference_stations_<MONTH>.csv
        data/processed/monthly_avg/humidity/all_sensors/<MONTH>.csv
Output: data/processed/plots/sensors_map_leaflet_<MONTH>.html
"""

from pathlib import Path

import folium
import pandas as pd

MONTH = "2024-01"  # swap for a year tag once yearly-avg files exist, rest of the script stays the same
BASE_DIR = Path(__file__).resolve().parent.parent
MONTHLY_DIR = BASE_DIR / "data" / "processed" / "monthly_avg"
OUT_DIR = BASE_DIR / "data" / "processed" / "plots"

PM_PATH = MONTHLY_DIR / "all_pm_sensors" / f"germany_monthly_avg_ALLPM_{MONTH}.csv"
UBA_PATH = MONTHLY_DIR / "uba" / f"pm_reference_stations_{MONTH}.csv"
HUMIDITY_PATH = MONTHLY_DIR / "humidity" / "all_sensors" / f"{MONTH}.csv"

BBOX_GERMANY = {"lat_min": 47.2, "lat_max": 55.1, "lon_min": 5.8, "lon_max": 15.1}
CENTER = [(BBOX_GERMANY["lat_min"] + BBOX_GERMANY["lat_max"]) / 2,
          (BBOX_GERMANY["lon_min"] + BBOX_GERMANY["lon_max"]) / 2]


def main() -> None:
    pm = pd.read_csv(PM_PATH)
    uba = pd.read_csv(UBA_PATH)
    humidity = pd.read_csv(HUMIDITY_PATH)

    m = folium.Map(location=CENTER, zoom_start=6, tiles="CartoDB positron")

    # draw order: dense/low-priority layers first, sparse/high-priority reference stations last (on top)
    humidity_layer = folium.FeatureGroup(name=f"humidity sensors ({len(humidity)})")
    for _, row in humidity.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=None,
            fill=True,
            fill_color="blue",
            fill_opacity=0.4,
            weight=0,
            popup=f"sensor {row['sensor_id']} ({row['sensor_type']})<br>temp: {row['temperature']:.1f}<br>humidity: {row['humidity']:.1f}",
        ).add_to(humidity_layer)
    humidity_layer.add_to(m)

    pm_layer = folium.FeatureGroup(name=f"low-qual PM sensors ({len(pm)})")
    for _, row in pm.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color=None,
            fill=True,
            fill_color="green",
            fill_opacity=0.4,
            weight=0,
            popup=f"sensor {row['sensor_id']} ({row['sensor_type']})<br>P1: {row['P1']:.1f}<br>P2: {row['P2']:.1f}",
        ).add_to(pm_layer)
    pm_layer.add_to(m)

    uba_layer = folium.FeatureGroup(name=f"UBA reference stations ({len(uba)})")
    for _, row in uba.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="black",
            fill=True,
            fill_color="saddlebrown",
            fill_opacity=0.9,
            weight=1,
            popup=f"{row['station_name']} ({row['station_code']})<br>PM10: {row['PM10']:.1f}<br>PM2.5: {row['PM2_5']:.1f}",
        ).add_to(uba_layer)
    uba_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sensors_map_leaflet_{MONTH}.html"
    m.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
