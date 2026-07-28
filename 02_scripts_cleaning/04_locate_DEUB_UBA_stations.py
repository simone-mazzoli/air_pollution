"""

Some UBA state codes are "DEUB" (federally-administered, 7)
for these we check coordinates and find the actual state. It writes a station -> Land
lookup used by the leave-one-Land-out calibration.

Input:  data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
        data/processed/germany_states.geojson   (cached by the map script; fetched if absent)
Output: data/processed/uba/station_land.csv     (station_code, lat, lon, land, source)

If a DEUB point falls outside every polygon (happens for islands) it's assigned to the nearest polygon and marked "coords_nearest" so it's visible.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"
GEOJSON_CACHE = PROC / "germany_states.geojson"
OUT_PATH = PROC / "uba" / "station_land.csv"
YEAR = 2024

GEOJSON_URL = ("https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/"
               "main/2_bundeslaender/4_niedrig.geo.json")

# code prefix (ISO 3166-2:DE) -> Land, ASCII spelling used across the pipeline
STATE_CODE_TO_LAND = {
    "BW": "Baden-Wuerttemberg", "BY": "Bayern", "BE": "Berlin", "BB": "Brandenburg",
    "HB": "Bremen", "HH": "Hamburg", "HE": "Hessen", "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen", "NW": "Nordrhein-Westfalen", "RP": "Rheinland-Pfalz",
    "SL": "Saarland", "SN": "Sachsen", "ST": "Sachsen-Anhalt",
    "SH": "Schleswig-Holstein", "TH": "Thueringen",
}

# GeoJSON feature name -> same ASCII spelling
GEOJSON_NAME_TO_LAND = {
    "Baden-W\u00fcrttemberg": "Baden-Wuerttemberg", "Bayern": "Bayern", "Berlin": "Berlin",
    "Brandenburg": "Brandenburg", "Bremen": "Bremen", "Hamburg": "Hamburg",
    "Hessen": "Hessen", "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Niedersachsen": "Niedersachsen", "Nordrhein-Westfalen": "Nordrhein-Westfalen",
    "Rheinland-Pfalz": "Rheinland-Pfalz", "Saarland": "Saarland", "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt", "Schleswig-Holstein": "Schleswig-Holstein",
    "Th\u00fcringen": "Thueringen",
}


def load_geojson():
    if not GEOJSON_CACHE.exists():
        import requests
        GEOJSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(GEOJSON_URL, timeout=30)
        resp.raise_for_status()
        GEOJSON_CACHE.write_text(resp.text)
    return json.loads(GEOJSON_CACHE.read_text())


def build_land_lookup(geojson):
    lands = []
    for feature in geojson["features"]:
        name = feature.get("properties", {}).get("name")
        land = GEOJSON_NAME_TO_LAND.get(name, name)
        geom = feature["geometry"]
        polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
                 else [geom["coordinates"]])
        lands.append((land, polys))
    return lands


def point_in_ring(lon, lat, ring):
    """Ray casting. ring is a list of [lon, lat] pairs."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_in_polygon(lon, lat, polygon):
    """polygon is [outer_ring, hole1, ...]: inside outer, outside every hole."""
    if not point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon[1:])


def locate(lon, lat, lands):
    for land, polys in lands:
        if any(point_in_polygon(lon, lat, poly) for poly in polys):
            return land, "coords"
    # missed every polygon (island/coastline on the coarse map): nearest vertex wins
    best_land, best_d2 = None, float("inf")
    for land, polys in lands:
        for poly in polys:
            for ring in poly:
                for x, y in ring:
                    d2 = (x - lon) ** 2 + (y - lat) ** 2
                    if d2 < best_d2:
                        best_d2, best_land = d2, land
    return best_land, "coords_nearest"


def land_from_code(code):
    return STATE_CODE_TO_LAND.get(code[2:4])


def main():
    path = Path(str(UBA_DAILY).format(year=YEAR))
    df = pd.read_csv(path)
    stations = df.drop_duplicates("station_code")[["station_code", "lat", "lon"]].copy()

    lands = build_land_lookup(load_geojson())

    rows = []
    for _, s in stations.iterrows():
        code = s["station_code"]
        land = land_from_code(code)
        if land is not None:
            source = "code"
        else:
            # only the DEUB* (and any other non-state-coded) stations reach here
            land, source = locate(s["lon"], s["lat"], lands)
        rows.append({"station_code": code, "lat": s["lat"], "lon": s["lon"],
                     "land": land, "source": source})

    out = pd.DataFrame(rows)

    resolved = out[out["source"].str.startswith("coords")]
    print(f"Coordinate-resolved stations ({len(resolved)}):")
    for _, r in resolved.iterrows():
        flag = "  [nearest]" if r["source"] == "coords_nearest" else ""
        print(f"  {r['station_code']}  ({r['lat']:.3f}, {r['lon']:.3f})  ->  "
              f"{r['land']}{flag}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(out)} stations "
          f"({(out['source'] == 'code').sum()} by code, {len(resolved)} by coords) "
          f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
