"""
Tag German EEA stations with their Land and a held-out / train split.

The held-out region is sparse eastern + northern Germany; everything else in
Germany trains. Land is read from the station EoI code, which encodes it
directly (DEBW = Baden-Wuerttemberg, DEBB = Brandenburg, ...) for most
stations, so no shapefile is needed there.

The 7 UBA federal background stations (EoI prefix DEUB...) are an exception:
their code doesn't encode a Land. Rather than drop them, they're assigned a
Land by matching their reported lat/lon against the known locations of the
7 fixed UBA background sites (Westerland, Zingst, Waldhof, Neuglobsow,
Schmuecke, Schauinsland, Zugspitze) via nearest-neighbour distance. This list
is stable (the network hasn't changed in decades), so a hardcoded lookup is
more reliable here than a generic point-in-polygon/bounding-box approach,
which can misassign near-border sites (e.g. Neuglobsow sits close to the
Brandenburg/Mecklenburg-Vorpommern border).

Writes data/processed/eea/germany_region_split.csv:
    station_code, land, region   (region in {"heldout", "train"})
"""
import math
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
META_CACHE = PROC / "eea" / "station_meta.csv"
OUT = PROC / "eea" / "germany_region_split.csv"

# EoI 2-letter Land codes (the two chars after "DE")
LAND = {
    "BW": "Baden-Wuerttemberg", "BY": "Bayern", "BE": "Berlin", "BB": "Brandenburg",
    "HB": "Bremen", "HH": "Hamburg", "HE": "Hessen", "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen", "NW": "Nordrhein-Westfalen", "RP": "Rheinland-Pfalz",
    "SL": "Saarland", "SN": "Sachsen", "ST": "Sachsen-Anhalt",
    "SH": "Schleswig-Holstein", "TH": "Thueringen",
    "UB": "Umweltbundesamt",  # federal background stations, resolved by coordinates below
}

# sparse east + north = held out; the rest of Germany trains
HELDOUT_LANDS = {
    "Brandenburg", "Berlin", "Sachsen", "Sachsen-Anhalt", "Thueringen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen", "Hamburg", "Bremen", "Schleswig-Holstein",
}

# The 7 fixed UBA (Umweltbundesamt) background stations, with their known
# approximate coordinates (WGS84) and the Land each actually sits in.
# Source: umweltbundesamt.de station network pages. Coordinates are accurate
# to well within the ~15 km tolerance used for matching below.
UBA_BACKGROUND_STATIONS = [
    # name,          lat,      lon,      land
    ("Westerland",   54.9130,  8.3113,   "Schleswig-Holstein"),
    ("Zingst",       54.4337, 12.6866,   "Mecklenburg-Vorpommern"),
    ("Waldhof",      52.8027, 10.7592,   "Niedersachsen"),
    ("Neuglobsow",   53.1691, 13.0326,   "Brandenburg"),
    ("Schmuecke",    50.6497, 10.7708,   "Thueringen"),
    ("Schauinsland", 47.9147,  7.9061,   "Baden-Wuerttemberg"),
    ("Zugspitze",    47.4165, 10.9805,   "Bayern"),
]

MATCH_TOLERANCE_KM = 15.0


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def land_from_code(code):
    if not isinstance(code, str) or not code.startswith("DE") or len(code) < 4:
        return None
    return LAND.get(code[2:4])


def land_from_coords(lat, lon, tolerance_km=MATCH_TOLERANCE_KM):
    """Assign a Land by nearest-neighbour match against the 7 known UBA
    background station coordinates. Returns None if nothing is within
    tolerance (so callers can flag it instead of silently mis-assigning)."""
    if pd.isna(lat) or pd.isna(lon):
        return None, None
    best_name, best_land, best_dist = None, None, float("inf")
    for name, ref_lat, ref_lon, land in UBA_BACKGROUND_STATIONS:
        d = _haversine_km(lat, lon, ref_lat, ref_lon)
        if d < best_dist:
            best_name, best_land, best_dist = name, land, d
    if best_dist <= tolerance_km:
        return best_land, best_name
    return None, None


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    meta = pd.read_csv(META_CACHE, dtype={"station_code": str})
    de = meta[meta["station_code"].str.startswith("DE")].copy()
    de["land"] = de["station_code"].map(land_from_code)

    # Resolve DEUB (and any other unrecognised-code) rows by coordinates.
    unresolved = de["land"].isna()
    if unresolved.any():
        lat_col = _find_col(de, ["lat", "latitude", "Latitude", "y"])
        lon_col = _find_col(de, ["lon", "lng", "longitude", "Longitude", "x"])
        if lat_col is None or lon_col is None:
            print(
                "WARNING: could not find lat/lon columns in station_meta.csv "
                "(looked for lat/latitude/y and lon/lng/longitude/x); "
                f"{int(unresolved.sum())} stations left unresolved."
            )
        else:
            for idx in de.index[unresolved]:
                lat, lon = de.at[idx, lat_col], de.at[idx, lon_col]
                land, matched_name = land_from_coords(lat, lon)
                if land is not None:
                    de.at[idx, "land"] = land
                else:
                    code = de.at[idx, "station_code"]
                    print(
                        f"WARNING: {code} at ({lat}, {lon}) did not match any "
                        f"known UBA background station within {MATCH_TOLERANCE_KM} km "
                        "-> left unassigned"
                    )

    still_unknown = de["land"].isna()
    if still_unknown.any():
        bad = de.loc[still_unknown, "station_code"]
        print(f"Dropping {len(bad)} stations with no resolvable Land:\n{bad.tolist()}")
        de = de[~still_unknown]

    de["region"] = de["land"].map(lambda l: "heldout" if l in HELDOUT_LANDS else "train")

    out = de[["station_code", "land", "region"]].sort_values(["region", "land"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"{len(out)} German stations tagged -> {OUT}\n")
    print("held-out region:")
    for land, n in out[out.region == "heldout"].land.value_counts().sort_index().items():
        print(f"  {land:<26} {n}")
    print(f"  {'TOTAL heldout':<26} {(out.region=='heldout').sum()}")
    print("\ntrain region:")
    for land, n in out[out.region == "train"].land.value_counts().sort_index().items():
        print(f"  {land:<26} {n}")
    print(f"  {'TOTAL train':<26} {(out.region=='train').sum()}")


if __name__ == "__main__":
    main()
