"""
Assign every EEA station to a cross-validation fold and write it to disk, plus a
static QC map so the German east/north split can be verified visually.

Fold membership is computed once here and frozen into station_fold.csv. The CV
trainer, the final-model trainer, and the test-inference script all read that
column instead of recomputing routing logic.

Folds:
Spain, Portugal, Andorra
France, Ireland
Italy, Malta
South/West Germany, Austria, Switzerland
Benelux, Finalnd, Sweden, Norway, Denmark, Iceland
Balkans
Turkey, Greece, Bulgaria, Cyprus
Poland, Lithuania, Latvia, Estonia

Outputs:
  data/processed/eea/station_fold.csv   (station_code, country, land, fold, lat, lon)
  data/processed/eea/fold_map.png       (static scatter, colour = fold)

Run:
  python3 00_assign_folds.py
  python3 00_assign_folds.py --out-csv custom.csv --out-plot custom.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # no display needed; write PNG straight to disk
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
LABELS = PROC / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"
STATION_LAND = PROC / "uba" / "station_land.csv"   # station_code -> land (DEUB resolved)
OUT_CSV = PROC / "eea" / "station_fold.csv"
OUT_PLOT = PROC / "eea" / "fold_map.png"

FOLDS = {
    "fold1_iberia": ["PT", "ES", "AD"],
    "fold2_france": ["FR", "IE"],
    "fold3_italy": ["IT", "MT"],
    "fold4_alpine": ["DE", "CH", "AT"],   # DE = western/southern Laender only
    "fold5_north": ["NL", "BE", "LU", "DK", "SE", "NO", "FI", "IS"],
    "fold6_balkan_e": ["CZ", "SK", "HU", "SI", "HR", "BA", "RS", "XK", "ME", "RO"],
    "fold7_balkan_s": ["BG", "AL", "GR", "CY", "TR", "MK"],
    "fold8_poland_baltics": ["PL", "LT", "LV", "EE"],   # ordinary CV fold now
}
# SEALED TEST SET = eastern + northern German Laender ONLY: never trained on,
# never a CV fold -> labelled "TEST" here. Western/southern German Laender stay
# in fold4_alpine. There are no whole-country test members anymore (PL/LT/LV/EE
# became fold8_poland_baltics).
COUNTRY_TO_FOLD = {cc: f for f, ccs in FOLDS.items() for cc in ccs}

DE_TEST_LAENDER = {
    "Brandenburg", "Mecklenburg-Vorpommern", "Sachsen", "Sachsen-Anhalt",
    "Thueringen", "Berlin",                                       # east
    "Hamburg", "Bremen", "Niedersachsen", "Schleswig-Holstein",  # north
}

# stable colour per fold for the QC plot; test set + unassigned get greys
FOLD_ORDER = list(FOLDS)
CMAP = plt.get_cmap("tab10")
FOLD_COLOR = {f: CMAP(i % 10) for i, f in enumerate(FOLD_ORDER)}
FOLD_COLOR["TEST"] = (0.35, 0.35, 0.35, 1.0)
FOLD_COLOR["UNASSIGNED"] = (0.80, 0.80, 0.80, 1.0)


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def load_de_land():
    if STATION_LAND.exists():
        sl = pd.read_csv(STATION_LAND, dtype={"station_code": str})
        return dict(zip(sl["station_code"], sl["land"]))
    print(f"WARNING: {STATION_LAND} not found -- no German Land split; all DE -> fold4_alpine")
    return {}


def assign_fold(code, de_land):
    """CV fold, or 'TEST' for the sealed set (east/north German Laender ONLY), or
    None if unlisted. Other DE (west/south) -> fold4_alpine. PL/LT/LV/EE now map
    to fold8_poland_baltics via the country table like any other CV country."""
    cc = code[:2]
    if cc == "DE":
        land = de_land.get(code)
        if land in DE_TEST_LAENDER:
            return "TEST"                 # east/north DE -> sealed test set
    return COUNTRY_TO_FOLD.get(cc)   # None if the country isn't in any cluster


def build_table():
    lab = pd.read_csv(LABELS, dtype={"station_code": str})
    lat_col = _find_col(lab, ["lat", "latitude", "Latitude", "LAT"])
    lon_col = _find_col(lab, ["lon", "lng", "longitude", "Longitude", "LON"])
    if lat_col is None or lon_col is None:
        raise SystemExit(f"ERROR: no lat/lon columns in {LABELS} (has {list(lab.columns)})")
    # one row per station: median coordinate (defends against per-day jitter)
    g = lab.groupby("station_code")
    coords = g[[lat_col, lon_col]].median().rename(
        columns={lat_col: "lat", lon_col: "lon"})
    df = coords.reset_index()
    df["country"] = df["station_code"].str[:2]

    de_land = load_de_land()
    df["land"] = df["station_code"].map(lambda c: de_land.get(c) if c[:2] == "DE" else "")
    df["fold"] = df["station_code"].map(lambda c: assign_fold(c, de_land))
    # keep unlisted (fold is None) visible in the plot, but tag them
    df["fold"] = df["fold"].fillna("UNASSIGNED")
    return df


def summarise(df):
    print(f"{len(df)} stations total")
    print("  per fold:")
    for f in FOLD_ORDER + ["TEST", "UNASSIGNED"]:
        sub = df[df["fold"] == f]
        if len(sub) == 0 and f in ("UNASSIGNED",):
            continue
        note = ""
        if f == "TEST" and len(sub):
            de_sub = sub[sub["country"] == "DE"]
            if len(de_sub):
                lands = de_sub["land"].value_counts().to_dict()
                note = "   DE: " + ", ".join(f"{k}:{v}" for k, v in lands.items())
        print(f"    {f:<16} {len(sub):>4}{note}")
    # explicit sanity line: how DE stations split between TEST and alpine
    de = df[df["country"] == "DE"]
    to_test = int((de["fold"] == "TEST").sum())
    alp = int((de["fold"] == "fold4_alpine").sum())
    other = int((~de["fold"].isin(["TEST", "fold4_alpine"])).sum())
    print(f"  DE check: {len(de)} German stations -> {to_test} in sealed TEST, "
          f"{alp} in fold4_alpine, {other} elsewhere/unmapped")
    if len(de) and to_test == 0:
        print("  !! 0 German stations in TEST -- station_land.csv codes probably "
              "don't match the EEA station_codes. Check the code formats.")


def make_plot(df, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    # draw CV folds first, then TEST and UNASSIGNED on top in grey
    order = FOLD_ORDER + ["TEST", "UNASSIGNED"]
    for f in order:
        sub = df[df["fold"] == f]
        if len(sub) == 0:
            continue
        # emphasise the sealed TEST set (now incl. east/north DE): larger, edged
        is_test = f == "TEST"
        ax.scatter(sub["lon"], sub["lat"], s=26 if is_test else 10,
                   c=[FOLD_COLOR[f]], label=f"{f} ({len(sub)})",
                   edgecolors="black" if is_test else "none",
                   linewidths=0.4 if is_test else 0.0,
                   alpha=0.9 if is_test else 0.7, zorder=3 if is_test else 2)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("EEA station fold assignment\n"
                 "(TEST = east/north German Laender only, black-edged)")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(df["lat"].median()))))  # rough equal-area
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=1)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--out-plot", default=str(OUT_PLOT))
    ap.add_argument("--no-plot", action="store_true", help="skip the QC map")
    args = ap.parse_args()

    df = build_table()
    summarise(df)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[["station_code", "country", "land", "fold", "lat", "lon"]].to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    if not args.no_plot:
        make_plot(df, Path(args.out_plot))


if __name__ == "__main__":
    main()
