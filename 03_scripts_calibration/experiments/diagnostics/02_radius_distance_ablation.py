"""
Representativeness-vs-distance diagnostic.

we fit each sensor's (a,b) against its nearest station, then validate against its second-nearest
station's actual daily values. Bin sensors by distance-to-nearest and watch holdout RMSE. If it stays
flat out to large distances, a hard radius cutoff is unnecessary. If it rises sharply past some
distance, that's empirical support for a cutoff near that point.
This is to check how detrimental it is to always go to nearest station regardless of distance

Also reports the naive/uncorrected RMSE per bin as a baseline, so you can see
whether per-sensor fitting still beats "no correction at all" even at long
range, or whether it stops helping past some distance.

Input:  data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet
        data/processed/hourly/humidity/all_sensors/<YYYY-MM>.parquet
        data/processed/hourly/pm/nodes/<type>_<YYYY-MM>.parquet
        data/processed/hourly/humidity/nodes/<type>_<YYYY-MM>.parquet
        data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
        data/processed/calibration/global_kappa.json  (optional legacy input)
Output: printed distance-bin table, plus
        data/processed/calibration/radius_diagnostic.csv (one row per bin)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PROC = BASE_DIR / "data" / "processed"

PM_HOURLY_DIR = PROC / "hourly" / "pm" / "all_pm_sensors"
RH_HOURLY_DIR = PROC / "hourly" / "humidity" / "all_sensors"
NODES_DIR = PROC / "hourly" / "pm" / "nodes"
RH_NODES_DIR = PROC / "hourly" / "humidity" / "nodes"
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"
GLOBAL_KAPPA_PATH = PROC / "calibration" / "global_kappa.json"
OUT_PATH = PROC / "calibration" / "radius_diagnostic.csv"

UTC_TO_MEZ_HOURS = 1
MAX_RH_DIST_KM = 20.0
MIN_HOURS_PER_DAY = 12
MIN_DAYS_FOR_FIT = 10
MIN_DAYS_FOR_VALIDATION = 10
B_MIN, B_MAX = 0.3, 2.5

# distance-to-nearest bins, km. Widened at the tail since far sensors are sparse.
DIST_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 99999]

POLLUTANTS = {
    "PM10": {"lowcost": "P1", "ref": "PM10"},
    "PM2.5": {"lowcost": "P2", "ref": "PM2.5"},
}

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def growth_divisor(rh, kappa):
    return 1.0 + kappa / (-1.0 + 100.0 / rh)


def discover_months():
    if not PM_HOURLY_DIR.exists():
        return []
    return sorted(p.stem for p in PM_HOURLY_DIR.glob("*.parquet")
                  if MONTH_RE.match(p.stem))


def _load_node_coords(directory, month):
    parts = [pd.read_parquet(p) for p in directory.glob(f"*_{month}.parquet")]
    if not parts:
        raise FileNotFoundError(f"no node coordinate files for {month} in {directory}")
    nodes = pd.concat(parts, ignore_index=True)
    return nodes.drop_duplicates(subset="location", keep="first")[["location", "lat", "lon"]]


def load_nodes(month):
    return _load_node_coords(NODES_DIR, month)


def nearest_within(src, dst, max_km, dst_name="dst_location"):
    d = haversine_km(
        src["lat"].values[:, None], src["lon"].values[:, None],
        dst["lat"].values[None, :], dst["lon"].values[None, :],
    )
    best = d.argmin(axis=1)
    dist = d[np.arange(len(src)), best]
    out = pd.DataFrame({
        "location": src["location"].values,
        dst_name: dst["location"].values[best],
        "dist_km": dist,
    })
    return out[out["dist_km"] <= max_km].reset_index(drop=True)


def load_hourly(month):
    pm_path = PM_HOURLY_DIR / f"{month}.parquet"
    rh_path = RH_HOURLY_DIR / f"{month}.parquet"
    if not pm_path.exists() or not rh_path.exists():
        return None
    pm = pd.read_parquet(pm_path)[["location", "hour", "P1", "P2"]]
    rh = pd.read_parquet(rh_path)[["location", "hour", "humidity", "humidity_clip90"]]
    df = pm.merge(rh, on=["location", "hour"], how="inner")
    unpaired = set(pm["location"].unique()) - set(rh["location"].unique())
    if unpaired and MAX_RH_DIST_KM > 0:
        src = load_nodes(month)
        src = src[src["location"].isin(unpaired)]
        dst = _load_node_coords(RH_NODES_DIR, month)
        dst = dst[dst["location"].isin(set(rh["location"].unique()))]
        if not src.empty and not dst.empty:
            link = nearest_within(src, dst, MAX_RH_DIST_KM, dst_name="rh_location")
            if not link.empty:
                extra = (
                    pm[pm["location"].isin(link["location"])]
                    .merge(link[["location", "rh_location"]], on="location", how="inner")
                    .merge(rh.rename(columns={"location": "rh_location"}),
                           on=["rh_location", "hour"], how="inner")
                )
                if not extra.empty:
                    df = pd.concat([df, extra[df.columns]], ignore_index=True)
    df["date"] = (df["hour"] + pd.Timedelta(hours=UTC_TO_MEZ_HOURS)).dt.date
    return df


def to_daily_dry(hourly, kappa):
    h = hourly.copy()
    div = growth_divisor(h["humidity_clip90"].values, kappa)
    for spec in POLLUTANTS.values():
        h[spec["lowcost"]] = h[spec["lowcost"]].values / div
    agg = {spec["lowcost"]: (spec["lowcost"], "mean") for spec in POLLUTANTS.values()}
    agg["n_hours"] = ("hour", "nunique")
    d = h.groupby(["location", "date"]).agg(**agg).reset_index()
    return d[d["n_hours"] >= MIN_HOURS_PER_DAY].copy()


def load_uba(year):
    path = Path(str(UBA_DAILY).format(year=year))
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for col in ("PM10", "PM2.5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["station_code", "lat", "lon", "date", "PM10", "PM2.5"]]


def nearest_two_stations(nodes, uba):
    """For each node, the nearest AND second-nearest station, with distances."""
    stations = uba.drop_duplicates("station_code")[["station_code", "lat", "lon"]].reset_index(drop=True)
    d = haversine_km(
        nodes["lat"].values[:, None], nodes["lon"].values[:, None],
        stations["lat"].values[None, :], stations["lon"].values[None, :],
    )
    order = np.argsort(d, axis=1)
    idx1 = order[:, 0]
    idx2 = order[:, 1] if d.shape[1] > 1 else order[:, 0]
    rows = np.arange(len(nodes))
    return pd.DataFrame({
        "location": nodes["location"].values,
        "station_1": stations["station_code"].values[idx1],
        "dist_km_1": d[rows, idx1],
        "station_2": stations["station_code"].values[idx2],
        "dist_km_2": d[rows, idx2],
    })


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def run_diagnostic(months, year):
    uba = load_uba(year)
    kappas = {"PM10": 0.0, "PM2.5": 0.0}
    if GLOBAL_KAPPA_PATH.exists():
        saved = json.loads(GLOBAL_KAPPA_PATH.read_text())
        for name in POLLUTANTS:
            if name in saved:
                kappas[name] = saved[name]["kappa"]
        print(f"Using fitted kappa from {GLOBAL_KAPPA_PATH.name}: {kappas}\n")
    else:
        print(f"No {GLOBAL_KAPPA_PATH.name} found -- using kappa=0 for all pollutants "
              f"(no legacy global_kappa.json found; using kappa=1.0)\n")

    daily_parts = []
    nodes_parts = []
    for month in months:
        h = load_hourly(month)
        if h is None:
            continue
        nodes_parts.append(load_nodes(month))
        d0 = None
        merged = None
        for name, spec in POLLUTANTS.items():
            dd = to_daily_dry(h, kappas[name])[["location", "date", spec["lowcost"]]]
            merged = dd if merged is None else merged.merge(dd, on=["location", "date"], how="outer")
        daily_parts.append(merged)

    if not daily_parts:
        print("No hourly data found.")
        return

    daily = pd.concat(daily_parts, ignore_index=True)
    nodes = pd.concat(nodes_parts, ignore_index=True).drop_duplicates("location")

    links = nearest_two_stations(nodes, uba)
    print(f"{len(links):,} sensors matched to a nearest + second-nearest station "
          f"(no radius cutoff applied)\n")

    ref = uba.copy()
    results = []
    for name, spec in POLLUTANTS.items():
        lc = spec["lowcost"]
        rc = spec["ref"]

        m1 = (daily[["location", "date", lc]]
              .merge(links[["location", "station_1", "dist_km_1"]], on="location", how="inner")
              .merge(ref[["station_code", "date", rc]].rename(
                  columns={"station_code": "station_1", rc: "ref1"}),
                  on=["station_1", "date"], how="inner")
              .dropna(subset=[lc, "ref1"]))

        m2 = (daily[["location", "date", lc]]
              .merge(links[["location", "station_2", "dist_km_2"]], on="location", how="inner")
              .merge(ref[["station_code", "date", rc]].rename(
                  columns={"station_code": "station_2", rc: "ref2"}),
                  on=["station_2", "date"], how="inner")
              .dropna(subset=[lc, "ref2"]))

        rows = []
        for loc, g1 in m1.groupby("location"):
            if len(g1) < MIN_DAYS_FOR_FIT or np.std(g1[lc].values) < 1e-6:
                continue
            a, b = ols(g1[lc].values, g1["ref1"].values)
            if not (B_MIN <= b <= B_MAX):
                continue

            g2 = m2[m2["location"] == loc]
            if len(g2) < MIN_DAYS_FOR_VALIDATION:
                continue

            pred = a + b * g2[lc].values
            actual = g2["ref2"].values
            raw = g2[lc].values
            rows.append({
                "location": loc,
                "dist_km_1": float(g1["dist_km_1"].iloc[0]),
                "dist_km_2": float(g2["dist_km_2"].iloc[0]),
                "rmse_corrected": float(np.sqrt(np.mean((pred - actual) ** 2))),
                "rmse_raw": float(np.sqrt(np.mean((raw - actual) ** 2))),
                "n_validation_days": len(g2),
            })

        if not rows:
            print(f"  {name}: no sensors with enough data on BOTH nearest and "
                  f"second-nearest station to validate")
            continue

        val = pd.DataFrame(rows)
        val["dist_bin"] = pd.cut(val["dist_km_1"], DIST_BINS, right=False)

        print(f"  {name}: {len(val):,} sensors validated against their "
              f"second-nearest station\n")
        print(f"  {'dist_km_1 bin':<18}{'n_sensors':>10}{'RMSE corrected':>16}"
              f"{'RMSE raw':>12}{'improvement':>13}")
        for interval, g in val.groupby("dist_bin", observed=True):
            if g.empty:
                continue
            rc_med = g["rmse_corrected"].median()
            rr_med = g["rmse_raw"].median()
            improvement = rr_med - rc_med
            print(f"  {str(interval):<18}{len(g):>10}{rc_med:>16.2f}"
                  f"{rr_med:>12.2f}{improvement:>13.2f}")
            results.append({
                "pollutant": name, "dist_bin": str(interval), "n_sensors": len(g),
                "rmse_corrected_median": rc_med, "rmse_raw_median": rr_med,
                "improvement": improvement,
            })
        print()

    if results:
        PROC.joinpath("calibration").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(OUT_PATH, index=False)
        print(f"saved bin table -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()

    months = args.months or discover_months()
    if not months:
        print(f"No hourly PM files found in {PM_HOURLY_DIR}.")
        return

    run_diagnostic(months, args.year)


if __name__ == "__main__":
    main()
