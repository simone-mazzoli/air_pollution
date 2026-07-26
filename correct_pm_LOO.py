"""
Leave-one-Land-out version of the PM calibration.

the project requires evaluating on every German Land in turn
(leave-one-out across all 16), not just one fixed held-out Land. A single global
calibration fit on ALL reference stations would leak information , so we refit the calibration 16 times, once per fold,
each time excluding that fold's Land's reference stations from the fit.

UBA station codes already encode the Land. The correction itself is applied uniformly to the
whole national low-cost sensor network in a given fold (not just to sensors
physically inside the held-out Land. so no per-sensor Land lookup is
required anywhere.

Same physics/model as correct_pm.py:
  PM_dry = PM_raw / (1 + kappa / (-1 + 100/RH))        applied hourly
  PM_cal = a + b * PM_dry                              linear residual calibration

Input:  data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet
        data/processed/hourly/humidity/all_sensors/<YYYY-MM>.parquet
        data/processed/hourly/pm/nodes/<type>_<YYYY-MM>.parquet
        data/processed/hourly/humidity/nodes/<type>_<YYYY-MM>.parquet
        data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
Output: data/processed/calibration/params_by_land.json
        (16 entries, one per held-out Land, each with its own PM10/PM2.5 params)

        --apply-land "<Land name>" additionally writes the corrected dataset
        for that one fold to:
        data/processed/corrected/land_folds/<Land>/daily/<YYYY-MM>.csv
        data/processed/corrected/land_folds/<Land>/monthly/<YYYY-MM>.csv
        data/processed/corrected/land_folds/<Land>/annual/<YEAR>.csv
        (one row per sensor: n_days_total-weighted annual mean, with the
        year-level completeness filter MIN_DAYS_PER_YEAR applied -- this is
        the actual regression target for the CNN)

example of one row:
{
  "Baden-Wuerttemberg": {
    "PM10": {
      "kappa": 0.06,
      "a": 16.66,
      "b": 0.079,
      "rmse": 12.1,
      "n_pairs": 73329,
      "n_stations": 233,
      "holdout_rmse_uncorrected": 43.03,
      "holdout_rmse_corrected": 12.38
    },
    "PM2.5": {
      "kappa": 0.0,
      "a": 10.87,
      "b": 0.080,
      "rmse": 7.85,
      "n_pairs": 61422,
      "n_stations": 203,
      "holdout_rmse_uncorrected": 8.83,
      "holdout_rmse_corrected": 7.90
    }
  },
  "Bayern": { "...": "..." },
  "Berlin": { "...": "..." }
}

"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROC = BASE_DIR / "data" / "processed"

PM_HOURLY_DIR = PROC / "hourly" / "pm" / "all_pm_sensors"
RH_HOURLY_DIR = PROC / "hourly" / "humidity" / "all_sensors"
NODES_DIR = PROC / "hourly" / "pm" / "nodes"
RH_NODES_DIR = PROC / "hourly" / "humidity" / "nodes"
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"

CALIB_DIR = PROC / "calibration"
CORR_LAND_DIR = PROC / "corrected" / "land_folds"

UTC_TO_MEZ_HOURS = 1  # Sensor.Community = UTC, UBA = MEZ (fixed UTC+1, no DST)

RADIUS_KM = 5.0  # tight on purpose, PM varies over short distances

# Humidity varies smoothly over tens of km (unlike PM). A node with no humidity sensor
# of its own borrows readings from the nearest node that has one, within this distance.
# This is NOT Land-restricted: humidity is an input feature, not the PM ground-truth
# label, so borrowing it across a Land boundary doesn't leak label information --
# only the calibration FIT (which reference stations inform a,b,kappa) needs the
# leave-one-Land-out restriction.
MAX_RH_DIST_KM = 20.0

MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_MONTH = 10
MIN_DAYS_PER_YEAR = 60  # yearly completeness filter, per the assignment -- adjust once
                        # all 12 months are in; 60 is a placeholder for partial-year runs
HOLDOUT_FRAC = 0.25  # within-fold holdout, for the printed sanity-check RMSE only
KAPPA_GRID = np.round(np.arange(0.0, 1.21, 0.02), 3)

POLLUTANTS = {
    "PM10": {"lowcost": "P1", "ref": "PM10"},
    "PM2.5": {"lowcost": "P2", "ref": "PM2.5"},
}

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

# ISO 3166-2:DE state codes, as they appear in UBA station codes (e.g. "DEBW021").
STATE_CODE_TO_LAND = {
    "BW": "Baden-Wuerttemberg",
    "BY": "Bayern",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "HB": "Bremen",
    "HH": "Hamburg",
    "HE": "Hessen",
    "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen",
    "NW": "Nordrhein-Westfalen",
    "RP": "Rheinland-Pfalz",
    "SL": "Saarland",
    "SN": "Sachsen",
    "ST": "Sachsen-Anhalt",
    "SH": "Schleswig-Holstein",
    "TH": "Thueringen",
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def growth_divisor(rh, kappa):
    return 1.0 + kappa / (-1.0 + 100.0 / rh)  # rh clipped at 90 upstream, denom >= 0.111


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
    """For each src node, the nearest dst node within max_km."""
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
    """PM joined to humidity, one row per (location, hour). Not Land-restricted --
    see MAX_RH_DIST_KM comment above for why that's fine."""
    pm_path = PM_HOURLY_DIR / f"{month}.parquet"
    rh_path = RH_HOURLY_DIR / f"{month}.parquet"
    if not pm_path.exists() or not rh_path.exists():
        return None

    pm = pd.read_parquet(pm_path)[["location", "hour", "P1", "P2"]]
    rh = pd.read_parquet(rh_path)[["location", "hour", "humidity", "humidity_clip90"]]

    n_pm_nodes = pm.location.nunique()

    df = pm.merge(rh, on=["location", "hour"], how="inner")
    df["rh_source"] = "onnode"
    df["rh_dist_km"] = 0.0
    n_onnode = df.location.nunique()

    unpaired = set(pm["location"].unique()) - set(rh["location"].unique())
    n_borrowed = 0
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
                    .merge(link, on="location", how="inner")
                    .merge(rh.rename(columns={"location": "rh_location"}),
                           on=["rh_location", "hour"], how="inner")
                )
                if not extra.empty:
                    extra["rh_source"] = "nearest"
                    extra = extra.rename(columns={"dist_km": "rh_dist_km"})
                    n_borrowed = extra.location.nunique()
                    df = pd.concat([df, extra[df.columns]], ignore_index=True)

    n_kept = df.location.nunique()
    print(f"  [{month}] humidity: {n_onnode:,} on-node + {n_borrowed:,} borrowed "
          f"= {n_kept:,}/{n_pm_nodes:,} nodes "
          f"({100.0 * n_kept / max(n_pm_nodes, 1):.1f}%), {len(df):,} node-hours")

    df["date"] = (df["hour"] + pd.Timedelta(hours=UTC_TO_MEZ_HOURS)).dt.date
    return df


def to_daily(hourly, value_cols):
    agg = {c: (c, "mean") for c in value_cols}
    agg["n_hours"] = ("hour", "nunique")
    daily = hourly.groupby(["location", "date"]).agg(**agg).reset_index()
    return daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()


def load_uba(year):
    """Loads UBA reference data and tags each station with its Land, derived
    directly from the station code (e.g. "DEBW021" -> "BW" -> Baden-Wuerttemberg).
    No shapefile needed."""
    path = Path(str(UBA_DAILY).format(year=year))
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for col in ("PM10", "PM2.5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    state_code = df["station_code"].str.slice(2, 4)
    df["land"] = state_code.map(STATE_CODE_TO_LAND)
    unmapped = df.loc[df["land"].isna(), "station_code"].unique()
    if len(unmapped):
        print(f"  WARNING: {len(unmapped)} station code(s) didn't map to a known "
              f"Land, dropping: {list(unmapped)[:10]}")
        df = df.dropna(subset=["land"])

    return df[["station_code", "lat", "lon", "date", "PM10", "PM2.5", "land"]]


def colocate(nodes, uba_subset):
    """Nearest UBA station (within uba_subset only) within RADIUS_KM for each node."""
    stations = uba_subset.drop_duplicates("station_code")[["station_code", "lat", "lon"]]
    if stations.empty:
        return pd.DataFrame(columns=["location", "station_code", "dist_km"])

    d = haversine_km(
        nodes["lat"].values[:, None], nodes["lon"].values[:, None],
        stations["lat"].values[None, :], stations["lon"].values[None, :],
    )
    best = d.argmin(axis=1)
    dist = d[np.arange(len(nodes)), best]

    out = pd.DataFrame({
        "location": nodes["location"].values,
        "station_code": stations["station_code"].values[best],
        "dist_km": dist,
    })
    return out[out["dist_km"] <= RADIUS_KM].reset_index(drop=True)


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def build_fit_pairs(months, uba_train):
    """Colocated daily pairs against uba_train ONLY (already Land-filtered by the
    caller), with a PM_dry column precomputed for every kappa."""
    parts = []
    for month in months:
        hourly = load_hourly(month)
        if hourly is None:
            continue
        nodes = load_nodes(month)
        links = colocate(nodes, uba_train)
        if links.empty:
            print(f"  [{month}] no nodes within {RADIUS_KM} km of an in-fold station")
            continue

        # fit on on-node humidity only: borrowed humidity is good enough to apply the
        # correction but would add noise to the parameter estimates themselves
        h = hourly[hourly["location"].isin(links["location"])
                   & (hourly["rh_source"] == "onnode")].copy()
        if h.empty:
            continue

        rh = h["humidity_clip90"].values
        dry = {}
        for kappa in KAPPA_GRID:
            div = growth_divisor(rh, kappa)
            for spec in POLLUTANTS.values():
                dry[f"{spec['lowcost']}_dry_{kappa}"] = h[spec["lowcost"]].values / div
        h = pd.concat([h, pd.DataFrame(dry, index=h.index)], axis=1)

        value_cols = (["humidity_clip90"] + list(dry)
                      + [s["lowcost"] for s in POLLUTANTS.values()])

        daily = to_daily(h, value_cols)
        daily = daily.merge(links, on="location", how="inner")
        parts.append(daily)

    if not parts:
        return None

    pairs = pd.concat(parts, ignore_index=True)
    return pairs.merge(uba_train, on=["station_code", "date"], how="inner")


def fit_pollutant(pairs, lowcost_col, ref_col, rng):
    """Grid search kappa with the linear fit nested inside."""
    df = pairs.dropna(subset=[lowcost_col, ref_col, "humidity_clip90"])
    if df["station_code"].nunique() < 4:
        return None

    stations = np.sort(df["station_code"].unique())
    holdout = set(rng.choice(stations, max(1, int(len(stations) * HOLDOUT_FRAC)),
                             replace=False))
    fit_mask = ~df["station_code"].isin(holdout)
    fit_df, hold_df = df[fit_mask], df[~fit_mask]

    best = None
    for kappa in KAPPA_GRID:
        x = fit_df[f"{lowcost_col}_dry_{kappa}"].values
        y = fit_df[ref_col].values
        a, b = ols(x, y)
        rmse = float(np.sqrt(np.mean((a + b * x - y) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"kappa": float(kappa), "a": float(a), "b": float(b), "rmse": rmse}

    k = best["kappa"]
    xh = hold_df[f"{lowcost_col}_dry_{k}"].values
    yh = hold_df[ref_col].values
    raw = hold_df[lowcost_col].values

    best.update({
        "n_pairs": int(len(df)),
        "n_stations": int(len(stations)),
        "holdout_rmse_uncorrected": float(np.sqrt(np.mean((raw - yh) ** 2))),
        "holdout_rmse_corrected": float(np.sqrt(np.mean(
            (best["a"] + best["b"] * xh - yh) ** 2))),
    })
    return best


def fit_all_lands(months, year, seed):
    """The core leave-one-Land-out loop: for each of the 16 Laender, fit using
    only stations OUTSIDE that Land."""
    uba_all = load_uba(year)
    all_lands = sorted(uba_all["land"].unique())
    print(f"Found {len(all_lands)} Laender with reference stations: {all_lands}\n")

    results = {}
    for held_out_land in all_lands:
        uba_train = uba_all[uba_all["land"] != held_out_land]
        print(f"=== Held-out Land: {held_out_land} "
              f"(fitting on {uba_train['station_code'].nunique()} stations "
              f"from the other {len(all_lands) - 1} Laender) ===")

        pairs = build_fit_pairs(months, uba_train)
        if pairs is None or pairs.empty:
            print(f"  no colocated pairs -- skipping {held_out_land}\n")
            continue

        rng = np.random.default_rng(seed)
        land_params = {}
        for name, spec in POLLUTANTS.items():
            res = fit_pollutant(pairs, spec["lowcost"], spec["ref"], rng)
            if res is None:
                print(f"  {name}: not enough stations to fit")
                continue
            land_params[name] = res
            print(f"  {name}: kappa={res['kappa']:.2f}  a={res['a']:.2f}  "
                  f"b={res['b']:.3f}  ({res['n_pairs']:,} pairs, "
                  f"{res['n_stations']} stations)  "
                  f"holdout RMSE {res['holdout_rmse_uncorrected']:.2f} -> "
                  f"{res['holdout_rmse_corrected']:.2f} ug/m3")

        if land_params:
            results[held_out_land] = land_params
        print()

    return results


def aggregate_annual(held_out_land, months, year):
    """Collapses this fold's monthly outputs into ONE annual value per sensor --
    weighted by n_days per month (so a month with 28 valid days counts more than
    one with 8), with a year-level completeness filter (MIN_DAYS_PER_YEAR) applied
    on top of the per-month filter already baked into the monthly files. This is
    the actual training target the assignment asks for ("annual averages per
    sensor... keep only sensors with enough valid days in the year").
    """
    monthly_dir = CORR_LAND_DIR / held_out_land.replace(" ", "_") / "monthly"
    parts = [pd.read_csv(monthly_dir / f"{m}.csv") for m in months
             if (monthly_dir / f"{m}.csv").exists()]
    if not parts:
        print(f"  [{held_out_land}] no monthly files found -- skipping annual aggregation")
        return

    df = pd.concat(parts, ignore_index=True)
    value_cols = [c for c in df.columns
                  if c not in ("location", "month", "lat", "lon", "n_days")]

    def weighted_mean(g):
        w = g["n_days"].values
        return pd.Series({c: np.average(g[c].values, weights=w) for c in value_cols})

    annual = df.groupby("location").apply(weighted_mean).reset_index()
    annual["n_days_total"] = df.groupby("location")["n_days"].sum().values
    coords = df.groupby("location")[["lat", "lon"]].first().reset_index()
    annual = annual.merge(coords, on="location")

    before = len(annual)
    annual = annual[annual["n_days_total"] >= MIN_DAYS_PER_YEAR].copy()
    print(f"  [{held_out_land}] annual completeness filter (>= {MIN_DAYS_PER_YEAR} "
          f"days across the year): {before:,} -> {len(annual):,} sensors")

    annual_dir = CORR_LAND_DIR / held_out_land.replace(" ", "_") / "annual"
    annual_dir.mkdir(parents=True, exist_ok=True)
    out_path = annual_dir / f"{year}.csv"
    annual.to_csv(out_path, index=False)
    print(f"  [{held_out_land}] wrote {len(annual):,} annual sensor rows -> {out_path}")


def apply_land_fold(months, held_out_land, params, year):
    """Applies ONE fold's calibration to the WHOLE national dataset (not just
    the held-out Land's own sensors) and writes it out -- this is what a CNN
    training run for this fold should actually use as its input features."""
    out_dir = CORR_LAND_DIR / held_out_land.replace(" ", "_")
    daily_dir = out_dir / "daily"
    monthly_dir = out_dir / "monthly"
    daily_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    for month in months:
        hourly = load_hourly(month)
        if hourly is None:
            continue
        nodes = load_nodes(month)

        rh = hourly["humidity_clip90"].values
        out_cols = []
        for name, spec in POLLUTANTS.items():
            p = params.get(name)
            if p is None:
                continue
            col = f"{name}_corrected"
            dry = hourly[spec["lowcost"]].values / growth_divisor(rh, p["kappa"])
            hourly[col] = p["a"] + p["b"] * dry
            out_cols.append(col)

        if not out_cols:
            continue

        raw_cols = [s["lowcost"] for s in POLLUTANTS.values()]
        daily = to_daily(hourly, out_cols + raw_cols + ["humidity_clip90"])
        daily = daily.merge(nodes, on="location", how="left")

        d = daily.copy()
        d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
        agg = {c: (c, "mean") for c in out_cols + raw_cols + ["humidity_clip90"]}
        agg["n_days"] = ("date", "nunique")
        monthly = d.groupby(["location", "month"]).agg(**agg).reset_index()
        monthly = monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH]
        monthly = monthly.merge(nodes, on="location", how="left")

        daily.to_csv(daily_dir / f"{month}.csv", index=False)
        monthly.to_csv(monthly_dir / f"{month}.csv", index=False)
        print(f"  [{held_out_land}][{month}] wrote {len(daily):,} corrected "
              f"node-days, {len(monthly):,} node-months")

    aggregate_annual(held_out_land, months, year)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true",
                    help="Refit even if params_by_land.json exists")
    ap.add_argument("--apply-land", default=None,
                    help="Also write the corrected dataset for this one held-out "
                         "Land (e.g. --apply-land Bayern)")
    args = ap.parse_args()

    months = args.months or discover_months()
    if not months:
        print(f"No hourly PM files found in {PM_HOURLY_DIR}.")
        return

    params_path = CALIB_DIR / "params_by_land.json"

    if params_path.exists() and not args.refit:
        results = json.loads(params_path.read_text())
        print(f"Using existing calibrations from {params_path} (--refit to redo)\n")
    else:
        results = fit_all_lands(months, args.year, args.seed)
        if not results:
            print("Nothing fitted for any Land.")
            return
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        params_path.write_text(json.dumps(results, indent=2))
        print(f"saved {len(results)} Land calibrations -> {params_path}")

    if args.apply_land:
        if args.apply_land not in results:
            print(f"\nNo calibration found for '{args.apply_land}'. "
                  f"Available: {sorted(results.keys())}")
            return
        print(f"\nApplying {args.apply_land} fold to the full dataset:")
        apply_land_fold(months, args.apply_land, results[args.apply_land], args.year)

    print("\nDone.")


if __name__ == "__main__":
    main()
