"""
Corrects low-cost PM sensors for humidity bias and calibrates them against UBA reference
stations.

  PM_dry = PM_raw / (1 + kappa / (-1 + 100/RH))        applied hourly
  PM_cal = a + b * PM_dry                              linear residual calibration

kappa and (a, b) are fitted jointly against UBA daily means at colocated stations:
for each candidate kappa the linear fit is redone, so kappa is only responsible for the
RH-dependent curvature and b absorbs the overall scale offset. kappa=0 is in the grid,
so "linear calibration only, no humidity correction" is one of the candidates.

Stations are split into fit/holdout so the reported improvement is out-of-sample.

Input:  data/processed/hourly/pm/all_pm_sensors/<YYYY-MM>.parquet
        data/processed/hourly/humidity/all_sensors/<YYYY-MM>.parquet
        data/processed/hourly/pm/nodes/<type>_<YYYY-MM>.parquet
        data/processed/daily_avg/uba/pm_reference_stations_<YEAR>.csv
Output: data/processed/calibration/params.json
        data/processed/corrected/daily/<YYYY-MM>.csv
        data/processed/corrected/monthly/<YYYY-MM>.csv
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
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"

CALIB_DIR = PROC / "calibration"
CORR_DAILY_DIR = PROC / "corrected" / "daily"
CORR_MONTHLY_DIR = PROC / "corrected" / "monthly"

UTC_TO_MEZ_HOURS = 1  # Sensor.Community = UTC, UBA = MEZ (fixed UTC+1, no DST)

RADIUS_KM = 5.0  # tight on purpose, PM varies over short distances
MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_MONTH = 10
HOLDOUT_FRAC = 0.25
KAPPA_GRID = np.round(np.arange(0.0, 1.21, 0.02), 3)

MIN_DAYS_FOR_PER_STATION_R = 10  # min days before trusting a per-node r

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
    return 1.0 + kappa / (-1.0 + 100.0 / rh)  # rh clipped at 90 upstream, denom >= 0.111


def discover_months():
    if not PM_HOURLY_DIR.exists():
        return []
    return sorted(p.stem for p in PM_HOURLY_DIR.glob("*.parquet")
                  if MONTH_RE.match(p.stem))


def load_nodes(month):
    parts = [pd.read_parquet(p) for p in NODES_DIR.glob(f"*_{month}.parquet")]
    if not parts:
        raise FileNotFoundError(f"no node coordinate files for {month} in {NODES_DIR}")
    nodes = pd.concat(parts, ignore_index=True)
    return nodes.drop_duplicates(subset="location", keep="first")[["location", "lat", "lon"]]


def load_hourly(month):
    """PM joined to on-node RH, one row per (location, hour)."""
    pm_path = PM_HOURLY_DIR / f"{month}.parquet"
    rh_path = RH_HOURLY_DIR / f"{month}.parquet"
    if not pm_path.exists() or not rh_path.exists():
        return None

    pm = pd.read_parquet(pm_path)[["location", "hour", "P1", "P2"]]
    rh = pd.read_parquet(rh_path)[["location", "hour", "humidity", "humidity_clip90"]]

    n_pm = len(pm)
    df = pm.merge(rh, on=["location", "hour"], how="inner")

    yield_pct = 100.0 * len(df) / max(n_pm, 1)
    node_yield = 100.0 * df.location.nunique() / max(pm.location.nunique(), 1)
    print(f"  [{month}] RH join: {len(df):,}/{n_pm:,} hours ({yield_pct:.1f}%), "
          f"{df.location.nunique()}/{pm.location.nunique()} nodes ({node_yield:.1f}%)")

    # unmatched PM hours dropped, not backfilled -- would reintroduce averaging bias
    df["date"] = (df["hour"] + pd.Timedelta(hours=UTC_TO_MEZ_HOURS)).dt.date
    return df


def to_daily(hourly, value_cols):
    agg = {c: (c, "mean") for c in value_cols}
    agg["n_hours"] = ("hour", "nunique")
    daily = hourly.groupby(["location", "date"]).agg(**agg).reset_index()
    return daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()


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


def colocate(nodes, uba):
    """Nearest UBA station within RADIUS_KM for each node."""
    stations = uba.drop_duplicates("station_code")[["station_code", "lat", "lon"]]

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


def per_station_correlation_diagnostic(df, lowcost_col, ref_col, dry_col=None):
    """
    Computes r per individual sensor instead of per reference station (many sensors
    share one station) to separate real signal from between-sensor offsets. Also
    checks whether low per-node r tracks n_days/dist_km, and compares raw-vs-dry r
    when dry_col is given.
    """
    def node_stats(g):
        if len(g) < MIN_DAYS_FOR_PER_STATION_R:
            base = {"r_raw": np.nan, "n_days": len(g),
                    "dist_km": g["dist_km"].iloc[0],
                    "station_code": g["station_code"].iloc[0]}
        else:
            base = {"r_raw": g[lowcost_col].corr(g[ref_col]),
                    "n_days": len(g),
                    "dist_km": g["dist_km"].iloc[0],
                    "station_code": g["station_code"].iloc[0]}
        if dry_col is not None:
            base["r_dry"] = (g[dry_col].corr(g[ref_col])
                             if len(g) >= MIN_DAYS_FOR_PER_STATION_R else np.nan)
        return pd.Series(base)

    per_node = df.groupby("location").apply(node_stats)
    valid = per_node.dropna(subset=["r_raw"])

    if valid.empty:
        print(f"    {ref_col}: no node has >= {MIN_DAYS_FOR_PER_STATION_R} days "
              f"-- cannot compute per-node r")
        return

    print(f"    {ref_col}: per-node r (RAW, before humidity correction) -- "
          f"median={valid['r_raw'].median():.2f}, "
          f"25th/75th pct=({valid['r_raw'].quantile(.25):.2f}/{valid['r_raw'].quantile(.75):.2f}), "
          f"n_nodes={len(valid)}, "
          f"pct with r>0.3: {(valid['r_raw'] > 0.3).mean() * 100:.0f}%")

    if dry_col is not None:
        print(f"    {ref_col}: per-node r (DRY, after best-fit humidity correction) -- "
              f"median={valid['r_dry'].median():.2f}, "
              f"25th/75th pct=({valid['r_dry'].quantile(.25):.2f}/{valid['r_dry'].quantile(.75):.2f}), "
              f"pct with r>0.3: {(valid['r_dry'] > 0.3).mean() * 100:.0f}%")
        delta = (valid["r_dry"] - valid["r_raw"]).median()
        print(f"    {ref_col}: median per-node CHANGE from humidity correction: "
              f"{delta:+.3f}  "
              f"({'helps' if delta > 0.01 else 'hurts' if delta < -0.01 else 'no real effect'})")

        pooled_r_raw = df[lowcost_col].corr(df[ref_col])
        pooled_r_dry = df[dry_col].corr(df[ref_col])
        print(f"    {ref_col}: POOLED r raw={pooled_r_raw:.3f}  "
              f"POOLED r dry={pooled_r_dry:.3f}  "
              f"(if these two are close despite the humidity correction being applied, "
              f"the pooled number is dominated by between-sensor offsets, not humidity)")

    r_vs_ndays = valid["r_raw"].corr(valid["n_days"])
    r_vs_dist = valid["r_raw"].corr(valid["dist_km"])
    print(f"    {ref_col}: corr(node_r, n_days)={r_vs_ndays:.2f}  "
          f"corr(node_r, dist_km)={r_vs_dist:.2f}")
    print(f"      (near 0 for either => low r there isn't explained by "
          f"sample size / colocation distance -- worth checking those "
          f"nodes individually)")

    nodes_per_station = valid.groupby("station_code").size().sort_values(ascending=False)
    print(f"    {ref_col}: busiest reference stations by # colocated nodes: "
          f"{nodes_per_station.head(5).to_dict()}")

    worst = valid.sort_values("r_raw").head(10)
    print(f"    {ref_col}: 10 worst nodes by RAW r "
          f"(location, station_code, r_raw, n_days, dist_km):")
    for location, row in worst.iterrows():
        print(f"      {location} -> {row['station_code']}: r_raw={row['r_raw']:.2f}  "
              f"n_days={int(row['n_days'])}  dist_km={row['dist_km']:.2f}")


def fit_pollutant(pairs, lowcost_col, ref_col, rng):
    """Grid search kappa with the linear fit nested inside. Returns params + diagnostics."""
    df = pairs.dropna(subset=[lowcost_col, ref_col, "humidity_clip90"])
    if df["station_code"].nunique() < 4:
        print(f"    {ref_col}: only {df['station_code'].nunique()} colocated station(s) "
              f"-- not enough to fit, skipping")
        return None

    stations = np.sort(df["station_code"].unique())
    holdout = set(rng.choice(stations, max(1, int(len(stations) * HOLDOUT_FRAC)),
                             replace=False))
    fit_mask = ~df["station_code"].isin(holdout)
    fit_df, hold_df = df[fit_mask], df[~fit_mask]

    best = None
    for kappa in KAPPA_GRID:
        x = fit_df[f"{lowcost_col}_dry_{kappa}"].values  # precomputed per kappa upstream
        y = fit_df[ref_col].values
        a, b = ols(x, y)
        rmse = float(np.sqrt(np.mean((a + b * x - y) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"kappa": float(kappa), "a": float(a), "b": float(b), "rmse": rmse}

    k = best["kappa"]

    per_station_correlation_diagnostic(df, lowcost_col, ref_col,
                                        dry_col=f"{lowcost_col}_dry_{k}")

    xh = hold_df[f"{lowcost_col}_dry_{k}"].values
    yh = hold_df[ref_col].values
    raw = hold_df[lowcost_col].values

    xf = fit_df[f"{lowcost_col}_dry_{k}"].values
    yf = fit_df[ref_col].values
    r = float(np.corrcoef(xf, yf)[0, 1]) if len(xf) > 2 else float("nan")

    if not (0.1 < best["b"] < 3.0) or not np.isfinite(r) or abs(r) < 0.3:
        print(f"    !! {ref_col}: implausible fit (b={best['b']:.3f}, r={r:.2f}). "
              f"Too few stations, bad colocation, or no real correlation. "
              f"DO NOT apply these parameters.")

    best.update({
        "r": r,
        "n_pairs": int(len(df)),
        "n_stations": int(len(stations)),
        "n_holdout_stations": int(len(holdout)),
        "holdout_rmse_uncorrected": float(np.sqrt(np.mean((raw - yh) ** 2))),
        "holdout_rmse_corrected": float(np.sqrt(np.mean((best["a"] + best["b"] * xh - yh) ** 2))),
        "holdout_bias_uncorrected": float(np.mean(raw - yh)),
        "holdout_bias_corrected": float(np.mean(best["a"] + best["b"] * xh - yh)),
    })
    return best


def build_fit_pairs(months, uba, year):
    """Colocated daily pairs, with a PM_dry column precomputed for every kappa."""
    parts = []
    for month in months:
        hourly = load_hourly(month)
        if hourly is None:
            continue
        nodes = load_nodes(month)
        links = colocate(nodes, uba)
        if links.empty:
            print(f"  [{month}] no nodes within {RADIUS_KM} km of a UBA station")
            continue

        h = hourly[hourly["location"].isin(links["location"])].copy()
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

        print(f"  [{month}] {len(daily):,} colocated node-days across "
              f"{daily['station_code'].nunique()} station(s)")

    if not parts:
        return None

    pairs = pd.concat(parts, ignore_index=True)
    return pairs.merge(uba, on=["station_code", "date"], how="inner")


def apply_month(month, params):
    hourly = load_hourly(month)
    if hourly is None:
        return
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
        print(f"  [{month}] no fitted parameters -- nothing to apply")
        return

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

    CORR_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    CORR_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(CORR_DAILY_DIR / f"{month}.csv", index=False)
    monthly.to_csv(CORR_MONTHLY_DIR / f"{month}.csv", index=False)
    print(f"  [{month}] wrote {len(daily):,} corrected node-days, "
          f"{len(monthly):,} node-months")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true",
                    help="Refit even if params.json exists")
    args = ap.parse_args()

    months = args.months or discover_months()
    if not months:
        print(f"No hourly PM files found in {PM_HOURLY_DIR}.")
        return

    params_path = CALIB_DIR / "params.json"

    if params_path.exists() and not args.refit:
        params = json.loads(params_path.read_text())["params"]
        print(f"Using existing calibration from {params_path} (--refit to redo)\n")
    else:
        print("Fitting calibration:")
        uba = load_uba(args.year)
        pairs = build_fit_pairs(months, uba, args.year)
        if pairs is None or pairs.empty:
            print("\nNo colocated pairs -- cannot fit. "
                  f"Try increasing RADIUS_KM (currently {RADIUS_KM}).")
            return

        rng = np.random.default_rng(args.seed)
        params = {}
        print()
        for name, spec in POLLUTANTS.items():
            res = fit_pollutant(pairs, spec["lowcost"], spec["ref"], rng)
            if res is None:
                continue
            params[name] = res
            print(f"    {name}: kappa={res['kappa']:.2f}  "
                  f"a={res['a']:.2f}  b={res['b']:.3f}  "
                  f"({res['n_pairs']:,} pairs, {res['n_stations']} stations)")
            print(f"      holdout RMSE  {res['holdout_rmse_uncorrected']:.2f} -> "
                  f"{res['holdout_rmse_corrected']:.2f} ug/m3   "
                  f"bias {res['holdout_bias_uncorrected']:+.2f} -> "
                  f"{res['holdout_bias_corrected']:+.2f}")

        if not params:
            print("\nNothing fitted.")
            return

        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        params_path.write_text(json.dumps(
            {"radius_km": RADIUS_KM, "holdout_frac": HOLDOUT_FRAC,
             "seed": args.seed, "params": params}, indent=2))
        print(f"\n  saved {params_path}")

    print("\nApplying correction:")
    for month in months:
        apply_month(month, params)

    print("\nDone.")


if __name__ == "__main__":
    main()
