"""
Calibrate low-cost PM against UBA reference stations, leave-one-fold-out (no leakage).

Global linear calibration per pollutant per fold: fit ref = a + b*raw by least squares
over all colocated sensor-days on the OTHER 11 folds, then apply a + b*raw to every
sensor. One (a,b) shared by all sensors, so sensors stay independent measurements (not
pinned to individual stations). Refit 12 times, each excluding the held-out fold's own
stations. Quality is reported on the held-out fold's stations (never seen by the fit):
annual-mean RMSE of raw vs calibrated sensors against those references.

Humidity note: low-cost PM over-reads at high RH (hygroscopic growth) and averaging
doesn't remove it; on-node RH sensors saturate near 100%, so it's a documented
limitation, not corrected here.

16 Laender -> 12 folds. Output: corrected/fold/<fold>/annual/<year>.csv (CNN target).
"""

import argparse, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
PM_HOURLY_DIR = PROC / "hourly" / "pm" / "all_pm_sensors"
NODES_DIR = PROC / "hourly" / "pm" / "nodes"
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"
STATION_LAND_PATH = PROC / "uba" / "station_land.csv"
CALIB_DIR = PROC / "calibration"
CORR_FOLD_DIR = PROC / "corrected" / "fold"

UTC_TO_MEZ_HOURS = 1
RADIUS_KM = 20.0
MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_MONTH = 10
MIN_DAYS_PER_YEAR = 182
MIN_COLOCATED_DAYS = 200

POLLUTANTS = {"PM10": {"lowcost": "P1", "ref": "PM10"},
              "PM2.5": {"lowcost": "P2", "ref": "PM2.5"}}
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

LAND_TO_FOLD = {
    "Berlin": "Berlin-Brandenburg", "Brandenburg": "Berlin-Brandenburg",
    "Bremen": "Bremen-Niedersachsen", "Niedersachsen": "Bremen-Niedersachsen",
    "Hamburg": "Hamburg-Schleswig-Holstein", "Schleswig-Holstein": "Hamburg-Schleswig-Holstein",
    "Saarland": "Saarland-Rheinland-Pfalz", "Rheinland-Pfalz": "Saarland-Rheinland-Pfalz",
    "Baden-Wuerttemberg": "Baden-Wuerttemberg", "Bayern": "Bayern", "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen", "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt", "Thueringen": "Thueringen",
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0; p = np.radians
    dp = p(lat2) - p(lat1); dl = p(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p(lat1))*np.cos(p(lat2))*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def discover_months():
    if not PM_HOURLY_DIR.exists(): return []
    return sorted(p.stem for p in PM_HOURLY_DIR.glob("*.parquet") if MONTH_RE.match(p.stem))


def load_nodes(month):
    parts = [pd.read_parquet(p) for p in NODES_DIR.glob(f"*_{month}.parquet")]
    n = pd.concat(parts, ignore_index=True)
    return n.drop_duplicates("location")[["location", "lat", "lon"]]


def load_daily(month):
    p = PM_HOURLY_DIR / f"{month}.parquet"
    if not p.exists(): return None
    h = pd.read_parquet(p)[["location", "hour", "P1", "P2"]]
    h["date"] = (h["hour"] + pd.Timedelta(hours=UTC_TO_MEZ_HOURS)).dt.date
    agg = {s["lowcost"]: (s["lowcost"], "mean") for s in POLLUTANTS.values()}
    agg["n_hours"] = ("hour", "nunique")
    d = h.groupby(["location", "date"]).agg(**agg).reset_index()
    return d[d["n_hours"] >= MIN_HOURS_PER_DAY].copy()


def load_uba(year):
    df = pd.read_csv(Path(str(UBA_DAILY).format(year=year)))
    df["date"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for c in ("PM10", "PM2.5"): df[c] = pd.to_numeric(df[c], errors="coerce")
    sl = pd.read_csv(STATION_LAND_PATH)[["station_code", "land"]]
    df = df.merge(sl, on="station_code", how="left")
    df["fold"] = df["land"].map(LAND_TO_FOLD)
    return df.dropna(subset=["fold"])[["station_code", "lat", "lon", "date", "PM10", "PM2.5", "fold"]]


def colocate(nodes, uba_subset):
    st = uba_subset.drop_duplicates("station_code")[["station_code", "lat", "lon"]]
    if st.empty: return pd.DataFrame(columns=["location", "station_code", "dist_km"])
    D = haversine_km(nodes["lat"].values[:, None], nodes["lon"].values[:, None],
                     st["lat"].values[None, :], st["lon"].values[None, :])
    bi = D.argmin(1)
    out = pd.DataFrame({"location": nodes["location"].values,
                        "station_code": st["station_code"].values[bi],
                        "dist_km": D[np.arange(len(nodes)), bi]})
    return out[out["dist_km"] <= RADIUS_KM].reset_index(drop=True)


def _pairs(daily, links, uba, lowcost_col, ref_col):
    ref = uba[["station_code", "date", ref_col]].rename(columns={ref_col: "ref_val"})
    m = (daily[["location", "date", lowcost_col]]
         .merge(links[["location", "station_code"]], on="location")
         .merge(ref, on=["station_code", "date"]).dropna())
    return m[m[lowcost_col] > 0]


def fit_linear(daily, links, uba_train, lowcost_col, ref_col):
    """Least-squares ref = a + b*raw over colocated sensor-days. Robust: drops the top
    1% of raw values first so humidity-inflated spike days don't tilt the line."""
    m = _pairs(daily, links, uba_train, lowcost_col, ref_col)
    if len(m) < MIN_COLOCATED_DAYS: return None
    x, y = m[lowcost_col].values, m["ref_val"].values
    keep = x <= np.quantile(x, 0.99)
    x, y = x[keep], y[keep]
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def heldout_annual_rmse(daily, nodes, uba_ho, a, b, lowcost_col, ref_col):
    """Annual-mean RMSE raw vs calibrated on HELD-OUT stations (the graded thing)."""
    links = colocate(nodes, uba_ho)
    m = _pairs(daily, links, uba_ho, lowcost_col, ref_col)
    if len(m) < 30: return None
    # collapse to one annual mean per sensor and its station's annual mean
    g = m.groupby("location").agg(raw=(lowcost_col, "mean"),
                                  ref=("ref_val", "mean")).reset_index()
    raw, ref = g["raw"].values, g["ref"].values
    cal = a + b*raw
    rr = float(np.sqrt(np.mean((raw - ref)**2)))
    rc = float(np.sqrt(np.mean((cal - ref)**2)))
    return rr, rc, int(len(g))


def apply_fold(months, fold, coeffs, year):
    root = CORR_FOLD_DIR / fold.replace(" ", "_")
    for s in ("daily", "monthly", "annual"): (root/s).mkdir(parents=True, exist_ok=True)
    all_daily = []
    for month in months:
        d = load_daily(month)
        if d is None: continue
        nodes = load_nodes(month)
        for name, spec in POLLUTANTS.items():
            if name in coeffs:
                a, b = coeffs[name]
                d[f"{name}_corrected"] = a + b * d[spec["lowcost"]].values
                d = d.rename(columns={spec["lowcost"]: f"{name}_raw"})
        d = d.drop(columns=["n_hours"]).merge(nodes, on="location", how="left")
        all_daily.append(d)
        d.to_csv(root/"daily"/f"{month}.csv", index=False)
        d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
        vcols = [c for c in d.columns if c.endswith("_corrected") or c.endswith("_raw")]
        agg = {c: (c, "mean") for c in vcols}; agg["n_days"] = ("date", "nunique")
        mo = d.groupby("location").agg(**agg).reset_index()
        mo = mo[mo["n_days"] >= MIN_DAYS_PER_MONTH].merge(nodes, on="location", how="left")
        mo.to_csv(root/"monthly"/f"{month}.csv", index=False)
    yr = pd.concat(all_daily, ignore_index=True)
    vcols = [c for c in yr.columns if c.endswith("_corrected") or c.endswith("_raw")]
    g = yr.groupby("location")
    annual = g[vcols].mean().reset_index()
    annual["n_days_total"] = g["date"].nunique().values
    annual = annual.merge(yr.drop_duplicates("location")[["location","lat","lon"]], on="location")
    annual = annual[annual["n_days_total"] >= MIN_DAYS_PER_YEAR]
    annual.to_csv(root/"annual"/f"{year}.csv", index=False)
    return len(annual)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--apply-fold", default=None)
    args = ap.parse_args()

    months = args.months or discover_months()
    if not months:
        print(f"No hourly PM files in {PM_HOURLY_DIR}."); return

    daily = pd.concat([load_daily(m) for m in months if load_daily(m) is not None], ignore_index=True)
    nodes = pd.concat([load_nodes(m) for m in months], ignore_index=True).drop_duplicates("location")
    uba = load_uba(args.year)
    folds = sorted(uba["fold"].unique())

    cp = CALIB_DIR / "linear_by_fold.json"
    if cp.exists() and not args.refit:
        by_fold = json.loads(cp.read_text())
        print(f"Using existing calibration ({cp.name})")
    else:
        print(f"{len(folds)} folds | fit on OTHER folds, annual RMSE checked on HELD-OUT fold:\n")
        print(f"  {'fold':<28} {'poll':<6} {'a':>7} {'b':>6}  {'RMSE raw':>8} {'RMSE cal':>8}  {'sensors':>7}")
        by_fold = {}
        for f in folds:
            uba_tr = uba[uba["fold"] != f]
            uba_ho = uba[uba["fold"] == f]
            links_tr = colocate(nodes, uba_tr)
            fc = {}
            for name, spec in POLLUTANTS.items():
                ab = fit_linear(daily, links_tr, uba_tr, spec["lowcost"], spec["ref"])
                if ab is None: continue
                a, b = ab; fc[name] = [a, b]
                chk = heldout_annual_rmse(daily, nodes, uba_ho, a, b, spec["lowcost"], spec["ref"])
                if chk:
                    rr, rc, n = chk
                    flag = "" if rc < rr else "  (worse)"
                    print(f"  {f:<28} {name:<6} {a:>7.2f} {b:>6.3f}  {rr:>8.2f} {rc:>8.2f}  {n:>7}{flag}")
                else:
                    print(f"  {f:<28} {name:<6} {a:>7.2f} {b:>6.3f}  {'--':>8} {'--':>8}  {'few':>7}")
            by_fold[f] = fc
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(by_fold, indent=2))
        print(f"\nsaved {cp}")

    print("\nWriting corrected datasets:")
    for f in ([args.apply_fold] if args.apply_fold else folds):
        if f in by_fold:
            n = apply_fold(months, f, by_fold[f], args.year)
            print(f"  {f:<28} annual: {n:,} sensors")


if __name__ == "__main__":
    main()
