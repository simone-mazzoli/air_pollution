"""
Three independent questions, each answered directly against co-located UBA
reference stations (no model, no CV):

  Q1 PRECISION   -- within a tight cluster, how scattered are the sensors?
                    Small within-cluster SD => precise, usable after calibration.
                    Large SD => irreducible per-sensor noise.
  Q2 BIAS        -- is the sensor-minus-reference offset the SAME across clusters?
                    Consistent offset => one calibration number fixes it => usable.
                    Random per cluster => not calibratable.
  Q3 SPATIAL     -- do cluster values track reference across the FULL PM range?
                    Spearman (rank) correlation survives restricted range far
                    better than Pearson; plus a low-vs-high tercile separation
                    test: can the sensors tell clean Laender from dirty ones?

Reports on RAW and CALIBRATED sensor labels side by side (if a raw column is
given), so you can see whether the bias is calibration-induced.

Usage:
  python3 sensor_verdict.py --radius 1 --min-in-radius 3
  python3 sensor_verdict.py --radius 1 --raw-col P1_annual_mean
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _find_proc():
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        p = start
        for _ in range(6):
            if (p / "data" / "processed").is_dir():
                return p / "data" / "processed"
            p = p.parent
    raise SystemExit("could not locate data/processed")


PROC = _find_proc()
FOLD_DIR = PROC / "corrected" / "fold"
SENSOR_LAND = PROC / "sensor_land.csv"
UBA_DAILY = PROC / "daily_avg" / "uba" / "pm_reference_stations_{year}.csv"
STATION_LAND = PROC / "uba" / "station_land.csv"

TEST_LAND = "Sachsen-Anhalt"
CAL_COL = "PM10_corrected"
UBA_REF = "PM10"
UBA_MIN_DAYS = 182
LAT0 = 51.0
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(LAT0))


def to_km_xy(lat, lon):
    return np.stack([lon * KM_PER_DEG_LON, lat * KM_PER_DEG_LAT], axis=1)


def load_sensors(raw_col):
    cols_want = {"location", CAL_COL, "lat", "lon"}
    if raw_col:
        cols_want.add(raw_col)
    frames = []
    for f in FOLD_DIR.glob("*/annual/*.csv"):
        df = pd.read_csv(f)
        have = [c for c in cols_want if c in df.columns]
        if {"location", CAL_COL, "lat", "lon"} <= set(have):
            frames.append(df[have])
    s = pd.concat(frames, ignore_index=True).drop_duplicates("location")
    s = s.dropna(subset=[CAL_COL, "lat", "lon"])
    s = s[s[CAL_COL] > 0].reset_index(drop=True)
    return s


def load_uba(year=2024):
    df = pd.read_csv(Path(str(UBA_DAILY).format(year=year)))
    df[UBA_REF] = pd.to_numeric(df[UBA_REF], errors="coerce")
    df = df.dropna(subset=[UBA_REF])
    g = df.groupby("station_code")
    ann = g[UBA_REF].mean().rename("true").to_frame()
    ann["n_days"] = g[UBA_REF].size()
    ann[["lat", "lon"]] = g[["lat", "lon"]].median()
    ann = ann.reset_index()
    ann = ann[ann["n_days"] >= UBA_MIN_DAYS]
    sl = pd.read_csv(STATION_LAND)[["station_code", "land"]]
    ann = ann.merge(sl, on="station_code", how="inner")
    ann = ann[ann["land"] != TEST_LAND]
    return ann[ann["true"] > 0].reset_index(drop=True)


def _spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(label, sensors, val_col, uba, radius, min_in_radius):
    sxy = to_km_xy(sensors["lat"].values, sensors["lon"].values)
    sval = sensors[val_col].values.astype("float64")
    rows = []
    for _, st in uba.iterrows():
        stxy = to_km_xy(np.array([st["lat"]]), np.array([st["lon"]]))[0]
        d = np.sqrt(((sxy - stxy) ** 2).sum(1))
        idx = np.where(d <= radius)[0]
        if len(idx) < min_in_radius:
            continue
        vals = sval[idx]
        rows.append({"true": st["true"], "sensor_mean": vals.mean(),
                     "sensor_median": float(np.median(vals)),
                     "sensor_sd": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                     "n": len(idx)})
    c = pd.DataFrame(rows)
    if len(c) < 10:
        print(f"\n[{label}] only {len(c)} co-located clusters "
              f"(need >=10) -- widen --radius or lower --min-in-radius")
        return None

    offset = c["sensor_mean"] - c["true"]      # sensor minus reference
    within_sd = c["sensor_sd"].mean()
    ref_spread = c["true"].std()
    off_mean, off_sd = offset.mean(), offset.std()
    sp = _spearman(c["sensor_mean"].values, c["true"].values)

    # after removing the CONSTANT bias, how well would sensors do?
    corrected = c["sensor_mean"] - off_mean
    rmse_raw = float(np.sqrt(np.mean((c["sensor_mean"] - c["true"]) ** 2)))
    rmse_debias = float(np.sqrt(np.mean((corrected - c["true"]) ** 2)))
    base = float(np.sqrt(np.mean((c["true"] - c["true"].mean()) ** 2)))

    # tercile test: can sensors separate cleanest third of Laender from dirtiest?
    q1, q2 = c["true"].quantile([1/3, 2/3])
    lo = c[c["true"] <= q1]["sensor_mean"].mean()
    hi = c[c["true"] >= q2]["sensor_mean"].mean()

    print(f"\n===== [{label}] {len(c)} co-located clusters "
          f"(>={min_in_radius} sensors within {radius:g} km) =====")
    print(f"Q1 PRECISION")
    print(f"   mean within-cluster sensor SD : {within_sd:.2f} ug/m3")
    print(f"   (compare to spread of true values across clusters: {ref_spread:.2f})")
    verdict_prec = ("PRECISE (sensors in a cluster agree)"
                    if within_sd < ref_spread else
                    "NOISY (within-cluster scatter >= between-site signal)")
    print(f"   -> {verdict_prec}")
    print(f"Q2 BIAS")
    print(f"   sensor - reference offset      : mean {off_mean:+.2f}, sd {off_sd:.2f} ug/m3")
    verdict_bias = ("CONSISTENT (one calibration constant fixes it)"
                    if off_sd < abs(off_mean) or off_sd < 1.0 else
                    "INCONSISTENT (offset varies by site -> not a single constant)")
    print(f"   -> {verdict_bias}")
    print(f"   RMSE vs reference: raw {rmse_raw:.2f} | after removing mean bias "
          f"{rmse_debias:.2f} | baseline(mean) {base:.2f}")
    print(f"Q3 SPATIAL SIGNAL")
    print(f"   Spearman rank corr (sensor vs true): {sp:+.3f}")
    print(f"   cleanest-third sites: sensors read {lo:.1f} | "
          f"dirtiest-third: {hi:.1f}  (gap {hi-lo:+.1f})")
    verdict_spatial = ("TRACKS reference ranking"
                       if (sp > 0.3 and hi - lo > 0) else
                       "does NOT track reference ranking")
    print(f"   -> {verdict_spatial}")
    return {"label": label, "within_sd": within_sd, "ref_spread": ref_spread,
            "offset_mean": off_mean, "offset_sd": off_sd, "spearman": sp,
            "rmse_debias": rmse_debias, "baseline": base, "tercile_gap": hi - lo,
            "n_clusters": len(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.0,
                    help="co-location radius in km (default 1 = same air)")
    ap.add_argument("--min-in-radius", type=int, default=3)
    ap.add_argument("--raw-col", default=None,
                    help="raw (uncalibrated) sensor annual column, to see if the "
                         "bias is calibration-induced")
    args = ap.parse_args()

    sensors = load_sensors(args.raw_col)
    uba = load_uba()
    print(f"{len(sensors)} sensors, {len(uba)} UBA stations, "
          f"co-location radius {args.radius:g} km")

    res = []
    r = analyze("CALIBRATED", sensors, CAL_COL, uba, args.radius, args.min_in_radius)
    if r: res.append(r)
    if args.raw_col and args.raw_col in sensors.columns:
        r2 = analyze(f"RAW ({args.raw_col})", sensors, args.raw_col, uba,
                     args.radius, args.min_in_radius)
        if r2: res.append(r2)
    elif args.raw_col:
        print(f"\n(raw column '{args.raw_col}' not found in fold CSVs; "
              f"columns available include: {list(sensors.columns)})")

    print("\n" + "=" * 62)
    print("BOTTOM LINE")
    if not res:
        print("  No usable clusters -- widen --radius / lower --min-in-radius.")
        return
    r = res[0]
    precise = r["within_sd"] < r["ref_spread"]
    calibratable = r["offset_sd"] < 1.0 or r["offset_sd"] < abs(r["offset_mean"])
    tracks = r["spearman"] > 0.3 and r["tercile_gap"] > 0
    debias_helps = r["rmse_debias"] < r["baseline"]

    if tracks and debias_helps:
        print("  USABLE. After removing a constant bias the sensors beat the mean")
        print("  and track reference ranking -> keep them (calibrated + aggregated).")
    elif precise and calibratable and not tracks:
        print("  PRECISE & CALIBRATABLE but they DON'T track spatial variation:")
        print("  usable as a coverage/regularization signal for a model, NOT as a")
        print("  stand-in for reference values. (Matches the CV: helps on sparse")
        print("  folds via coverage, doesn't reproduce reference truth.)")
    elif not precise:
        print("  TOO NOISY at the single-sensor level; only heavy aggregation helps,")
        print("  and even then they don't track reference -> weak signal. Keep only")
        print("  if a model shows CV skill gain; otherwise drop.")
    else:
        print("  MARGINAL. No clear spatial signal vs reference; justify keeping them")
        print("  ONLY by the held-out-Land CV skill number, not by this test.")
    if len(res) == 2:
        a, b = res[0], res[1]
        print(f"\n  calibration check: offset {b['label']}={b['offset_mean']:+.2f} -> "
              f"{a['label']}={a['offset_mean']:+.2f}")
        if abs(a["offset_mean"]) > abs(b["offset_mean"]) + 0.3:
            print("  -> calibration ADDED bias (raw was closer to reference). "
                  "Check the calibration step.")
        elif abs(a["offset_mean"]) + 0.3 < abs(b["offset_mean"]):
            print("  -> calibration REDUCED bias, as intended.")
        else:
            print("  -> calibration left the bias roughly unchanged.")
    print("=" * 62)


if __name__ == "__main__":
    main()
