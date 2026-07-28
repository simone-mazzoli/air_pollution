"""
Sanity-check newly received hourly data before running calibrate_pm_leave_one_fold_out.py.

Checks, per month found in data/processed/hourly/pm/all_pm_sensors/:
  1. All four required files exist (pm/all_pm_sensors, pm/nodes, humidity/all_sensors,
     humidity/nodes) -- a month missing one silently gets skipped by load_hourly(),
     which is easy to not notice.
  2. Column schema matches what the pipeline expects.
  3. No duplicate/overlapping months from different teammates (same month processed
     twice, possibly with different data -- second one silently wins on load).
  4. Value sanity: P1/P2 in a plausible range, humidity in [0, 100], no all-NaN columns.
  5. Node/location coverage is the same order of magnitude as your own months (catches
     a teammate accidentally processing a tiny subset).

Run this BEFORE calibrate_pm_leave_one_fold_out.py -- that script now refits 12x
instead of once, so catching a bad month here is much cheaper than after a long run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROC = BASE_DIR / "data" / "processed"

PM_HOURLY_DIR = PROC / "hourly" / "pm" / "all_pm_sensors"
PM_NODES_DIR = PROC / "hourly" / "pm" / "nodes"
RH_HOURLY_DIR = PROC / "hourly" / "humidity" / "all_sensors"
RH_NODES_DIR = PROC / "hourly" / "humidity" / "nodes"

EXPECTED_PM_COLS = {"location", "hour", "P1", "P2"}
EXPECTED_RH_COLS = {"location", "hour", "humidity", "humidity_clip90"}

# plausible ranges -- outside these, something's probably wrong upstream, not a real reading
P_MIN, P_MAX = 0, 1000       # ug/m3, generous upper bound for PM10/PM2.5
RH_MIN, RH_MAX = 0, 100      # percent


def discover_months():
    return sorted(p.stem for p in PM_HOURLY_DIR.glob("*.parquet"))


def check_month(month):
    problems = []
    pm_path = PM_HOURLY_DIR / f"{month}.parquet"
    rh_path = RH_HOURLY_DIR / f"{month}.parquet"
    pm_nodes = list(PM_NODES_DIR.glob(f"*_{month}.parquet"))
    rh_nodes = list(RH_NODES_DIR.glob(f"*_{month}.parquet"))

    if not pm_path.exists():
        problems.append("MISSING pm/all_pm_sensors file")
    if not rh_path.exists():
        problems.append("MISSING humidity/all_sensors file")
    if not pm_nodes:
        problems.append("MISSING pm/nodes file(s)")
    if not rh_nodes:
        problems.append("MISSING humidity/nodes file(s)")
    if problems:
        return problems, {}

    pm = pd.read_parquet(pm_path)
    rh = pd.read_parquet(rh_path)

    stats = {
        "pm_rows": len(pm),
        "pm_locations": pm["location"].nunique() if "location" in pm.columns else None,
        "rh_rows": len(rh),
        "rh_locations": rh["location"].nunique() if "location" in rh.columns else None,
    }

    missing_pm_cols = EXPECTED_PM_COLS - set(pm.columns)
    missing_rh_cols = EXPECTED_RH_COLS - set(rh.columns)
    if missing_pm_cols:
        problems.append(f"pm file missing columns: {missing_pm_cols} "
                        f"(actual: {list(pm.columns)})")
    if missing_rh_cols:
        problems.append(f"humidity file missing columns: {missing_rh_cols} "
                        f"(actual: {list(rh.columns)})")
    if problems:
        return problems, stats

    for col in ("P1", "P2"):
        vals = pd.to_numeric(pm[col], errors="coerce")
        if vals.isna().all():
            problems.append(f"{col} is all-NaN")
            continue
        out_of_range = ((vals < P_MIN) | (vals > P_MAX)).sum()
        if out_of_range > 0:
            pct = 100.0 * out_of_range / len(vals)
            problems.append(f"{col}: {out_of_range:,} values ({pct:.1f}%) outside "
                            f"[{P_MIN}, {P_MAX}] -- check units/corruption")
        stats[f"{col}_median"] = float(vals.median())

    rh_vals = pd.to_numeric(rh["humidity"], errors="coerce")
    if rh_vals.isna().all():
        problems.append("humidity is all-NaN")
    else:
        out_of_range = ((rh_vals < RH_MIN) | (rh_vals > RH_MAX)).sum()
        if out_of_range > 0:
            pct = 100.0 * out_of_range / len(rh_vals)
            problems.append(f"humidity: {out_of_range:,} values ({pct:.1f}%) outside "
                            f"[{RH_MIN}, {RH_MAX}]")
        stats["humidity_median"] = float(rh_vals.median())

    if "hour" in pm.columns:
        if not pd.api.types.is_datetime64_any_dtype(pm["hour"]):
            problems.append(f"pm 'hour' column is not datetime (dtype: {pm['hour'].dtype})")
        else:
            month_str = pm["hour"].dt.strftime("%Y-%m")
            wrong_month = (month_str != month).sum()
            if wrong_month > 0:
                problems.append(f"{wrong_month:,} rows have 'hour' outside the "
                                f"{month} filename -- possible mislabeled file")

    return problems, stats


def main():
    months = discover_months()
    if not months:
        print(f"No month files found in {PM_HOURLY_DIR}.")
        return

    print(f"Found {len(months)} months: {months}\n")

    print("IMPORTANT: if two teammates each sent a file for the same month, only "
          "the LAST one you copied in survives -- there's no way to detect that "
          "from the files alone. Confirm with your team which months each of you "
          "processed and check there's no overlap before trusting row counts below.\n")

    all_stats = {}
    any_problems = False
    for month in months:
        problems, stats = check_month(month)
        all_stats[month] = stats
        if problems:
            any_problems = True
            print(f"[{month}] PROBLEMS FOUND:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[{month}] OK -- {stats['pm_locations']:,} PM locations, "
                  f"{stats['rh_locations']:,} humidity locations, "
                  f"P1 median={stats.get('P1_median', 'n/a'):.1f}, "
                  f"P2 median={stats.get('P2_median', 'n/a'):.1f}, "
                  f"humidity median={stats.get('humidity_median', 'n/a'):.1f}")

    loc_counts = {m: s["pm_locations"] for m, s in all_stats.items()
                  if s.get("pm_locations")}
    if loc_counts:
        med = np.median(list(loc_counts.values()))
        print(f"\nMedian PM locations per month: {med:.0f}")
        for m, c in loc_counts.items():
            if c < med * 0.5 or c > med * 1.5:
                print(f"  WARNING: {m} has {c:,} locations -- far from the "
                      f"median, worth double-checking this file")
                any_problems = True

    print("\n" + ("Some issues found above -- review before running the "
                  "12-fold calibration." if any_problems else
                  "All checks passed. Safe to proceed to "
                  "calibrate_pm_leave_one_fold_out.py."))


if __name__ == "__main__":
    main()
