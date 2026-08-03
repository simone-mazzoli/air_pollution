"""
Build the EU-wide annual PM label file from the PM airbase raw parquet, matching the
schema the pipeline expects:
Output:
    data/processed/daily_avg/eea/pm_reference_stations_2024.csv
    columns: station_code, station_name, lat, lon, Datum, PM10, PM2.5

One row per (station, valid day). 

Filter: 
min_days = 90%  (--min-frac, default 0.90)
+
only valid observations (Validity > 0, value != -999) are kept.

Reuses count_eea_stations.py's duckdb + metadata-join logic.
Input Coordinates come from the airbase metadata.csv (Longitude/Latitude).
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
import airbase

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
POLLUTANT_CODES = {"PM10": 5, "PM2.5": 6001}
MISSING = -999.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROC / "eea" / "airbase_raw"))
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--min-frac", type=float, default=0.90,
                    help="min fraction of days valid to keep a station (match the patch list)")
    ap.add_argument("--out", default=str(PROC / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"))
    args = ap.parse_args()

    days = (pd.Timestamp(f"{args.year+1}-01-01") - pd.Timestamp(f"{args.year}-01-01")).days
    min_days = int(args.min_frac * days)
    y0, y1 = f"{args.year}-01-01", f"{args.year+1}-01-01"
    root = Path(args.data_dir)
    con = duckdb.connect()

    # metadata: sampling point -> EoI code + coordinates
    client = airbase.AirbaseClient()
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        client.download_metadata(str(meta_path))
    meta = pd.read_csv(meta_path, low_memory=False)
    meta_pm = meta[meta["Air Pollutant"].isin(POLLUTANT_CODES)][
        ["Sampling Point Id", "Air Quality Station EoI Code",
         "Air Quality Station Name", "Longitude", "Latitude"]
    ].dropna(subset=["Sampling Point Id", "Air Quality Station EoI Code"])
    con.register("meta_pm", meta_pm[["Sampling Point Id", "Air Quality Station EoI Code"]])

    info = (meta_pm.rename(columns={"Air Quality Station EoI Code": "station_code",
                                    "Air Quality Station Name": "station_name",
                                    "Longitude": "lon", "Latitude": "lat"})
            .groupby("station_code")
            .agg(station_name=("station_name", "first"),
                 lat=("lat", "median"), lon=("lon", "median"))
            .reset_index())

    print(f"year {args.year}: need >= {min_days} valid days ({args.min_frac:.0%})\n")

    all_daily = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cc = cdir.name
        glob = str(cdir / "**" / "*.parquet")
        if not any(cdir.rglob("*.parquet")):
            continue
        # daily mean per station/pollutant, keeping only valid non-missing obs
        q = con.execute(f"""
            WITH obs AS (
                SELECT regexp_replace(Samplingpoint, '^[A-Za-z]{{2}}/', '') AS sp_id,
                       Pollutant AS pol,
                       CAST(Start AS DATE) AS d,
                       Value AS v,
                       Validity AS val
                FROM read_parquet('{glob}', union_by_name=true)
                WHERE Pollutant IN ({POLLUTANT_CODES['PM10']}, {POLLUTANT_CODES['PM2.5']})
                  AND Start >= '{y0}' AND Start < '{y1}'
            )
            SELECT m."Air Quality Station EoI Code" AS station_code,
                   obs.pol AS pol, obs.d AS date,
                   avg(TRY_CAST(obs.v AS DOUBLE)) AS value
            FROM obs JOIN meta_pm m ON obs.sp_id = m."Sampling Point Id"
            WHERE obs.val > 0 AND TRY_CAST(obs.v AS DOUBLE) IS NOT NULL
              AND TRY_CAST(obs.v AS DOUBLE) != {MISSING}
              AND TRY_CAST(obs.v AS DOUBLE) >= 0
            GROUP BY 1, 2, 3
        """).fetchdf()
        if not q.empty:
            all_daily.append(q)
            print(f"  {cc}: {q['station_code'].nunique()} stations, {len(q)} station-days")

    daily = pd.concat(all_daily, ignore_index=True)
    daily["pollutant"] = daily["pol"].map({v: k for k, v in POLLUTANT_CODES.items()})

    # completeness filter: keep a station if PM10 or PM2.5 has >= min_days
    vd = daily.groupby(["station_code", "pollutant"])["date"].nunique().unstack(fill_value=0)
    keep = vd[(vd.get("PM10", 0) >= min_days) | (vd.get("PM2.5", 0) >= min_days)].index
    daily = daily[daily["station_code"].isin(keep)]
    print(f"\n{len(keep)} stations pass completeness")

    # wide: one row per station/day with PM10 + PM2.5 columns
    wide = daily.pivot_table(index=["station_code", "date"], columns="pollutant",
                             values="value", aggfunc="first").reset_index()
    for c in ("PM10", "PM2.5"):
        if c not in wide.columns:
            wide[c] = pd.NA
    wide = wide.merge(info, on="station_code", how="left")
    wide["Datum"] = pd.to_datetime(wide["date"]).dt.strftime("%d.%m.%Y")
    wide = wide[["station_code", "station_name", "lat", "lon", "Datum", "PM10", "PM2.5"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")
    print(f"  {wide['station_code'].nunique()} stations, {len(wide)} station-day rows")
    print(f"  countries: {wide['station_code'].str[:2].nunique()}")


if __name__ == "__main__":
    main()
