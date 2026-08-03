"""
Count distinct EEA stations (by EoI code) that pass a validity threshold (at least one obs for 90% of days of the year),
per country, for a given year. 

Coordinates come from the airbase metadata.csv (Longitude/Latitude), so no separate coordinate file is needed.

Output: data/processed/eea/stations_to_download.csv:
    station_code, lat, lon, valid_days_pm10, valid_days_pm25
(one row per passing EoI station, the input list for the patch downloader).
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
import airbase

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"

POLLUTANT_CODES = {"PM10": 5, "PM2.5": 6001}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROC / "eea" / "airbase_raw"))
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--min-frac", type=float, default=0.90,
                    help="min fraction of the year's days that must be valid (default 0.90)")
    ap.add_argument("--out", default=str(PROC / "eea" / "stations_to_download.csv"))
    args = ap.parse_args()

    days_in_year = (pd.Timestamp(f"{args.year+1}-01-01") - pd.Timestamp(f"{args.year}-01-01")).days
    min_days = int(args.min_frac * days_in_year)
    year_start, year_end = f"{args.year}-01-01", f"{args.year+1}-01-01"
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # airbase metadata: sampling point -> EoI code + station coordinates
    client = airbase.AirbaseClient()
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        client.download_metadata(str(meta_path))
    meta = pd.read_csv(meta_path, low_memory=False)
    meta_pm = meta[meta["Air Pollutant"].isin(POLLUTANT_CODES)][
        ["Sampling Point Id", "Air Quality Station EoI Code", "Longitude", "Latitude"]
    ].dropna(subset=["Sampling Point Id", "Air Quality Station EoI Code"])
    con.register("meta_pm", meta_pm[["Sampling Point Id", "Air Quality Station EoI Code"]])

    # one coordinate per station (median over its sampling points, guards GPS jitter)
    coords = (meta_pm.rename(columns={"Air Quality Station EoI Code": "station_code",
                                      "Longitude": "lon", "Latitude": "lat"})
              .groupby("station_code")[["lat", "lon"]].median().reset_index())

    print(f"year {args.year}: {days_in_year} days, need >= {min_days} valid days "
          f"({args.min_frac:.0%})\n")

    rows, per_country = [], []
    for country_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cc = country_dir.name
        glob = str(country_dir / "**" / "*.parquet")
        if not any(country_dir.rglob("*.parquet")):
            continue
        q = con.execute(f"""
            WITH obs AS (
                SELECT
                    regexp_replace(Samplingpoint, '^[A-Za-z]{{2}}/', '') AS sp_id,
                    Pollutant AS pol,
                    CAST(Start AS DATE) AS d,
                    max(CASE WHEN Validity > 0 THEN 1 ELSE 0 END) AS day_valid
                FROM read_parquet('{glob}', union_by_name=true)
                WHERE Pollutant IN ({POLLUTANT_CODES['PM10']}, {POLLUTANT_CODES['PM2.5']})
                  AND Start >= '{year_start}' AND Start < '{year_end}'
                GROUP BY 1, 2, 3
            )
            SELECT m."Air Quality Station EoI Code" AS station_code,
                   obs.pol,
                   sum(obs.day_valid) AS valid_days
            FROM obs JOIN meta_pm m ON obs.sp_id = m."Sampling Point Id"
            GROUP BY 1, 2
        """).fetchdf()
        if q.empty:
            continue
        wide = q.pivot_table(index="station_code", columns="pol",
                             values="valid_days", aggfunc="sum").fillna(0)
        pm10 = wide.get(POLLUTANT_CODES["PM10"], pd.Series(0, index=wide.index))
        pm25 = wide.get(POLLUTANT_CODES["PM2.5"], pd.Series(0, index=wide.index))
        passing = wide[(pm10 >= min_days) | (pm25 >= min_days)].index
        for st in passing:
            rows.append({"station_code": st,
                         "valid_days_pm10": int(pm10.get(st, 0)),
                         "valid_days_pm25": int(pm25.get(st, 0))})
        per_country.append((cc, len(wide), len(passing)))

    passed = pd.DataFrame(rows)
    print(f"{'country':>8}{'stations':>10}{'passing':>9}")
    for cc, tot, p in sorted(per_country):
        print(f"{cc:>8}{tot:>10}{p:>9}")
    print(f"\ntotal passing stations: {len(passed)}")

    out = passed.merge(coords, on="station_code", how="left")
    missing = out["lat"].isna()
    if missing.any():
        print(f"WARNING: {int(missing.sum())} passing stations have no coordinates, dropped")
        out = out[~missing]
    out = out[["station_code", "lat", "lon", "valid_days_pm10", "valid_days_pm25"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"with coordinates: {len(out)} stations -> {args.out}")


if __name__ == "__main__":
    main()
