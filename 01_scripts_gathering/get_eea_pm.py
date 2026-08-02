"""
Download PM10/PM2.5 using the airbase package instead of the raw
ParquetFile/async endpoint (much less time needed). airbase requests individual
per-station parquet file URLs and downloads them concurrently.
"""
import argparse
from pathlib import Path
import airbase
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", nargs="+", default=None,
                     help="2-letter country codes; default: all EEA-reporting countries")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out-dir", default="data/processed/eea/airbase_raw")
    args = ap.parse_args()

    needed_cols = ["Samplingpoint", "Pollutant", "Start", "Value", "Validity"]
    client = airbase.AirbaseClient()
    countries = args.country if args.country else sorted(client.countries)

    for country in countries:
        print(f"\n=== {country} ===")
        out_dir = Path(args.out_dir) / country
        out_dir.mkdir(parents=True, exist_ok=True)
        r = client.request("Verified", country, poll=["PM10", "PM2.5"])
        r.download(dir=str(out_dir), skip_existing=True)
        files = list(out_dir.rglob("*.parquet"))
        print(f"downloaded {len(files)} parquet files -> {out_dir}")
        chunks = []
        for i, f in enumerate(files, 1):
            try:
                d = pd.read_parquet(f, columns=needed_cols)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue
            ts = pd.to_datetime(d["Start"], errors="coerce")
            d = d[ts.dt.year == args.year]
            if len(d) > 0:
                chunks.append(d)
            if i % 100 == 0 or i == len(files):
                print(f"  processed {i}/{len(files)} files")
        if not chunks:
            print(f"no {args.year} rows found for {country}")
            continue
        df = pd.concat(chunks, ignore_index=True)
        print(f"rows for {country} {args.year}: {len(df)}")
        print(df.head())


if __name__ == "__main__":
    main()
