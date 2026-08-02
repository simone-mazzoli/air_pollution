"""
Check whether dataset=2 (E1a/Verified) actually supports aggregationType="day"
on the EEA parquet download API, or whether it silently ignores it and still
returns hourly rows (or nothing).

Submits two small test jobs for one small/fast country (LU by default):
  - aggregationType="hour"  (known to work)
  - aggregationType="day"   (the thing we're not sure about)
then compares row counts and timestamp spacing to see whether "day" really
changed anything.
"""
import argparse
import io
import time
import zipfile

import pandas as pd
import requests

API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
DATASET_E1A = 2
POLLUTANTS = ["PM10"]
TIMEOUT = 120
POLL_INTERVAL = 10


def submit(country, year, agg):
    body = {
        "countries": [country],
        "cities": [],
        "pollutants": POLLUTANTS,
        "dataset": DATASET_E1A,
        "dateTimeStart": f"{year}-01-01T00:00:00Z",
        "dateTimeEnd": f"{year}-01-10T23:59:59Z",
        "aggregationType": agg,
        "source": "Custom script",
    }
    r = requests.post(f"{API}ParquetFile/async", json=body, timeout=TIMEOUT)
    print(f"  [{agg}] submit status {r.status_code}")
    if r.status_code not in (200, 202):
        print(f"  [{agg}] response body: {r.text[:300]}")
        return None
    url = r.text.strip().strip('"')
    if not url.startswith("http"):
        print(f"  [{agg}] no job URL returned: {url[:200]}")
        return None
    return url


def wait_for_zip(url, agg):
    t0 = time.time()
    poll_count = 0
    while True:
        pr = requests.get(url, timeout=TIMEOUT)
        if pr.status_code == 404:
            poll_count += 1
            print(f"  [{agg}] still waiting ({int(time.time() - t0)}s)...")
            time.sleep(POLL_INTERVAL)
            continue
        if pr.status_code not in (200, 206):
            print(f"  [{agg}] job returned {pr.status_code}")
            return None
        print(f"  [{agg}] zip ready after {int(time.time() - t0)}s")
        return pr.content


def load_parquet_zip(content, agg):
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        print(f"  [{agg}] response wasn't a valid zip")
        return None
    names = [n for n in zf.namelist() if n.lower().endswith(".parquet")]
    if not names:
        print(f"  [{agg}] zip had no parquet files, entries: {zf.namelist()[:10]}")
        return None
    frames = [pd.read_parquet(io.BytesIO(zf.read(n))) for n in names]
    return pd.concat(frames, ignore_index=True)


def inspect(df, agg):
    print(f"\n[{agg}] result summary")
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    cols = {c.lower(): c for c in df.columns}
    stt = cols.get("start")
    if stt is None:
        print("  no Start column found, can't check timestamp spacing")
        return
    ts = pd.to_datetime(df[stt], errors="coerce").dropna().sort_values().unique()
    print(f"  distinct timestamps: {len(ts)}")
    if len(ts) >= 2:
        deltas = pd.Series(ts).diff().dropna()
        print(f"  most common gap between timestamps: {deltas.mode().iloc[0]}")
        print(f"  first few timestamps: {list(ts[:5])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="LU")
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()

    results = {}
    for agg in ("hour", "day"):
        print(f"\nSubmitting {agg} job for {args.country} {args.year}-01-01..10...")
        url = submit(args.country, args.year, agg)
        if url is None:
            continue
        content = wait_for_zip(url, agg)
        if content is None:
            continue
        df = load_parquet_zip(content, agg)
        if df is None:
            continue
        results[agg] = df

    for agg, df in results.items():
        inspect(df, agg)

    if "hour" in results and "day" in results:
        print("\nComparison")
        print(f"  hour rows: {len(results['hour'])}   day rows: {len(results['day'])}")
        if len(results["day"]) < len(results["hour"]):
            print("  -> 'day' returned fewer rows than 'hour': looks like server-side "
                  "daily aggregation IS happening.")
        elif len(results["day"]) == len(results["hour"]):
            print("  -> 'day' returned the SAME row count as 'hour': aggregationType "
                  "is probably being ignored for this dataset.")
        else:
            print("  -> 'day' returned MORE rows than 'hour', which is unexpected; "
                  "inspect both DataFrames manually.")


if __name__ == "__main__":
    main()
