"""
Download 2024 daily PM10/PM2.5 for all EEA reference stations and write a CSV:
Output
    data/processed/daily_avg/eea/pm_reference_stations_<YEAR>.csv
    columns: station_code, station_name, lat, lon, Datum, PM10, PM2.5

Notes:
  - Coordinates and measurements live in different places. Measurements come from
    the download API keyed by sampling point. coordinates come from the ArcGIS
    "AQ Stations" layer keyed by station EoI code. We join on the EoI code, which
    is embedded in the sampling-point id (e.g. "BE/SPO-BETR202_..." -> BETR202).
  - 2024 exists in the verified E1a set (dataset=2).
  - E1a has no daily aggregate, so we pull hourly and average to daily ourselves.
  - Values of -999 mean missing; Validity < 1 means invalid.

Two phases, each resumable:
  --meta      download + cache station coordinates
  --measure   download measurements per country, join coords, write the CSV
Both run by default. Per-country results are cached.
"""
import argparse
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
OUT_DIR = PROC / "daily_avg" / "eea"
META_CACHE = PROC / "eea" / "station_meta.csv"
CACHE_DIR = PROC / "eea" / "daily_by_country"

API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/"
META_ARCGIS = ("https://air.discomap.eea.europa.eu/arcgis/rest/services/AirQuality/"
               "AirQualityDownloadServiceEUMonitoringStations/MapServer/0/query")
ARCGIS_PAGE = 2000

DATASET_E1A = 2              # verified data; the only set that still holds 2024
AGG_HOURLY = "hour"
POLLUTANTS = ["PM10", "PM2.5"]
POLLUTANT_CODE = {5: "PM10", 6001: "PM2.5"}   # EEA integer codes
MISSING_VALUE = -999.0

RETRY, RETRY_DELAY, TIMEOUT, POLL_TIMEOUT = 3, 5, 120, 3600

# EEA-38 + cooperating countries. Trim with --countries.
EEA_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IS", "IT", "LI", "LT", "LU", "LV", "ME", "MK",
    "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "TR", "XK",
]


def download_meta():
    """Page through the ArcGIS AQ Stations layer, cache coords keyed by EoI code."""
    META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading EEA station metadata (ArcGIS AQ Stations layer)...")
    rows, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "AirQualityStationEoICode,CountryCode,AirQualityStation",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": ARCGIS_PAGE,
        }
        for attempt in range(1, RETRY + 1):
            try:
                js = requests.get(META_ARCGIS, params=params, timeout=TIMEOUT).json()
                break
            except Exception as e:
                print(f"  offset {offset} attempt {attempt} failed: {e}")
                if attempt == RETRY:
                    raise SystemExit("Could not reach the ArcGIS station service.")
                time.sleep(RETRY_DELAY)
        feats = js.get("features", [])
        if not feats:
            break
        for f in feats:
            a, g = f.get("attributes", {}), f.get("geometry", {})
            rows.append({
                "station_code": str(a.get("AirQualityStationEoICode", "")),
                "station_name": str(a.get("AirQualityStation", "")),
                "lat": g.get("y"),
                "lon": g.get("x"),
            })
        offset += len(feats)
        print(f"  fetched {offset} stations...")
        if len(feats) < ARCGIS_PAGE:
            break
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["station_code"] != ""].drop_duplicates("station_code")
    df.to_csv(META_CACHE, index=False)
    print(f"  saved {len(df)} stations -> {META_CACHE}\n")
    return df


def load_meta():
    if META_CACHE.exists():
        print(f"Using cached station metadata ({META_CACHE.name})")
        return pd.read_csv(META_CACHE, dtype={"station_code": str})
    return None


def fetch_country(country, year):
    """Kick off the async E1a hourly download for one country, wait for the zip,
    return the concatenated raw parquet as a DataFrame (or None)."""
    body = {
        "countries": [country],
        "cities": [],
        "pollutants": POLLUTANTS,
        "dataset": DATASET_E1A,
        "aggregationType": AGG_HOURLY,
        "source": "Custom script",
    }
    try:
        r = requests.post(f"{API}ParquetFile/async", json=body, timeout=TIMEOUT)
    except Exception as e:
        print(f"    {country}: request failed ({e})"); return None
    if r.status_code not in (200, 202):
        print(f"    {country}: API returned {r.status_code}"); return None
    url = r.text.strip().strip('"')
    if not url.startswith("http"):
        print(f"    {country}: no job URL ({url[:100]})"); return None

    print(f"    {country}: job submitted, waiting for the zip...")
    t0 = time.time()
    while True:
        if time.time() - t0 > POLL_TIMEOUT:
            print(f"    {country}: timed out"); return None
        pr = requests.get(url, timeout=TIMEOUT)
        if pr.status_code == 404:
            time.sleep(15); continue
        if pr.status_code not in (200, 206):
            print(f"    {country}: job returned {pr.status_code}"); return None
        break
    try:
        zf = zipfile.ZipFile(io.BytesIO(pr.content))
    except zipfile.BadZipFile:
        print(f"    {country}: no data"); return None

    names = [n for n in zf.namelist() if n.lower().endswith(".parquet")]
    print(f"    {country}: downloaded {len(pr.content) / 1e6:.1f} MB, "
          f"parsing {len(names)} files...")
    frames = []
    for k, name in enumerate(names, 1):
        try:
            frames.append(pd.read_parquet(io.BytesIO(zf.read(name))))
        except Exception as e:
            print(f"      skip {name}: {e}")
        if k % 25 == 0 or k == len(names):
            print(f"      {country}: parsed {k}/{len(names)}")
    return pd.concat(frames, ignore_index=True) if frames else None


def to_daily(df, year):
    """EEA hourly parquet -> daily means. Keep valid, non-missing PM10/PM2.5 rows
    of the target year, then average hourly to one value per point/pollutant/day."""
    cols = {c.lower(): c for c in df.columns}
    sp, po, stt, val, valid = (cols.get("samplingpoint"), cols.get("pollutant"),
                               cols.get("start"), cols.get("value"), cols.get("validity"))
    if not all([sp, po, stt, val]):
        raise SystemExit(f"unexpected EEA columns: {list(df.columns)}")

    pol = pd.to_numeric(df[po], errors="coerce")
    value = pd.to_numeric(df[val], errors="coerce")     # object -> float
    ts = pd.to_datetime(df[stt], errors="coerce")
    ok = (pol.isin(POLLUTANT_CODE) & value.notna() & (value != MISSING_VALUE)
          & (value >= 0) & ts.notna() & (ts.dt.year == year))
    if valid is not None:
        ok &= pd.to_numeric(df[valid], errors="coerce").fillna(-1) >= 1
    if not ok.any():
        return pd.DataFrame(columns=["sampling_point", "pollutant", "date", "value"])

    out = pd.DataFrame({
        "sampling_point": df[sp].astype(str)[ok].values,
        "pollutant": pol[ok].map(POLLUTANT_CODE).values,
        "date": ts[ok].dt.date.values,
        "value": value[ok].values,
    })
    return out.groupby(["sampling_point", "pollutant", "date"], as_index=False)["value"].mean()


def map_to_station(sampling_points, meta):
    """Each sampling-point id embeds its station EoI code; find it. Longest code
    first so e.g. DEBW0872 isn't shadowed by DEBW087."""
    codes = sorted(meta["station_code"].unique(), key=len, reverse=True)
    by_prefix = {}
    for c in codes:
        by_prefix.setdefault(c[:2], []).append(c)

    def find(sp):
        for c in by_prefix.get(sp[:2], codes):
            if c and c in sp:
                return c
        for c in codes:
            if c and c in sp:
                return c
        return None

    return {sp: find(sp) for sp in sampling_points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--countries", nargs="+", default=None,
                    help="subset of EEA country codes (default: all)")
    ap.add_argument("--meta", action="store_true", help="only (re)download metadata")
    ap.add_argument("--measure", action="store_true", help="only download measurements")
    ap.add_argument("--force", action="store_true", help="re-download cached countries")
    args = ap.parse_args()
    do_meta = args.meta or not args.measure
    do_measure = args.measure or not args.meta

    meta = load_meta()
    if do_meta or meta is None:
        meta = download_meta()
    if not do_measure:
        return

    countries = args.countries or EEA_COUNTRIES
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.year} PM10/PM2.5 for {len(countries)} countries\n")

    all_daily = []
    for i, c in enumerate(countries, 1):
        cache_f = CACHE_DIR / f"{c}_{args.year}.parquet"
        if cache_f.exists() and not args.force:
            d = pd.read_parquet(cache_f)
            print(f"[{i}/{len(countries)}] {c}: cached ({len(d)} records)")
            all_daily.append(d); continue
        print(f"[{i}/{len(countries)}] {c}...")
        raw = fetch_country(c, args.year)
        if raw is None:
            continue
        d = to_daily(raw, args.year)
        if len(d) == 0:
            print(f"    {c}: no {args.year} PM data"); continue
        d.to_parquet(cache_f, index=False)
        print(f"    {c}: {len(d)} records ({d['sampling_point'].nunique()} points) -> cached")
        all_daily.append(d)

    if not all_daily:
        print("Nothing downloaded."); return

    long = pd.concat(all_daily, ignore_index=True)
    long["station_code"] = long["sampling_point"].map(
        map_to_station(long["sampling_point"].unique(), meta))
    unmatched = long["station_code"].isna()
    if unmatched.any():
        print(f"\n{long.loc[unmatched, 'sampling_point'].nunique()} points had no "
              f"station match, dropped")
        long = long[~unmatched]

    j = long.merge(meta[["station_code", "station_name", "lat", "lon"]],
                   on="station_code", how="inner")
    # one row per station/day; average across a station's sampling points
    g = j.groupby(["station_code", "station_name", "lat", "lon", "date", "pollutant"],
                  as_index=False)["value"].mean()
    wide = g.pivot_table(index=["station_code", "station_name", "lat", "lon", "date"],
                         columns="pollutant", values="value", aggfunc="first").reset_index()
    for col in ("PM10", "PM2.5"):
        if col not in wide.columns:
            wide[col] = pd.NA
    wide["Datum"] = pd.to_datetime(wide["date"]).dt.strftime("%d.%m.%Y")
    wide = wide[["station_code", "station_name", "lat", "lon", "Datum", "PM10", "PM2.5"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pm_reference_stations_{args.year}.csv"
    wide.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"  {wide['station_code'].nunique()} stations, {len(wide)} station-day rows")


if __name__ == "__main__":
    main()
