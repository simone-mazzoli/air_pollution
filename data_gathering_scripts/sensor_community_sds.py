"""
pulls the sensor zip files for whichever months you
specify, for year 2024.

resumable downloads since
the archive server drops connections very often.

TO use examples:
I recommend downloading one month at a time since it takes 2/3 hours for me
    python sensor_community_sds.py 01 02 03
    python sensor_community_sds.py 04 05 06 --workers 2
    python sensor_community_sds.py 10 11 12 --types sds011 pms5003
"""

import argparse
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

YEAR = 2024
BASE_URL = "https://archive.sensor.community/csv_per_month/"
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
HEADERS = {"User-Agent": "dl-course-pm-research/1.0 (student project)"}
LOG_EVERY_SECONDS = 30
SENSOR_TYPES = ["sds011", "pms1003", "pms3003", "pms5003", "pms6003", "pms7003", "sps30"]


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def download_zip(url: str, dest: Path, max_no_progress: int = 5) -> bool:
    """
    Download with resume support. A retry only counts against the
    failure budget if it made zero progress
    """
    failures = 0
    while failures < max_no_progress:
        size_before = dest.stat().st_size if dest.exists() else 0
        try:
            headers = dict(HEADERS)
            mode = "wb"
            if size_before > 0:
                headers["Range"] = f"bytes={size_before}-"
                mode = "ab"

            r = requests.get(url, headers=headers, timeout=180, stream=True)

            if r.status_code == 404:
                print(f"  no data (404): {url}")
                return False
            if size_before > 0 and r.status_code == 200:
                mode = "wb"  # server ignored Range, resending whole file
            else:
                r.raise_for_status()

            total = r.headers.get("Content-Length")
            total_mb = (int(total) + size_before) / 1e6 if total else None
            downloaded = size_before if mode == "ab" else 0
            last_log = time.time()

            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_log > LOG_EVERY_SECONDS:
                        if total_mb:
                            print(f"  {dest.name}: {downloaded/1e6:.0f} / "
                                  f"{total_mb:.0f} MB ({100*downloaded/1e6/total_mb:.0f}%)")
                        last_log = time.time()

            if is_valid_zip(dest):
                return True
            print(f"  {dest.name}: failed validation, restarting")
            dest.unlink(missing_ok=True)
            failures += 1

        except requests.RequestException as e:
            size_after = dest.stat().st_size if dest.exists() else 0
            if size_after > size_before:
                print(f"  {dest.name}: dropped at {size_after/1e6:.0f} MB, resuming")
                time.sleep(2)
                continue
            failures += 1
            print(f"  {dest.name}: no-progress retry {failures}/{max_no_progress} ({e})")
            time.sleep(min(2 ** failures, 30))
    print(f"  {dest.name}: giving up after {max_no_progress} no-progress attempts")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("months", nargs="+",
                        help="Two-digit months to download, e.g. 01 02 03")
    parser.add_argument("--types", nargs="+", default=SENSOR_TYPES,
                        help=f"Sensor types to download (default: all of {SENSOR_TYPES})")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for month in args.months:
        ym = f"{YEAR}-{month}"
        month_dir = RAW_DIR / ym
        month_dir.mkdir(parents=True, exist_ok=True)
        for sensor_type in args.types:
            fname = f"{ym}_{sensor_type}.zip"
            url = f"{BASE_URL}{ym}/{fname}"
            tasks.append((url, month_dir / fname))

    print(f"Downloading {len(tasks)} file(s):")
    for _, dest in tasks:
        print(f"  {dest}")
    print()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_zip, url, dest): dest.name
                for url, dest in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Overall", unit="file"):
            ok = fut.result()
            name = futs[fut]
            print(f"{'done' if ok else 'FAILED'}: {name}")


if __name__ == "__main__":
    main()
