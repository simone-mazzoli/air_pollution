"""
Download monthly Sensor.Community ZIP files for 2024.

Features:
- resumable downloads
- skips existing valid ZIP files
- estimates full and remaining compressed download size
- includes PM and humidity sensors by default

Examples:
    python sensor_community_sds.py 01
    python sensor_community_sds.py 04 05 06 --workers 2
    python sensor_community_sds.py 10 11 12 --types sds011 pms5003 dht22 bme280
"""

import argparse
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

YEAR = 2024
BASE_URL = "https://archive.sensor.community/csv_per_month/"
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
HEADERS = {"User-Agent": "dl-course-pm-research/1.0 (student project)"}
LOG_EVERY_SECONDS = 30
CHUNK_SIZE = 1 << 20  # 1 MiB

SENSOR_TYPES = [
    "sds011",
    "pms1003",
    "pms3003",
    "pms5003",
    "pms6003",
    "pms7003",
    "sps30",
    "dht22",
    "bme280",
]


@dataclass
class Task:
    url: str
    dest: Path
    remote_size: int | None = None
    remote_exists: bool | None = None
    local_size: int = 0
    complete: bool = False


def format_bytes(size: int | float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def valid_month(value: str) -> str:
    try:
        month = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("months must be between 01 and 12") from exc
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError("months must be between 01 and 12")
    return f"{month:02d}"


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return False


def get_remote_size(url: str) -> tuple[int | None, bool | None]:
    """Return (file size in bytes, whether the remote file exists)."""
    try:
        with requests.head(
            url, headers=HEADERS, timeout=30, allow_redirects=True
        ) as response:
            if response.status_code == 404:
                return None, False
            response.raise_for_status()
            if response.headers.get("Content-Length"):
                return int(response.headers["Content-Length"]), True
    except (requests.RequestException, ValueError):
        pass

    # Fallback when HEAD is unsupported or omits Content-Length.
    headers = {**HEADERS, "Range": "bytes=0-0"}
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=30,
            stream=True,
            allow_redirects=True,
        ) as response:
            if response.status_code == 404:
                return None, False
            response.raise_for_status()

            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1]
                if total != "*":
                    return int(total), True

            if response.headers.get("Content-Length"):
                return int(response.headers["Content-Length"]), True

            return None, True
    except (requests.RequestException, ValueError):
        # Unknown is not treated as missing; the real download will still try.
        return None, None


def inspect_tasks(tasks: list[Task], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        futures = {pool.submit(get_remote_size, task.url): task for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Checking sizes",
            unit="file",
        ):
            task = futures[future]
            task.remote_size, task.remote_exists = future.result()

    for task in tasks:
        task.local_size = task.dest.stat().st_size if task.dest.exists() else 0
        task.complete = is_valid_zip(task.dest)


def print_estimate(tasks: list[Task]) -> None:
    available = [task for task in tasks if task.remote_exists is not False]
    missing = [task for task in tasks if task.remote_exists is False]

    full_size = sum(task.remote_size or 0 for task in available)
    complete_size = sum(
        task.remote_size or 0
        for task in available
        if task.complete
    )
    reusable_partial = 0
    remaining_size = 0
    unknown_remaining = 0

    for task in available:
        if task.complete:
            continue
        if task.remote_size is None:
            unknown_remaining += 1
        elif 0 < task.local_size < task.remote_size:
            reusable_partial += task.local_size
            remaining_size += task.remote_size - task.local_size
        else:
            # Invalid files equal to/larger than the remote file are assumed
            # to require a full restart.
            remaining_size += task.remote_size

    print("\nDownload estimate (compressed ZIP sizes):")
    print(f"  Full selection, ignoring local files: {format_bytes(full_size)}")
    print(
        f"  Already complete and skipped:        {format_bytes(complete_size)} "
        f"({sum(task.complete for task in available)} file(s))"
    )
    print(f"  Reusable partial downloads:           {format_bytes(reusable_partial)}")
    print(f"  Estimated remaining transfer:         {format_bytes(remaining_size)}")

    unknown_sizes = sum(task.remote_size is None for task in available)
    if unknown_sizes:
        print(
            f"  Unknown remote size:                  {unknown_sizes} file(s), "
            "not included in byte totals"
        )
    if unknown_remaining:
        print(
            f"  Remaining also includes:              {unknown_remaining} "
            "file(s) of unknown size"
        )
    if missing:
        print(f"  Missing from archive:                 {len(missing)} file(s)")
        for task in missing:
            print(f"    {task.dest.name}")
    print()


def download_zip(url: str, dest: Path, max_no_progress: int = 5) -> bool:
    """
    Download with resume support.

    A retry only counts against the failure budget if it made zero progress.
    """
    if is_valid_zip(dest):
        print(f"  already complete, skipping: {dest.name}")
        return True

    failures = 0
    while failures < max_no_progress:
        size_before = dest.stat().st_size if dest.exists() else 0

        try:
            headers = dict(HEADERS)
            mode = "wb"
            if size_before > 0:
                headers["Range"] = f"bytes={size_before}-"
                mode = "ab"

            response = requests.get(
                url, headers=headers, timeout=180, stream=True
            )

            if response.status_code == 404:
                print(f"  no data (404): {url}")
                response.close()
                return False

            if response.status_code == 416 and size_before > 0:
                response.close()
                if is_valid_zip(dest):
                    return True
                print(f"  {dest.name}: invalid local file; restarting")
                dest.unlink(missing_ok=True)
                failures += 1
                continue

            if size_before > 0 and response.status_code == 200:
                mode = "wb"  # Server ignored Range and resent the full file.

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            response_size = int(content_length) if content_length else None
            total_size = (
                response_size + size_before
                if response_size is not None and mode == "ab"
                else response_size
            )
            downloaded = size_before if mode == "ab" else 0
            last_log = time.time()

            with response, open(dest, mode) as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)

                    if time.time() - last_log > LOG_EVERY_SECONDS:
                        if total_size:
                            percent = 100 * downloaded / total_size
                            print(
                                f"  {dest.name}: {format_bytes(downloaded)} / "
                                f"{format_bytes(total_size)} ({percent:.0f}%)"
                            )
                        else:
                            print(f"  {dest.name}: {format_bytes(downloaded)} downloaded")
                        last_log = time.time()

            if is_valid_zip(dest):
                return True

            print(f"  {dest.name}: failed validation, restarting")
            dest.unlink(missing_ok=True)
            failures += 1

        except requests.RequestException as exc:
            size_after = dest.stat().st_size if dest.exists() else 0
            if size_after > size_before:
                print(
                    f"  {dest.name}: dropped at {format_bytes(size_after)}, resuming"
                )
                time.sleep(2)
                continue

            failures += 1
            print(
                f"  {dest.name}: no-progress retry "
                f"{failures}/{max_no_progress} ({exc})"
            )
            time.sleep(min(2**failures, 30))

        except OSError as exc:
            print(f"  {dest.name}: local file error ({exc})")
            return False

    print(f"  {dest.name}: giving up after {max_no_progress} no-progress attempts")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "months",
        nargs="+",
        type=valid_month,
        help="Months to download, e.g. 01 02 03",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=SENSOR_TYPES,
        help=f"Sensor types to download (default: {SENSOR_TYPES})",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Remove accidental duplicates while preserving the supplied order.
    months = list(dict.fromkeys(args.months))
    sensor_types = list(dict.fromkeys(args.types))

    tasks: list[Task] = []
    for month in months:
        ym = f"{YEAR}-{month}"
        month_dir = RAW_DIR / ym
        month_dir.mkdir(parents=True, exist_ok=True)

        for sensor_type in sensor_types:
            filename = f"{ym}_{sensor_type}.zip"
            tasks.append(
                Task(
                    url=f"{BASE_URL}{ym}/{filename}",
                    dest=month_dir / filename,
                )
            )

    print(f"Preparing {len(tasks)} archive file(s)...")
    inspect_tasks(tasks, args.workers)
    print_estimate(tasks)

    pending = [
        task
        for task in tasks
        if not task.complete and task.remote_exists is not False
    ]

    if not pending:
        print("Nothing to download: all available files are already complete.")
        return

    print(
        f"Downloading {len(pending)} file(s); "
        f"skipping {len(tasks) - len(pending)} complete or missing file(s):"
    )
    for task in pending:
        print(f"  {task.dest}")
    print()

    succeeded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_zip, task.url, task.dest): task
            for task in pending
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Overall",
            unit="file",
        ):
            task = futures[future]
            try:
                ok = future.result()
            except Exception as exc:
                ok = False
                print(f"FAILED: {task.dest.name} ({exc})")
            else:
                print(f"{'done' if ok else 'FAILED'}: {task.dest.name}")

            succeeded += int(ok)
            failed += int(not ok)

    print(f"\nFinished: {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()