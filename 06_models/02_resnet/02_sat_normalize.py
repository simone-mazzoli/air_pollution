"""
Normalization for the Sentinel-2 patches downloaded by download_satellite_patches.py,
matched to the BigEarthNetv2 resnet50-s2 pretrained backbone's expected input stats.

Band order check: MULTISPECTRAL_BANDS in the download script is
["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"], which is already the same
order find_stats.py reported for the model (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12).
So the per-band MEAN/STD below line up index-for-index with the saved .npy arrays --
no reordering needed.

MEAN/STD are copied verbatim from find_stats.py's output for
BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0, normalization table '120_nearest'.

Patches are saved as (H, W, C) float32 by the download script. Normalization is
applied per-channel, then any remaining nodata pixels (non-finite or <=0 --
same definition as nodata_fraction() in the download script) are zeroed out AFTER
normalization, so the small fraction of bad pixels that slipped through the
download-time nodata_fraction<=0.10 filter don't hand the CNN extreme values.

This is an on-the-fly transform, not a script that resaves normalized .npy files --
apply it inside a Dataset's __getitem__ so raw patches stay on disk for reuse
(e.g. by the from-scratch CNN, which will want its own normalization stats).

SatellitePatchDataset filters the requested locations against manifest.csv (written
by the download script) so only "ok" (location, stream) pairs are loaded -- anything
"corrupted", "failed", or missing from the manifest entirely is dropped, with a
printed breakdown of how many fell into each bucket.

debug_report() at the bottom saves a before/after RGB grid for a random sample of
patches plus printed value-range/nodata/normalization stats, as a sanity check that
downloading + normalization actually did something reasonable.
"""
import numpy as np
import pandas as pd
import torch
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset

# this script lives at 06_models/02_resnet/, two levels below the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC = BASE_DIR / "data" / "processed"
SATELLITE_DIR = PROC / "satellite"
FOLD_ANNUAL_DIR = PROC / "corrected" / "fold"
# must match the STREAMS keys in download_satellite_patches.py
STREAM_NAMES = ["high_res_multispec", "low_res_multispec"]

MEAN = [438.3721, 614.0557, 588.4096, 942.8433, 1769.9316, 2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325]
STD = [607.0269, 603.2968, 684.5688, 738.4327, 1100.4561, 1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276]

MEAN_ARR = np.array(MEAN, dtype="float32").reshape(1, 1, -1)
STD_ARR = np.array(STD, dtype="float32").reshape(1, 1, -1)


def _canon_loc(loc):
    """Canonical string form of a location ID, matching the actual on-disk .npy
    filenames -- confirmed via --diagnose that download_satellite_patches.py's
    load_locations() read the location column as float64, so files are named
    like "49.0.npy", not "49.npy". Fold annual CSVs, however, have location as
    clean int64. So: always format as a float string, regardless of whether the
    input arrived as an int or a float, so both sides land on "49.0" instead of
    one side guessing "49" and missing every file."""
    return f"{float(loc):.1f}"


def normalize_patch(arr):
    """arr: (H, W, C) float32, raw Sentinel-2 SR values, band order matching MEAN/STD.
    Returns (C, H, W) float32, z-score normalized, nodata pixels zeroed."""
    nodata_mask = ~np.isfinite(arr) | (arr <= 0)
    safe = np.where(nodata_mask, 0.0, arr).astype("float32")
    normed = (safe - MEAN_ARR) / STD_ARR
    normed = np.where(nodata_mask, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


class SatellitePatchDataset(Dataset):
    """Loads .npy patches for one stream (high_res_multispec or low_res_multispec)
    and returns them normalized and channel-first, keyed to a location list so
    labels (calibrated PM values) can be joined in by the caller.

    Requested locations are filtered against manifest.csv (expected at
    stream_dir.parent/manifest.csv, matching the download script's layout) so only
    (location, stream) pairs marked "ok" are kept -- "corrupted", "failed", and
    locations missing from the manifest entirely are dropped, with a printed
    breakdown of counts."""

    def __init__(self, stream_dir, locations, targets=None, manifest_path=None):
        self.stream_dir = Path(stream_dir)
        self.stream_name = self.stream_dir.name
        manifest_path = (Path(manifest_path) if manifest_path is not None
                         else self.stream_dir.parent / "manifest.csv")

        requested_raw = list(locations)
        requested = [_canon_loc(l) for l in requested_raw]
        targets = list(targets) if targets is not None else None

        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            manifest = manifest[manifest["stream"] == self.stream_name]
            status_by_location = {
                _canon_loc(loc): status
                for loc, status in zip(manifest["location"], manifest["status"])
            }
        else:
            print(f"WARNING: no manifest found at {manifest_path}, "
                  f"skipping ok/corrupted filtering")
            status_by_location = {}

        n_float_like = sum(1 for l in requested_raw if isinstance(l, float))
        if n_float_like:
            print(f"[{self.stream_name}] note: {n_float_like}/{len(requested_raw)} "
                  f"location values were read as floats (e.g. "
                  f"{requested_raw[0]!r}) and canonicalized to int-like strings "
                  f"for filename/manifest matching")

        status_counts = Counter()
        ok_locations, ok_targets = [], []
        missing_file_count = 0
        for i, loc in enumerate(requested):
            status = status_by_location.get(loc, "missing_from_manifest")
            status_counts[status] += 1
            if status == "ok":
                if not (self.stream_dir / f"{loc}.npy").exists():
                    missing_file_count += 1
                    continue
                ok_locations.append(loc)
                if targets is not None:
                    ok_targets.append(targets[i])

        print(f"[{self.stream_name}] {len(requested)} requested locations -> "
              f"{len(ok_locations)} ok on disk, "
              f"{missing_file_count} manifest says ok but file missing, "
              f"{status_counts.get('corrupted', 0)} corrupted, "
              f"{status_counts.get('failed', 0)} failed, "
              f"{status_counts.get('missing_from_manifest', 0)} missing from manifest")

        self.locations = ok_locations
        self.targets = ok_targets if targets is not None else None

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        location = self.locations[idx]
        arr = np.load(self.stream_dir / f"{location}.npy")
        patch = torch.from_numpy(normalize_patch(arr))
        if self.targets is None:
            return patch, location
        return patch, torch.tensor(self.targets[idx], dtype=torch.float32)


def _rgb_stretch(rgb, low_pct=2, high_pct=98):
    """Percentile stretch for display only -- raw reflectance values aren't in
    [0,1], and z-scored values aren't either, so both need this to be viewable."""
    lo = np.percentile(rgb, low_pct)
    hi = np.percentile(rgb, high_pct)
    if hi <= lo:
        return np.zeros_like(rgb)
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def debug_report(dataset, n_samples=6, save_path="debug_patches.png", seed=0):
    """Loads a random sample of patches straight from disk (bypassing __getitem__'s
    tensor conversion), prints raw value range / nodata fraction / post-normalization
    mean+std across the sample, and saves a raw-vs-normalized RGB grid to save_path
    so you can actually look at what got downloaded and check normalization isn't
    doing something broken (all-black patches, huge nodata holes, wrong band order
    giving off colors, etc).

    normalized mean/std are NOT expected to land near exactly 0/1 -- MEAN/STD came
    from BigEarthNet's training distribution, not this dataset's, so some offset is
    normal. What you're checking for is "roughly centered, no wild outliers",
    not an exact match."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(dataset.locations))
    if n_samples == 0:
        print("no locations to sample -- dataset is empty after manifest filtering")
        return
    idxs = rng.choice(len(dataset.locations), size=n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6), squeeze=False)

    raw_mins, raw_maxs, raw_means, nodata_fracs = [], [], [], []
    normed_means, normed_stds = [], []

    for col, idx in enumerate(idxs):
        loc = dataset.locations[idx]
        raw = np.load(dataset.stream_dir / f"{loc}.npy")
        nodata_mask = ~np.isfinite(raw) | (raw <= 0)
        nodata_frac = float(np.mean(nodata_mask))
        nodata_fracs.append(nodata_frac)
        raw_mins.append(float(np.nanmin(raw)))
        raw_maxs.append(float(np.nanmax(raw)))
        raw_means.append(float(np.nanmean(raw)))

        normed = normalize_patch(raw)  # (C, H, W)
        normed_means.append(float(normed.mean()))
        normed_stds.append(float(normed.std()))

        # B4/B3/B2 sit at indices 2/1/0 in MEAN/STD's band order
        raw_rgb = raw[..., [2, 1, 0]]
        axes[0, col].imshow(_rgb_stretch(raw_rgb))
        axes[0, col].set_title(f"{loc}\nraw (nodata={nodata_frac:.1%})", fontsize=8)
        axes[0, col].axis("off")

        normed_rgb = np.transpose(normed[[2, 1, 0], :, :], (1, 2, 0))
        axes[1, col].imshow(_rgb_stretch(normed_rgb))
        axes[1, col].set_title("normalized", fontsize=8)
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved debug grid ({n_samples} patches) -> {save_path}")

    print(f"raw value range across sample: "
          f"min={min(raw_mins):.1f} max={max(raw_maxs):.1f} "
          f"mean={np.mean(raw_means):.1f}")
    print(f"nodata fraction across sample: "
          f"min={min(nodata_fracs):.1%} max={max(nodata_fracs):.1%} "
          f"mean={np.mean(nodata_fracs):.1%}")
    print(f"post-normalization per-patch mean/std across sample: "
          f"mean={np.mean(normed_means):.3f} std={np.mean(normed_stds):.3f}")


def run_all_folds_all_streams(n_samples=6, out_dir="debug_patches", seed=0):
    """Default entry point: for every fold's annual calibration file under
    FOLD_ANNUAL_DIR, and for every stream under SATELLITE_DIR, build a dataset
    and run debug_report. Locations differ across fold annual files only in their
    calibrated PM values (fold affects calibration, not which imagery exists), but
    running the manifest-filtering + debug grid per fold still catches a fold-
    specific location set being smaller than expected, so it's kept explicit
    rather than assuming one fold's coverage represents all of them."""
    annual_files = sorted(FOLD_ANNUAL_DIR.glob("*/annual/*.csv"))
    if not annual_files:
        raise FileNotFoundError(
            f"no fold annual files found under {FOLD_ANNUAL_DIR}/*/annual/*.csv -- "
            f"run the calibration script first")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for annual_file in annual_files:
        fold_name = annual_file.parent.parent.name
        locations = pd.read_csv(annual_file)["location"]
        for stream_name in STREAM_NAMES:
            stream_dir = SATELLITE_DIR / stream_name
            if not stream_dir.exists():
                print(f"SKIPPING {fold_name}/{stream_name}: {stream_dir} does not exist")
                continue
            print(f"\n=== fold={fold_name} stream={stream_name} ===")
            dataset = SatellitePatchDataset(stream_dir, locations)
            save_path = out_dir / f"{fold_name}_{stream_name}.png"
            debug_report(dataset, n_samples=n_samples, save_path=save_path, seed=seed)


def diagnose(stream_dir, locations_csv, manifest_path=None):
    """Prints raw facts about what's actually going on -- exact repr+type of
    location values from the CSV, exact filenames actually present on disk, and
    the manifest's own location repr+type -- instead of guessing at a cause."""
    stream_dir = Path(stream_dir)
    manifest_path = (Path(manifest_path) if manifest_path is not None
                     else stream_dir.parent / "manifest.csv")

    print(f"stream_dir = {stream_dir}")
    print(f"  exists: {stream_dir.exists()}")
    print(f"locations_csv = {locations_csv}")
    print(f"manifest_path = {manifest_path}")
    print(f"  exists: {manifest_path.exists()}")
    print()

    df = pd.read_csv(locations_csv)
    print(f"locations_csv columns: {list(df.columns)}")
    print(f"locations_csv 'location' dtype: {df['location'].dtype}")
    print(f"first 5 raw values from locations_csv:")
    for l in df["location"].head(5):
        print(f"  {l!r}  (python type: {type(l).__name__})  canonical: {_canon_loc(l)!r}")
    print()

    if stream_dir.exists():
        all_npy = sorted(stream_dir.glob("*.npy"))
        print(f"{len(all_npy)} .npy files found directly in {stream_dir}")
        print(f"first 5 filenames actually on disk:")
        for f in all_npy[:5]:
            print(f"  {f.name!r}  (stem: {f.stem!r})")
        if not all_npy:
            print(f"  NONE FOUND -- checking for nested subdirectories instead:")
            for p in sorted(stream_dir.iterdir())[:10]:
                print(f"    {p}")
    else:
        print(f"stream_dir does not exist at all")
    print()

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        print(f"manifest columns: {list(manifest.columns)}")
        print(f"manifest 'location' dtype: {manifest['location'].dtype}")
        print(f"manifest 'stream' unique values: {manifest['stream'].unique().tolist()}")
        print(f"first 5 raw manifest rows:")
        print(manifest.head(5).to_string())
    else:
        print(f"no manifest found at {manifest_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-dir", default=None,
                       help="e.g. data/processed/satellite/high_res_multispec -- "
                            "if given (with --locations-csv), runs a single "
                            "fold/stream instead of the all-folds-all-resolutions "
                            "default")
    parser.add_argument("--locations-csv", default=None,
                       help="CSV with a 'location' column, required together "
                            "with --stream-dir")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--save-path", default="debug_patches.png",
                       help="only used in single stream/locations mode")
    parser.add_argument("--out-dir", default="debug_patches",
                       help="only used in the default all-folds-all-resolutions mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnose", action="store_true",
                       help="print raw facts about CSV values, manifest values, "
                            "and actual on-disk filenames for one fold/stream, "
                            "then exit -- no guessing, just print what's there")
    args = parser.parse_args()

    if args.diagnose:
        stream_dir = args.stream_dir or (SATELLITE_DIR / STREAM_NAMES[0])
        locations_csv = args.locations_csv
        if locations_csv is None:
            candidates = sorted(FOLD_ANNUAL_DIR.glob("*/annual/*.csv"))
            if not candidates:
                raise FileNotFoundError(f"no annual files found under {FOLD_ANNUAL_DIR}")
            locations_csv = candidates[0]
        diagnose(stream_dir, locations_csv)
    elif args.stream_dir is not None or args.locations_csv is not None:
        if args.stream_dir is None or args.locations_csv is None:
            parser.error("--stream-dir and --locations-csv must be given together")
        locations = pd.read_csv(args.locations_csv)["location"]
        dataset = SatellitePatchDataset(args.stream_dir, locations)
        debug_report(dataset, n_samples=args.n_samples,
                    save_path=args.save_path, seed=args.seed)
    else:
        run_all_folds_all_streams(n_samples=args.n_samples,
                                  out_dir=args.out_dir, seed=args.seed)
