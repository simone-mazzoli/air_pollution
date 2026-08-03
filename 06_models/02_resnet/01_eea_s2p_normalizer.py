"""
Normalization + Dataset for the EEA Sentinel-2 patches (high_res / low_res),
matched to the BigEarthNetv2 resnet50-s2 backbone's expected input stats.

Band order: patches are saved (H,W,10) in order
[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], the same order the pretrained model expects,
so MEAN/STD below line up index-for-index -- no reordering.

Normalization is an on-the-fly transform (in __getitem__), not a resave, so raw
patches stay on disk for reuse (e.g. a from-scratch CNN with its own stats).

Only S2 is handled here. The S5P streams (no2/aer/co) are single-channel columns
normalized per-fold from TRAIN stats inside the trainer (to avoid leaking the
held-out region), so they stay raw on disk and are not touched by this script.

Patches are keyed by EEA station_code (e.g. "BELAL01"). Labels are the annual
mean PM per station, averaged from the daily EEA reference file.
"""
import numpy as np
import pandas as pd
import torch
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC = BASE_DIR / "data" / "processed"
SATELLITE_DIR = PROC / "satellite_eea"
LABELS_CSV = PROC / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"

# stream key -> (folder on disk, manifest filename)
STREAMS = {
    "high_res": ("high_res_multispec", "manifest_high_res.csv"),
    "low_res":  ("low_res_multispec", "manifest_low_res.csv"),
}

# BigEarthNetv2 resnet50-s2-v0.2.0, table '120_nearest'
MEAN = [438.3721, 614.0557, 588.4096, 942.8433, 1769.9316,
        2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325]
STD = [607.0269, 603.2968, 684.5688, 738.4327, 1100.4561,
       1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276]
MEAN_ARR = np.array(MEAN, dtype="float32").reshape(1, 1, -1)
STD_ARR = np.array(STD, dtype="float32").reshape(1, 1, -1)


def normalize_patch(arr):
    """(H,W,C) raw S2 -> (C,H,W) z-scored, nodata pixels zeroed after norming."""
    nodata = ~np.isfinite(arr) | (arr <= 0)
    safe = np.where(nodata, 0.0, arr).astype("float32")
    normed = (safe - MEAN_ARR) / STD_ARR
    normed = np.where(nodata, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


def load_labels(pollutant="PM10"):
    """Annual mean per station from the daily EEA reference file."""
    df = pd.read_csv(LABELS_CSV, dtype={"station_code": str})
    ann = df.groupby("station_code")[pollutant].mean().dropna()
    return ann.to_dict()


class EEAPatchDataset(Dataset):
    """Loads normalized S2 patches for one stream, keyed by station_code, keeping
    only stations the manifest marks "ok" and whose file is on disk. If a label
    dict is given, returns (patch, target); otherwise (patch, station_code)."""

    def __init__(self, stream, station_codes, labels=None):
        folder, manifest_name = STREAMS[stream]
        self.stream = stream
        self.stream_dir = SATELLITE_DIR / folder
        manifest_path = SATELLITE_DIR / manifest_name

        requested = [str(s) for s in station_codes]

        # manifest status by station (stream column holds the flag key, e.g. "high_res")
        if manifest_path.exists():
            man = pd.read_csv(manifest_path, dtype={"station_code": str})
            man = man[man["stream"] == stream]
            status = dict(zip(man["station_code"], man["status"]))
        else:
            print(f"WARNING: no manifest at {manifest_path}, skipping status filter")
            status = {}

        counts = Counter()
        self.codes, self.targets = [], ([] if labels is not None else None)
        missing_file = 0
        for code in requested:
            st = status.get(code, "missing_from_manifest")
            counts[st] += 1
            if st != "ok":
                continue
            if labels is not None and code not in labels:
                counts["no_label"] += 1
                continue
            if not (self.stream_dir / f"{code}.npy").exists():
                missing_file += 1
                continue
            self.codes.append(code)
            if labels is not None:
                self.targets.append(labels[code])

        print(f"[{stream}] {len(requested)} requested -> {len(self.codes)} usable  "
              f"(ok={counts.get('ok',0)}, corrupted={counts.get('corrupted',0)}, "
              f"failed={counts.get('failed',0)}, "
              f"missing_from_manifest={counts.get('missing_from_manifest',0)}, "
              f"no_label={counts.get('no_label',0)}, file_missing={missing_file})")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        arr = np.load(self.stream_dir / f"{code}.npy")
        patch = torch.from_numpy(normalize_patch(arr))
        if self.targets is None:
            return patch, code
        return patch, torch.tensor(self.targets[idx], dtype=torch.float32)


# --------------------------------------------------------------------- debug
def _rgb_stretch(rgb, low=2, high=98):
    lo, hi = np.percentile(rgb, low), np.percentile(rgb, high)
    return np.zeros_like(rgb) if hi <= lo else np.clip((rgb - lo) / (hi - lo), 0, 1)


def debug_report(dataset, n=6, save_path="debug_eea_patches.png", seed=0):
    """Raw-vs-normalized RGB grid + value/nodata/norm stats for a random sample."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n = min(n, len(dataset.codes))
    if n == 0:
        print("dataset empty after filtering"); return
    idxs = rng.choice(len(dataset.codes), size=n, replace=False)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    rmins, rmaxs, fracs, nmeans, nstds = [], [], [], [], []
    for c, idx in enumerate(idxs):
        code = dataset.codes[idx]
        raw = np.load(dataset.stream_dir / f"{code}.npy")
        frac = float(np.mean(~np.isfinite(raw) | (raw <= 0)))
        rmins.append(float(np.nanmin(raw))); rmaxs.append(float(np.nanmax(raw)))
        fracs.append(frac)
        normed = normalize_patch(raw)
        nmeans.append(float(normed.mean())); nstds.append(float(normed.std()))
        ax[0, c].imshow(_rgb_stretch(raw[..., [2, 1, 0]]))
        ax[0, c].set_title(f"{code}\nraw nodata={frac:.1%}", fontsize=8); ax[0, c].axis("off")
        ax[1, c].imshow(_rgb_stretch(np.transpose(normed[[2, 1, 0]], (1, 2, 0))))
        ax[1, c].set_title("normalized", fontsize=8); ax[1, c].axis("off")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"saved {save_path}")
    print(f"raw range: min={min(rmins):.1f} max={max(rmaxs):.1f}")
    print(f"nodata: min={min(fracs):.1%} max={max(fracs):.1%} mean={np.mean(fracs):.1%}")
    print(f"post-norm per-patch mean/std: {np.mean(nmeans):.3f} / {np.mean(nstds):.3f}")
    print("(mean/std need not be exactly 0/1: MEAN/STD are BigEarthNet's, not this "
          "dataset's -- checking for 'roughly centered, no wild outliers')")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="high_res", choices=list(STREAMS))
    ap.add_argument("--pollutant", default="PM10", choices=["PM10", "PM2.5"])
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--save-path", default="debug_eea_patches.png")
    args = ap.parse_args()

    labels = load_labels(args.pollutant)
    ds = EEAPatchDataset(args.stream, list(labels.keys()), labels=labels)
    debug_report(ds, n=args.n_samples, save_path=args.save_path)
