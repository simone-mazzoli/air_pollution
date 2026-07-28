"""
Sanity-check plot: 5 sensor locations, high-res vs low-res RGB side by side.

Reads the .npy patches written by fetch_satellite.py, slices B4/B3/B2 out of the
multispectral array for an RGB preview, percentile-stretches each for display
(Sentinel-2 reflectance is dark and skewed, so a raw 0-1 scale looks black), and
lays them out one row per location: high-res | low-res.

Usage: python visualize_patches.py [--n 5] [--locations 11789 20067 ...]
Output: data/processed/satellite/preview_patches.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
SAT_DIR = BASE_DIR / "data" / "processed" / "satellite"
HIGH_DIR = SAT_DIR / "high_res_multispec"
LOW_DIR = SAT_DIR / "low_res_multispec"
OUT_PATH = SAT_DIR / "preview_patches.png"

# B4/B3/B2 within MULTISPECTRAL_BANDS ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
RGB_SLICE_INDEX = [2, 1, 0]


def to_rgb(arr, low_pct=2, high_pct=98):
    """(H,W,10) -> (H,W,3) uint8, per-band percentile stretch for display."""
    rgb = arr[..., RGB_SLICE_INDEX].astype("float32")
    out = np.zeros_like(rgb)
    for c in range(3):
        band = rgb[..., c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1
        out[..., c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def pick_locations(n, explicit):
    if explicit:
        return [str(x) for x in explicit]
    # locations present in BOTH streams
    high = {p.stem for p in HIGH_DIR.glob("*.npy")}
    low = {p.stem for p in LOW_DIR.glob("*.npy")}
    both = sorted(high & low)
    if not both:
        raise SystemExit(f"no locations found in both {HIGH_DIR} and {LOW_DIR}")
    return both[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of locations (default 5)")
    ap.add_argument("--locations", nargs="+", default=None,
                    help="specific location ids (default: first N present in both streams)")
    args = ap.parse_args()

    locs = pick_locations(args.n, args.locations)
    n = len(locs)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, loc in enumerate(locs):
        for j, (d, label) in enumerate([(HIGH_DIR, "high-res 1.2km"),
                                        (LOW_DIR, "low-res 12km")]):
            ax = axes[i, j]
            path = d / f"{loc}.npy"
            if path.exists():
                arr = np.load(path)
                ax.imshow(to_rgb(arr))
                ax.set_title(f"{loc}  {label}\n{arr.shape[0]}x{arr.shape[1]}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_title(f"{loc}  {label}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}  ({n} locations)")


if __name__ == "__main__":
    main()
