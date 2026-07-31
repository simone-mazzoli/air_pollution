"""
Multi-stream CNN for annual PM10 from Sentinel-2 + Sentinel-5P, held-out-Land CV.

Streams:
  - HIGH-RES S2 (120x120x10)  -> pretrained BigEarthNet ResNet50 -> 2048 -> Linear -> PROJ
  - LOW-RES  S2 (60x60x10)    -> small from-scratch CNN          -> 128 feats
  - S5P atmospheric (3x3x1 each) -> per-stream SCALAR (patch mean, z-scored) -> MLP -> 32

Sachsen-Anhalt is the sealed test set, never trained on. Among the other 11 folds,
each takes a turn as validation; metrics are averaged. Targets are log-transformed
then z-scored on train stats; predictions de-standardized + exponentiated to ug/m3.

--------------------------------------------------------------------------------
CHANGES:

1. LABEL NOISE / CONTRADICTORY TARGETS  [the most likely cause]
   At 10 m/px a 120x120 patch covers 1.2 km. In cities many sensors fall inside
   one patch footprint, so we average the sensors.
   - `--noise-ceiling` measures the irreducible error directly: for sensor pairs
     within a distance threshold the true annual means are identical, so the
     RMSE between their measured values is sqrt(2)*sigma_sensor, and sigma_sensor
     is the floor no model can beat.
   - `--grid-km K` aggregates labels onto a K-km grid: one sample per cell, label
     = mean of the sensors in it, patch = sensor nearest the cell centroid.
     Fewer samples, sqrt(n) less label noise, no contradictory targets.

2. FEATURE-DIMENSION IMBALANCE
   2048 + 128 + 32 concatenated meant the NO2 scalar was 1.4% of the head input. 
    BatchNorm equalises scale, not capacity. The backbone
   is now projected to `--proj-dim` (default 128) before the concat so the three
   streams compete on comparable footing. BatchNorm1d(affine=False) is kept for
   scale balance (LayerNorm would wipe out the between-sample variation where
   haze/brightness/atmospheric-load signal lives).

3. PHOTOMETRIC AUGMENTATION WAS DESTROYING THE SIGNAL
   gain in [0.9,1.1] and bias in [-30,30] on raw reflectance randomises overall
   brightness/contrast -- which IS the aerosol/haze cue. Now +-2% / +-2 by
   default (`--aug-gain 0 --aug-bias 0` to disable entirely).

4. VISIBILITY
   Train RMSE/R2 are now logged every epoch (previously only train loss, so
   underfitting and non-generalisation were indistinguishable), plus
   pred.std()/target.std() on val -- a ratio < ~0.3 is mean-collapse.
   A closed-form NO2-scalar-only linear baseline is printed per fold: if that
   beats the CNN, the imagery is contributing nothing.
   Per-fold predictions are saved for a predicted-vs-true scatter, and R2 is
   also reported POOLED across folds. Per-fold R2 is brutal by construction:
   within one Land the target variance is far below the national variance, so
   R2 can sit near zero while the model has the national gradient right.

5. `--underfit-test`: one fold, no augmentation, no dropout, no weight decay,
   layer4 unfrozen. If train R2 still cannot exceed ~0.5 there, the imagery
   carries little annual-PM information and no regularisation tuning helps.

Kept from before: --wd actually reaches the optimizer; the nodata mask is
captured BEFORE gain/bias augmentation; backbone frozen except layer4 at a much
lower LR than the head (fully unfrozen at one LR overfit ~4k samples instantly);
BatchNorm1d needs batch>1 in train mode, hence drop_last=True.
--------------------------------------------------------------------------------
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC = BASE_DIR / "data" / "processed"
SAT_DIR = PROC / "satellite"
FOLD_DIR = PROC / "corrected" / "fold"
HIGH_RES = SAT_DIR / "high_res_multispec"
LOW_RES = SAT_DIR / "low_res_multispec"
SENSOR_LAND = PROC / "sensor_land.csv"

# S5P atmospheric streams. Add "aer_tropomi","co_tropomi" once those downloads
# finish; dataset, stats and model input dim all adapt automatically.
S5P_STREAMS = ["no2_tropomi"]
S5P_DIRS = {name: SAT_DIR / name for name in S5P_STREAMS}

TEST_LAND = "Sachsen-Anhalt"
TARGETS = ["PM10_corrected"]
LOG_TARGET = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAT_CANDIDATES = ["lat", "latitude", "Latitude", "LAT"]
LON_CANDIDATES = ["lon", "lng", "longitude", "Longitude", "LON"]

LAND_TO_FOLD = {
    "Berlin": "Berlin-Brandenburg", "Brandenburg": "Berlin-Brandenburg",
    "Bremen": "Bremen-Niedersachsen", "Niedersachsen": "Bremen-Niedersachsen",
    "Hamburg": "Hamburg-Schleswig-Holstein", "Schleswig-Holstein": "Hamburg-Schleswig-Holstein",
    "Saarland": "Saarland-Rheinland-Pfalz", "Rheinland-Pfalz": "Saarland-Rheinland-Pfalz",
    "Baden-Wuerttemberg": "Baden-Wuerttemberg", "Bayern": "Bayern", "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen", "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt", "Thueringen": "Thueringen",
}

MEAN = np.array([438.3721, 614.0557, 588.4096, 942.8433, 1769.9316,
                 2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325], dtype="float32").reshape(1, 1, -1)
STD = np.array([607.0269, 603.2968, 684.5688, 738.4327, 1100.4561,
                1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276], dtype="float32").reshape(1, 1, -1)

LAT0 = 51.0                      # Germany centroid, for the local equirectangular projection
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(LAT0))


# ----------------------------------------------------------------------------- utils

def _canon_loc(loc):
    return f"{float(loc):.1f}"


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_km_xy(lat, lon):
    """Local equirectangular projection. Accurate to well under 1% over Germany,
    which is far below the grid sizes and distance thresholds used here."""
    return lon * KM_PER_DEG_LON, lat * KM_PER_DEG_LAT


def normalize_patch(arr, mask):
    """mask: True where nodata/invalid, computed by the caller BEFORE any
    augmentation touches raw pixel values."""
    safe = np.where(mask, 0.0, arr).astype("float32")
    normed = (safe - MEAN) / STD
    normed = np.where(mask, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


def compute_s5p_stats(frame):
    """Per-stream mean/std of the S5P patch-mean over the TRAIN frame, for
    z-scoring. Train sensors only -> no leakage."""
    stats = {}
    for st in S5P_STREAMS:
        vals = []
        for loc in frame["loc_str"]:
            f = S5P_DIRS[st] / f"{loc}.npy"
            if f.exists():
                vals.append(float(np.nanmean(np.load(f).astype("float32"))))
        vals = np.array(vals, dtype="float32")
        stats[st] = (float(vals.mean()) if len(vals) else 0.0,
                     float(vals.std()) if len(vals) else 1.0)
    return stats


def s5p_scalar_table(frame):
    """One z-scoreable scalar per stream per row, materialised once so the
    linear NO2 baseline and the density diagnostics do not re-read the patches."""
    out = {}
    for st in S5P_STREAMS:
        vals = []
        for loc in frame["loc_str"]:
            f = S5P_DIRS[st] / f"{loc}.npy"
            vals.append(float(np.nanmean(np.load(f).astype("float32"))) if f.exists() else np.nan)
        out[st] = np.array(vals, dtype="float64")
    return out


# ------------------------------------------------------------------- diagnostics

def _pairwise_within(xy, max_km, chunk=512):
    """Yield (i, j, dist_km) for all i<j closer than max_km. Brute force in
    chunks -- ~5k points is 13M pairs, a couple of seconds, no scipy needed."""
    n = len(xy)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = np.sqrt(((xy[start:stop, None, :] - xy[None, :, :]) ** 2).sum(-1))
        rows, cols = np.where(d <= max_km)          # rows are chunk-local
        keep = (rows + start) < cols                # upper triangle in global indices
        yield rows[keep] + start, cols[keep], d[rows[keep], cols[keep]]


def noise_ceiling_report(frame, thresholds=(0.5, 1.0, 2.0, 5.0)):
    """Co-located sensor pairs measure (near) the same true annual mean, so the
    RMSE between them is sqrt(2)*sigma_sensor. sigma_sensor is the floor on any
    model's RMSE. Reported in ug/m3."""
    print("\n=== NOISE CEILING: co-located sensor pairs ===")
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")
    y_ug = np.exp(y) if LOG_TARGET else y

    max_km = max(thresholds)
    all_i, all_j, all_d = [], [], []
    for i, j, d in _pairwise_within(xy, max_km):
        all_i.append(i); all_j.append(j); all_d.append(d)
    if not all_i:
        print("  no pairs found within the largest threshold -- check coordinates")
        return {}
    I = np.concatenate(all_i); J = np.concatenate(all_j); D = np.concatenate(all_d)

    report = {}
    print(f"  {'radius':>8} {'n_pairs':>9} {'pair RMSE':>10} {'sigma_sensor':>13}")
    for t in thresholds:
        m = D <= t
        if m.sum() < 10:
            print(f"  {t:>7.1f}km {int(m.sum()):>9} {'-':>10} {'(too few pairs)':>13}")
            continue
        diff = y_ug[I[m]] - y_ug[J[m]]
        pair_rmse = float(np.sqrt(np.mean(diff ** 2)))
        sigma = pair_rmse / np.sqrt(2.0)
        report[f"{t}km"] = {"n_pairs": int(m.sum()), "pair_rmse": pair_rmse, "sigma_sensor": sigma}
        print(f"  {t:>7.1f}km {int(m.sum()):>9} {pair_rmse:>10.2f} {sigma:>13.2f}")

    # nearest-neighbour distances: how much of the data is inside one patch footprint
    nn = np.full(len(xy), np.inf)
    np.minimum.at(nn, I, D); np.minimum.at(nn, J, D)
    finite = nn[np.isfinite(nn)]
    frac_1km = float((nn <= 1.0).mean())
    print(f"\n  nearest-neighbour distance (of the {len(finite)} sensors with a neighbour "
          f"<= {max_km:.0f} km):")
    if len(finite):
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3d} {np.percentile(finite, q):.2f} km")
    print(f"  fraction of ALL sensors with a neighbour within 1 km "
          f"(i.e. inside one 1.2 km patch footprint): {frac_1km:.1%}")
    print("  -> if sigma_sensor is close to your model RMSE, you are at the label-noise")
    print("     ceiling and architecture changes cannot help. If frac(<1km) is large,")
    print("     run with --grid-km 2.\n")
    report["frac_neighbour_within_1km"] = frac_1km
    return report


def linear_s5p_baseline(train_df, val_df, s5p_tr, s5p_va):
    """Closed-form OLS of the target on the S5P scalars alone. If this is
    competitive with the CNN, the imagery streams are contributing nothing."""
    Xtr = np.stack([s5p_tr[st] for st in S5P_STREAMS], axis=1)
    Xva = np.stack([s5p_va[st] for st in S5P_STREAMS], axis=1)
    ytr = train_df[TARGETS[0]].values.astype("float64")
    yva = val_df[TARGETS[0]].values.astype("float64")
    ok_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    ok_va = np.isfinite(Xva).all(1) & np.isfinite(yva)
    if ok_tr.sum() < 10 or ok_va.sum() < 10:
        return None
    A = np.c_[Xtr[ok_tr], np.ones(ok_tr.sum())]
    coef, *_ = np.linalg.lstsq(A, ytr[ok_tr], rcond=None)
    pred = np.c_[Xva[ok_va], np.ones(ok_va.sum())] @ coef
    p, t = (np.exp(pred), np.exp(yva[ok_va])) if LOG_TARGET else (pred, yva[ok_va])
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    sst = np.sum((t - t.mean()) ** 2)
    r2 = float(1 - np.sum((t - p) ** 2) / sst) if sst > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


# ------------------------------------------------------------------ label frames

def aggregate_to_grid(frame, km):
    """One sample per km-grid cell. Label = mean of the (log) targets of the
    sensors in the cell -> sqrt(n) less label noise and, crucially, no two
    near-identical patches carrying different targets. The representative patch
    is the sensor nearest the cell centroid."""
    x, y = to_km_xy(frame["lat"].values, frame["lon"].values)
    cx = np.floor(x / km).astype(int)
    cy = np.floor(y / km).astype(int)
    f = frame.copy()
    f["_x"], f["_y"] = x, y
    f["_cell"] = [f"{a}_{b}" for a, b in zip(cx, cy)]

    rows = []
    for cell, g in f.groupby("_cell"):
        gx, gy = g["_x"].mean(), g["_y"].mean()
        d = np.sqrt((g["_x"].values - gx) ** 2 + (g["_y"].values - gy) ** 2)
        rep = g.iloc[int(np.argmin(d))].copy()
        for t in TARGETS:
            rep[t] = g[t].mean()          # mean in LOG space (targets already logged)
        rep["n_sensors"] = len(g)
        rep["cell"] = cell
        rows.append(rep)
    out = pd.DataFrame(rows).reset_index(drop=True)
    dup = (out["n_sensors"] > 1).sum()
    print(f"  grid {km} km: {len(frame)} sensors -> {len(out)} cells "
          f"({dup} cells merge >1 sensor, max {int(out['n_sensors'].max())})")
    return out


def remove_outlier_sensors(frame, radius_km=2.0, mad_mult=2.5, min_neighbors=2):
    """QC on MEASUREMENTS only (no model involved, so applying it to the whole
    non-test frame is not leakage): each sensor's deviation from the median of
    its neighbours within radius_km. Deviations beyond mad_mult robust-sigmas
    (1.4826*MAD) are miscalibrated/dying units -- exactly the heavy tail that
    inflates the noise ceiling. Sensors with < min_neighbors neighbours are kept
    (cannot be judged), so isolated rural sensors are never dropped by this."""
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")   # log space
    nbrs = [[] for _ in range(len(xy))]
    for i, j, _ in _pairwise_within(xy, radius_km):
        for a, b in zip(i, j):
            nbrs[a].append(b); nbrs[b].append(a)
    dev = np.full(len(xy), np.nan)
    for k, nb in enumerate(nbrs):
        if len(nb) >= min_neighbors:
            dev[k] = y[k] - np.median(y[nb])
    have = np.isfinite(dev)
    if have.sum() < 30:
        print("  outlier filter: too few sensors with neighbours, skipped")
        return frame.reset_index(drop=True)
    mad = np.median(np.abs(dev[have] - np.median(dev[have])))
    sigma = 1.4826 * mad
    bad = have & (np.abs(dev) > mad_mult * sigma)
    print(f"  outlier filter (r={radius_km:g} km, {mad_mult:g}x robust sigma={sigma:.3f} log-units): "
          f"{int(have.sum())}/{len(xy)} judgeable, {int(bad.sum())} dropped")
    return frame[~bad].reset_index(drop=True)


def smooth_labels(frame, radius_km=10.0, d0_km=1.0):
    """Redefine the target as the LOCAL ANNUAL FIELD: inverse-distance-weighted
    mean (w = 1/(d + d0), self included at d=0) of all sensors within radius_km
    IN THE SAME SPLIT. Per-sensor offset noise is zero-mean and independent, so
    a k-sensor average cuts it ~sqrt(k); the ~10 km-scale spatial signal is
    preserved. Poor-man's kriging.

    Leakage: called separately on the train and val frames, so a train label
    near a Land border never averages in a validation-Land measurement and vice
    versa. Sensors with no neighbour in range keep their own raw label (k=1) --
    isolated rural sites are not discarded, just less denoised. Averaging is in
    LOG space (targets already logged), consistent with grid aggregation."""
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")
    w0 = 1.0 / d0_km
    wsum = np.full(len(xy), w0)                      # self, d=0
    ysum = y * w0
    kcnt = np.ones(len(xy), dtype=int)
    for i, j, d in _pairwise_within(xy, radius_km):
        w = 1.0 / (d + d0_km)
        np.add.at(wsum, i, w); np.add.at(ysum, i, w * y[j]); np.add.at(kcnt, i, 1)
        np.add.at(wsum, j, w); np.add.at(ysum, j, w * y[i]); np.add.at(kcnt, j, 1)
    out = frame.copy()
    out[TARGETS[0]] = ysum / wsum
    out["k_smooth"] = kcnt
    q = np.percentile(kcnt, [10, 50, 90])
    print(f"  label smoothing (r={radius_km:g} km, IDW d0={d0_km:g} km): "
          f"k per sensor p10/p50/p90 = {int(q[0])}/{int(q[1])}/{int(q[2])}, "
          f"{int((kcnt == 1).sum())} self-only (raw label kept)")
    return out


def local_density(frame, radius_km=5.0):
    """Number of sensors within radius_km of each sensor (including itself)."""
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    dens = np.ones(len(xy))
    for i, j, _ in _pairwise_within(xy, radius_km):
        np.add.at(dens, i, 1.0)
        np.add.at(dens, j, 1.0)
    return dens


def attach_weights(frame, mode, density_radius=5.0):
    """Per-sample loss weight, normalised to mean 1.
      none    : uniform
      density : inverse local sensor density -- stops urban clusters dominating
      count   : number of sensors behind a grid cell (only with --grid-km)"""
    f = frame.copy()
    if mode == "none":
        f["w"] = 1.0
    elif mode == "density":
        d = local_density(f, density_radius)
        w = 1.0 / d
        f["w"] = w / w.mean()
    elif mode == "count":
        if "n_sensors" not in f.columns:
            raise ValueError("--weight-mode count requires --grid-km")
        w = f["n_sensors"].values.astype("float64")
        f["w"] = w / w.mean()
    else:
        raise ValueError(mode)
    return f


def patches_exist(loc):
    if not ((HIGH_RES / f"{loc}.npy").exists() and (LOW_RES / f"{loc}.npy").exists()):
        return False
    return all((S5P_DIRS[st] / f"{loc}.npy").exists() for st in S5P_STREAMS)


def filter_to_available(frame):
    """Drop rows without all patches on disk BEFORE grid aggregation, so a cell
    representative is always a sensor we can actually load."""
    keep = frame["loc_str"].map(patches_exist).values
    if (~keep).sum():
        print(f"  {int((~keep).sum())} of {len(frame)} sensors missing patches, dropped")
    return frame[keep].reset_index(drop=True)


def build_labeled_frame(require_coords):
    sl = pd.read_csv(SENSOR_LAND)
    lat_col, lon_col = _find_col(sl, LAT_CANDIDATES), _find_col(sl, LON_CANDIDATES)
    cols = ["location", "land"] + ([lat_col, lon_col] if lat_col and lon_col else [])
    sl = sl[cols].rename(columns={lat_col: "lat", lon_col: "lon"} if lat_col else {})
    sl = sl[sl["land"] != TEST_LAND]
    sl["fold"] = sl["land"].map(LAND_TO_FOLD)
    dropped = sl[sl["fold"].isna()]
    if len(dropped):
        print(f"WARNING: {len(dropped)} sensors have a 'land' not in LAND_TO_FOLD: "
              f"{sorted(dropped['land'].unique())}")
    sl = sl.dropna(subset=["fold"])

    sl_has_coords = "lat" in sl.columns
    keep_cols = ["location", "land", "fold"] + (["lat", "lon"] if sl_has_coords else [])
    frames = []
    for fold in sorted(sl["fold"].unique()):
        f = FOLD_DIR / fold.replace(" ", "_") / "annual" / "2024.csv"
        if not f.exists():
            print(f"WARNING: missing {f}, skipping fold {fold}")
            continue
        df = pd.read_csv(f)
        # the calibration script's apply_fold merges node lat/lon into these CSVs;
        # use them so no separate coordinate file is needed
        fold_lat, fold_lon = _find_col(df, LAT_CANDIDATES), _find_col(df, LON_CANDIDATES)
        take = ["location"] + TARGETS
        if not sl_has_coords and fold_lat and fold_lon:
            take += [fold_lat, fold_lon]
        df = df[take].rename(columns={fold_lat: "lat", fold_lon: "lon"}
                             if (not sl_has_coords and fold_lat) else {})
        frames.append(sl[sl["fold"] == fold][keep_cols].merge(df, on="location", how="inner"))
    out = pd.concat(frames, ignore_index=True).dropna(subset=TARGETS)
    if require_coords and "lat" not in out.columns:
        raise SystemExit(
            f"\nERROR: no coordinates found in {SENSOR_LAND} or the fold annual CSVs.\n"
            f"  --grid-km, --noise-ceiling and --weight-mode density need lat/lon.\n"
            f"  Pass --coords-csv PATH to a CSV with location,lat,lon.\n")
    if "lat" in out.columns:
        n_nan = out["lat"].isna().sum()
        if n_nan and require_coords:
            print(f"WARNING: {n_nan} sensors without coordinates dropped")
            out = out.dropna(subset=["lat", "lon"])
    if LOG_TARGET:
        for t in TARGETS:
            out = out[out[t] > 0]
            out[t] = np.log(out[t].values)
    out["loc_str"] = out["location"].map(_canon_loc)
    return out.reset_index(drop=True)


def merge_external_coords(frame, path):
    c = pd.read_csv(path)
    lat_col, lon_col = _find_col(c, LAT_CANDIDATES), _find_col(c, LON_CANDIDATES)
    if not (lat_col and lon_col and "location" in c.columns):
        raise SystemExit(f"ERROR: {path} needs location + lat/lon columns, found {list(c.columns)}")
    c = c[["location", lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"})
    out = frame.drop(columns=[x for x in ("lat", "lon") if x in frame.columns])
    out = out.merge(c, on="location", how="left")
    missing = out["lat"].isna().sum()
    if missing:
        print(f"WARNING: {missing} sensors have no coordinates in {path}, dropped")
        out = out.dropna(subset=["lat", "lon"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------- dataset

class PatchDataset(Dataset):
    def __init__(self, frame, tmean, tstd, s5p_stats, augment=False,
                 aug_gain=0.02, aug_bias=2.0):
        self.augment = augment
        self.aug_gain = aug_gain
        self.aug_bias = aug_bias
        self.s5p_stats = s5p_stats
        self.frame = frame.reset_index(drop=True)
        if "w" not in self.frame.columns:
            self.frame["w"] = 1.0
        self.tmean, self.tstd = tmean, tstd

    def __len__(self):
        return len(self.frame)

    def _aug_arr(self, arr, k, f0, f1, gain, bias, sh):
        if k: arr = np.rot90(arr, k, axes=(0, 1)).copy()
        if f0: arr = arr[::-1].copy()
        if f1: arr = arr[:, ::-1].copy()
        if gain is not None:
            arr = (arr * gain + bias).astype("float32")
        if sh is not None:
            arr = np.roll(arr, sh, axis=(0, 1)).copy()
        return arr

    def _aug_mask(self, mask, k, f0, f1, sh):
        # same geometric ops as _aug_arr, but NEVER gain/bias (masks are boolean)
        m = mask.astype("float32")
        if k: m = np.rot90(m, k, axes=(0, 1)).copy()
        if f0: m = m[::-1].copy()
        if f1: m = m[:, ::-1].copy()
        if sh is not None:
            m = np.roll(m, sh, axis=(0, 1)).copy()
        return m > 0.5

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        loc = r["loc_str"]
        hi = np.load(HIGH_RES / f"{loc}.npy")
        lo = np.load(LOW_RES / f"{loc}.npy")
        # capture nodata BEFORE augmentation can jitter a true zero into a fake value
        hi_mask = ~np.isfinite(hi) | (hi <= 0)
        lo_mask = ~np.isfinite(lo) | (lo <= 0)
        if self.augment:
            k = np.random.randint(4)
            f0 = np.random.rand() < 0.5
            f1 = np.random.rand() < 0.5
            # Photometric jitter is now tiny: overall brightness/contrast is the
            # aerosol cue, so large gain/bias randomises away the label signal.
            if (self.aug_gain > 0 or self.aug_bias > 0) and np.random.rand() < 0.5:
                gain = np.random.uniform(1 - self.aug_gain, 1 + self.aug_gain)
                bias = np.random.uniform(-self.aug_bias, self.aug_bias)
            else:
                gain = bias = None
            sh = np.random.randint(-6, 7, size=2)
            hi = self._aug_arr(hi, k, f0, f1, gain, bias, sh)
            lo = self._aug_arr(lo, k, f0, f1, gain, bias, sh)
            hi_mask = self._aug_mask(hi_mask, k, f0, f1, sh)
            lo_mask = self._aug_mask(lo_mask, k, f0, f1, sh)
        xh = torch.from_numpy(normalize_patch(hi, hi_mask))
        xl = torch.from_numpy(normalize_patch(lo, lo_mask))
        s5p = []
        for st in S5P_STREAMS:
            v = float(np.nanmean(np.load(S5P_DIRS[st] / f"{loc}.npy").astype("float32")))
            m, sd = self.s5p_stats[st]
            s5p.append((v - m) / sd if sd > 0 else 0.0)
        xs = torch.tensor(s5p, dtype=torch.float32)
        y = ((r[TARGETS].values.astype("float32") - self.tmean) / self.tstd).astype("float32")
        w = torch.tensor(float(r["w"]), dtype=torch.float32)
        return xh, xl, xs, torch.from_numpy(y), w


# ------------------------------------------------------------------------ model

class ResNetRegressor(nn.Module):
    def __init__(self, n_out=1, n_s5p=1, head_dropout=0.4, proj_dim=128):
        super().__init__()
        self.backbone, feat = self._load_backbone()
        # Project 2048 -> proj_dim BEFORE the concat. Without this the S5P scalar
        # is ~1.4% of the head input and is drowned out; BatchNorm equalises
        # scale but not capacity.
        self.proj_h = nn.Linear(feat, proj_dim)
        self.low_cnn = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(True))
        low_feat = 128
        self.s5p_mlp = nn.Sequential(
            nn.Linear(n_s5p, 32), nn.ReLU(True), nn.Linear(32, 32), nn.ReLU(True))
        s5p_feat = 32
        # BatchNorm1d(affine=False) normalises each FEATURE across the batch:
        # balances stream scales while preserving between-sample variation
        # (LayerNorm normalises each sample's own vector and destroys exactly
        # the haze/brightness signal we want).
        self.norm_h = nn.BatchNorm1d(proj_dim, affine=False)
        self.norm_l = nn.BatchNorm1d(low_feat, affine=False)
        self.norm_s = nn.BatchNorm1d(s5p_feat, affine=False)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout), nn.Linear(proj_dim + low_feat + s5p_feat, 256), nn.ReLU(True),
            nn.Dropout(head_dropout), nn.Linear(256, n_out))

    def _load_backbone(self):
        import timm
        core = timm.create_model("resnet50", pretrained=False, in_chans=10, num_classes=0)
        feat = core.num_features
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            repo = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
            path = hf_hub_download(repo_id=repo, filename="model.safetensors")
            ckpt = load_file(path)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            core_sd = core.state_dict()
            remapped, matched = {}, 0
            for ck, cv in ckpt.items():
                base = ck
                for pref in ("model.vision_encoder.", "vision_encoder.", "model.", "backbone.",
                             "network.", "encoder.", "resnet.", "timm_model."):
                    if base.startswith(pref):
                        base = base[len(pref):]
                if base in core_sd and core_sd[base].shape == cv.shape:
                    remapped[base] = cv; matched += 1
            info = core.load_state_dict(remapped, strict=False)
            print(f"loaded {matched}/{len(core_sd)} pretrained tensors into timm ResNet50 "
                  f"({len(info.missing_keys)} left at init)")
            if matched < 150:
                print(f"  WARNING: few tensors matched. Sample: {list(ckpt.keys())[:6]}")
        except Exception as e:
            print(f"WARNING: pretrained load failed ({type(e).__name__}: {e}); RANDOM init")
        return core, feat

    def forward(self, xh, xl, xs):
        fh = self.norm_h(self.proj_h(self.backbone(xh)))
        fl = self.norm_l(self.low_cnn(xl))
        fs = self.norm_s(self.s5p_mlp(xs))
        return self.head(torch.cat([fh, fl, fs], dim=1))

    def set_backbone_trainable(self, trainable, last_block_only=False):
        for p in self.backbone.parameters(): p.requires_grad = False
        if not trainable: return
        if last_block_only and hasattr(self.backbone, "layer4"):
            for p in self.backbone.layer4.parameters(): p.requires_grad = True
        else:
            for p in self.backbone.parameters(): p.requires_grad = True


# ------------------------------------------------------------------- train/eval

@torch.no_grad()
def evaluate(model, loader, tmean, tstd, tta=True, return_preds=False):
    model.eval(); P, T = [], []
    for xh, xl, xs, yb, _ in loader:
        xh, xl, xs = xh.to(DEVICE), xl.to(DEVICE), xs.to(DEVICE)
        if tta:
            views = []
            for k in range(4):
                rh = torch.rot90(xh, k, dims=(2, 3)); rl = torch.rot90(xl, k, dims=(2, 3))
                views.append(model(rh, rl, xs))
                views.append(model(torch.flip(rh, dims=(3,)), torch.flip(rl, dims=(3,)), xs))
            out = torch.stack(views).mean(0)
        else:
            out = model(xh, xl, xs)
        P.append(out.cpu().numpy()); T.append(yb.numpy())
    P = np.concatenate(P) * tstd + tmean; T = np.concatenate(T) * tstd + tmean
    if LOG_TARGET:
        P = np.exp(P); T = np.exp(T)
    out = {}
    for i, n in enumerate(TARGETS):
        p, t = P[:, i], T[:, i]
        rmse = float(np.sqrt(np.mean((p - t) ** 2))); mae = float(np.mean(np.abs(p - t)))
        ssr = np.sum((t - p) ** 2); sst = np.sum((t - t.mean()) ** 2)
        out[n] = {"rmse": rmse, "mae": mae,
                  "r2": float(1 - ssr / sst) if sst > 0 else float("nan"),
                  # < ~0.3 means the model is collapsing to the mean
                  "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}
    return (out, P, T) if return_preds else out


def run_epoch(model, loader, opt, lossf, freeze_backbone_bn=True):
    model.train()
    if freeze_backbone_bn:
        model.backbone.eval()
    tot, n = 0.0, 0
    for xh, xl, xs, yb, wb in loader:
        xh, xl, xs, yb, wb = (xh.to(DEVICE), xl.to(DEVICE), xs.to(DEVICE),
                              yb.to(DEVICE), wb.to(DEVICE))
        opt.zero_grad()
        per_elem = lossf(model(xh, xl, xs), yb)          # reduction='none'
        loss = (per_elem.mean(dim=1) * wb).sum() / wb.sum()
        loss.backward(); opt.step()
        tot += loss.item() * len(xh); n += len(xh)
    return tot / max(n, 1)


def fmt(m):
    return "  ".join(f"{n}: RMSE={v['rmse']:.2f} R2={v['r2']:.3f} sr={v['std_ratio']:.2f}"
                     for n, v in m.items())


def mean_rmse(m):
    return float(np.mean([v["rmse"] for v in m.values()]))


def pooled_metrics(preds, trues):
    p = np.concatenate(preds)[:, 0]; t = np.concatenate(trues)[:, 0]
    sst = np.sum((t - t.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "mae": float(np.mean(np.abs(p - t))),
            "r2": float(1 - np.sum((t - p) ** 2) / sst) if sst > 0 else float("nan"),
            "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}


def sanity_check_overfit(train_df, s5p_stats, tmean, tstd, proj_dim, n=16, steps=300):
    """Can the model memorise a tiny batch with no regularisation? If the loss
    does not go near 0, something upstream is broken (data/label alignment, a
    dead gradient path) and no amount of epoch-budget or regularisation tuning
    on the full dataset will fix it."""
    print(f"\n--- sanity check: overfitting {n} samples, full unfreeze, no reg ---")
    debug_df = train_df.sample(min(n, len(train_df)), random_state=0)
    debug_ds = PatchDataset(debug_df, tmean, tstd, s5p_stats, augment=False)
    debug_loader = DataLoader(debug_ds, batch_size=len(debug_ds), shuffle=True, drop_last=True)
    model = ResNetRegressor(len(TARGETS), n_s5p=len(S5P_STREAMS),
                            head_dropout=0.0, proj_dim=proj_dim).to(DEVICE)
    model.set_backbone_trainable(True, last_block_only=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    lossf = nn.MSELoss(reduction="none")
    for step in range(steps):
        loss = run_epoch(model, debug_loader, opt, lossf, freeze_backbone_bn=False)
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:03d}  loss={loss:.5f}")
    print("--- sanity check done: loss should be near 0 by the end ---\n")


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--head-dropout", type=float, default=0.4)
    ap.add_argument("--proj-dim", type=int, default=128,
                    help="backbone features are projected to this before the concat")
    ap.add_argument("--unfreeze-last-block", action="store_true", default=True)
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--out", default="resnet_cv_results.json")
    # data / label handling
    ap.add_argument("--grid-km", type=float, default=0.0,
                    help="aggregate labels onto a K-km grid (0 = off, try 2)")
    ap.add_argument("--smooth-km", type=float, default=0.0,
                    help="IDW label smoothing radius in km (0 = off, try 10); "
                         "applied per split, so no cross-fold measurement leakage")
    ap.add_argument("--outlier-mad", type=float, default=0.0,
                    help="drop sensors deviating > this many robust sigmas from their "
                         "<=2 km neighbour median (0 = off, try 2.5)")
    ap.add_argument("--weight-mode", choices=["none", "density", "count"], default="none")
    ap.add_argument("--density-radius", type=float, default=5.0)
    ap.add_argument("--coords-csv", default=None,
                    help="CSV with location,lat,lon if sensor_land.csv has no coordinates")
    # augmentation
    ap.add_argument("--aug-gain", type=float, default=0.02, help="0 disables photometric jitter")
    ap.add_argument("--aug-bias", type=float, default=2.0)
    # diagnostics
    ap.add_argument("--noise-ceiling", action="store_true",
                    help="measure the irreducible error from co-located sensor pairs and exit")
    ap.add_argument("--sanity-check", action="store_true")
    ap.add_argument("--underfit-test", action="store_true",
                    help="one fold, no aug / no dropout / no wd, layer4 unfrozen")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--train-metric-samples", type=int, default=1500,
                    help="subsample size for the per-epoch train RMSE/R2 (0 = off)")
    args = ap.parse_args()

    needs_coords = (bool(args.grid_km) or bool(args.smooth_km) or bool(args.outlier_mad)
                    or args.noise_ceiling or args.weight_mode == "density")
    print(f"device: {DEVICE}  |  S5P streams: {S5P_STREAMS}")

    data = build_labeled_frame(require_coords=needs_coords and not args.coords_csv)
    if args.coords_csv:
        data = merge_external_coords(data, args.coords_csv)

    if args.noise_ceiling:
        rep = noise_ceiling_report(data)
        Path("noise_ceiling.json").write_text(json.dumps(rep, indent=2))
        print("saved noise_ceiling.json")
        return

    data = filter_to_available(data)
    if args.outlier_mad:
        # pure measurement QC, applied once to the whole non-test frame BEFORE any
        # fold split -- no model, no labels-from-model, hence no leakage
        data = remove_outlier_sensors(data, mad_mult=args.outlier_mad)

    if args.underfit_test:
        args.aug_gain = args.aug_bias = 0.0
        args.head_dropout = 0.0
        args.wd = 0.0
        # The whole backbone is unfrozen for this test, but leaving lr-backbone at
        # 1e-5 means it barely moves and the "can it fit at all?" question goes
        # unanswered. Match the head LR so the test measures capacity, not LR.
        args.lr_backbone = args.lr_head
        args.epochs = min(args.epochs, 30)
        print(f"UNDERFIT TEST: no aug, dropout=0, wd=0, full backbone unfrozen at "
              f"lr={args.lr_backbone:g}, single fold")

    all_folds = sorted(data["fold"].unique())
    run_folds = args.folds or all_folds
    if args.underfit_test:
        run_folds = run_folds[:1]
    print(f"{len(data)} sensors across {len(all_folds)} folds; validating on: {run_folds}\n")

    if args.sanity_check:
        first_train = data[data["fold"] != run_folds[0]].copy()
        s0 = compute_s5p_stats(first_train)
        m0 = first_train[TARGETS].values.mean(0).astype("float32")
        s0_ = first_train[TARGETS].values.std(0).astype("float32")
        sanity_check_overfit(first_train, s0, m0, s0_, args.proj_dim)

    cv, pooled_p, pooled_t, scatter = {}, [], [], []
    for val_fold in run_folds:
        print(f"########## VALIDATION FOLD: {val_fold} ##########")
        train_df = data[data["fold"] != val_fold].copy()
        val_df = data[data["fold"] == val_fold].copy()

        if args.smooth_km:
            # per split: train labels never average in val-Land measurements
            print("  train:", end=" "); train_df = smooth_labels(train_df, args.smooth_km)
            print("  val:  ", end=" "); val_df = smooth_labels(val_df, args.smooth_km)
        if args.grid_km:
            print("  train:", end=" "); train_df = aggregate_to_grid(train_df, args.grid_km)
            print("  val:  ", end=" "); val_df = aggregate_to_grid(val_df, args.grid_km)
        train_df = attach_weights(train_df, args.weight_mode, args.density_radius)
        val_df = attach_weights(val_df, "none")

        tmean = train_df[TARGETS].values.mean(0).astype("float32")
        tstd = train_df[TARGETS].values.std(0).astype("float32")
        s5p_stats = compute_s5p_stats(train_df)   # train-only, no leakage

        tr_ds = PatchDataset(train_df, tmean, tstd, s5p_stats, augment=True,
                             aug_gain=args.aug_gain, aug_bias=args.aug_bias)
        va_ds = PatchDataset(val_df, tmean, tstd, s5p_stats)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)  # BatchNorm1d needs batch>1
        va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                               num_workers=4, pin_memory=True)
        # fixed train subsample, un-augmented, for a comparable per-epoch train metric
        tm_loader = None
        if args.train_metric_samples:
            sub = train_df.sample(min(args.train_metric_samples, len(train_df)), random_state=0)
            tm_loader = DataLoader(PatchDataset(sub, tmean, tstd, s5p_stats),
                                   batch_size=args.batch, shuffle=False, num_workers=4)

        # --- reference points -------------------------------------------------
        base_pred = train_df[TARGETS].values.mean(0)
        vy = val_df[TARGETS].values
        for i, t in enumerate(TARGETS):
            bp, vt = base_pred[i], vy[:, i]
            if LOG_TARGET:
                bp = np.exp(bp); vt = np.exp(vt)
            print(f"  BASELINE (predict train mean) {t}: RMSE="
                  f"{float(np.sqrt(np.mean((vt - bp) ** 2))):.2f} ug/m3")
        lin = linear_s5p_baseline(train_df, val_df,
                                  s5p_scalar_table(train_df), s5p_scalar_table(val_df))
        if lin:
            print(f"  BASELINE (OLS on S5P scalars only): RMSE={lin['rmse']:.2f} R2={lin['r2']:.3f}"
                  f"   <- if the CNN cannot beat this, the imagery adds nothing")

        # --- model ------------------------------------------------------------
        model = ResNetRegressor(len(TARGETS), n_s5p=len(S5P_STREAMS),
                                head_dropout=args.head_dropout, proj_dim=args.proj_dim).to(DEVICE)
        lossf = nn.SmoothL1Loss(reduction="none")

        model.set_backbone_trainable(True, last_block_only=(args.unfreeze_last_block
                                                            and not args.underfit_test))
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best, best_state, bad = np.inf, None, 0
        for ep in range(1, args.epochs + 1):
            # BN must be trainable in the underfit test: frozen pretrained running
            # stats + full-backbone training at head LR can prevent fitting outright.
            tr = run_epoch(model, tr_loader, opt, lossf,
                           freeze_backbone_bn=not args.underfit_test)
            sched.step()
            val = evaluate(model, va_loader, tmean, tstd, tta=not args.no_tta)
            trm = evaluate(model, tm_loader, tmean, tstd, tta=False) if tm_loader else None
            r = mean_rmse(val)
            flag = ""
            if r < best - 1e-4:
                best = r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0; flag = " *"
            else:
                bad += 1
            tr_str = (f"TRAIN R2={trm[TARGETS[0]]['r2']:.3f} RMSE={trm[TARGETS[0]]['rmse']:.2f}  "
                      if trm else "")
            print(f"  [{ep:02d}] loss={tr:.3f}  {tr_str}VAL {fmt(val)}{flag}")
            if bad >= args.patience:
                print(f"  early stop (no val improvement for {args.patience} epochs)")
                break

        model.load_state_dict(best_state)
        final, P, T = evaluate(model, va_loader, tmean, tstd,
                               tta=not args.no_tta, return_preds=True)
        print(f"  FINAL [{val_fold}]: {fmt(final)}\n")
        cv[val_fold] = {"metrics": final, "best_rmse": best, "n_train": len(train_df),
                        "n_val": len(val_df), "s5p_linear_baseline": lin}
        pooled_p.append(P); pooled_t.append(T)
        scatter.append(pd.DataFrame({"fold": val_fold, "pred": P[:, 0], "true": T[:, 0]}))

    print("=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    for t in TARGETS:
        rmse = np.mean([cv[f]["metrics"][t]["rmse"] for f in cv])
        r2 = np.mean([cv[f]["metrics"][t]["r2"] for f in cv])
        print(f"  {t}: mean-of-folds RMSE={rmse:.2f}  mean-of-folds R2={r2:.3f}")
    if pooled_p:
        pm = pooled_metrics(pooled_p, pooled_t)
        print(f"  POOLED across folds: RMSE={pm['rmse']:.2f} MAE={pm['mae']:.2f} "
              f"R2={pm['r2']:.3f} std_ratio={pm['std_ratio']:.2f}")
        print("  (per-fold R2 is pessimistic by construction: within-Land target variance")
        print("   is far below national variance, so R2 can be ~0 while the national")
        print("   gradient is right. Pooled R2 is the fairer headline number.)")
        cv["_pooled"] = pm
        pd.concat(scatter, ignore_index=True).to_csv("cv_predictions.csv", index=False)
        print("  saved cv_predictions.csv (for the predicted-vs-true scatter)")
    Path(args.out).write_text(json.dumps(cv, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
