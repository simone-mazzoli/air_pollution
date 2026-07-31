"""
Multi-stream CNN for annual PM10 from Sentinel-2 + Sentinel-5P, held-out-Land CV.
Streams:
  - HIGH-RES S2 (120x120x10)  -> pretrained BigEarthNet ResNet50 -> 2048 -> Linear -> PROJ
  - LOW-RES  S2 (60x60x10)    -> small from-scratch CNN          -> 128 feats
  - S5P scalars (patch mean, z-scored) -> MLP -> 32 feats  PLUS a direct skip of the
    raw z-scored scalars into the head (see below)
  - OPTIONAL wide S5P context grids (--s5p-wide): coarse NO2/AER/CO fields over
    ~100 km -> small CNN -> 64 feats. This is where the held-out Land's LEVEL
    lives; the OLS-on-scalars baseline already beats the train mean, proving the
    regional atmospheric field carries exactly the between-Land signal that
    held-out-Land CV demands.

CHANGES vs previous version:
  1. SKILL SCORE. Negative per-fold R2 compares against the val Land's own mean --
     an oracle unavailable in deployment. The deployable reference is the TRAIN
     mean; skill = 1 - RMSE_model / RMSE_trainmean is reported per fold and
     averaged. Skill > 0 is the honest "beats predicting the mean" claim (both
     recent runs already have it). Skill is also comparable across different
     --smooth-km settings, unlike raw RMSE (heavier smoothing shrinks target
     variance, deflating RMSE without the model being any better -- do NOT
     compare raw RMSE across runs with different smoothing radii).
  2. S5P SKIP CONNECTION. The raw z-scored scalars are concatenated directly
     into the head input alongside the MLP features, so the model can represent
     the linear solution (which beats the train mean on its own) without pushing
     2 numbers through BN + dropout.
  3. OVERFITTING DEFAULTS. Best val at epoch ~4 with train R2 still climbing =
     overfitting on the now-learnable target. lr-head 1e-3 -> 3e-4, patience
     20 -> 10. If the very-early-peak persists, lower --epochs too: the cosine
     schedule anneals over --epochs, so a 70-epoch schedule never cools down for
     a model that peaks at 5.
  4. WIDE S5P STREAM (--s5p-wide no2 aer). Expects <name>_tropomi_wide/<loc>.npy
     coarse grids (any HxW, e.g. 15x15 over ~100 km), one channel per stream,
     z-scored on train pixel stats. Same rot/flip TTA and augmentation as the
     S2 streams. Sensors missing a wide patch are dropped when the flag is on.

Label-quality tools (binding constraint measured at sigma_sensor ~3.3 ug/m3):
  --grid-km K / --smooth-km R / --outlier-mad M / --weight-mode reliability
Pick ONE smoothing radius for headline results (r=10 recommended); ablate
5/10/15 via skill score only.

Sachsen-Anhalt is the sealed test set, never trained on. Targets log-transformed
then z-scored on train stats; predictions de-standardized + exponentiated.
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

# resolved at runtime from --s5p / --s5p-wide
S5P_STREAMS = []
S5P_DIRS = {}
WIDE_STREAMS = []
WIDE_DIRS = {}

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
LAT0 = 51.0
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(LAT0))


def _canon_loc(loc): return f"{float(loc):.1f}"
def _find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None
def to_km_xy(lat, lon):
    return lon * KM_PER_DEG_LON, lat * KM_PER_DEG_LAT
def normalize_patch(arr, mask):
    safe = np.where(mask, 0.0, arr).astype("float32")
    normed = (safe - MEAN) / STD
    normed = np.where(mask, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


def compute_s5p_stats(frame):
    """Scalar streams: mean/std of the patch mean. Wide streams: mean/std over
    ALL pixels of the train patches. Train-only -> no leakage."""
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
    for st in WIDE_STREAMS:
        px = []
        for loc in frame["loc_str"]:
            f = WIDE_DIRS[st] / f"{loc}.npy"
            if f.exists():
                a = np.load(f).astype("float32").ravel()
                px.append(a[np.isfinite(a)])
        px = np.concatenate(px) if px else np.zeros(1, "float32")
        stats["wide:" + st] = (float(px.mean()), float(px.std()) if px.std() > 0 else 1.0)
    return stats


def s5p_scalar_table(frame):
    out = {}
    for st in S5P_STREAMS:
        vals = []
        for loc in frame["loc_str"]:
            f = S5P_DIRS[st] / f"{loc}.npy"
            vals.append(float(np.nanmean(np.load(f).astype("float32"))) if f.exists() else np.nan)
        out[st] = np.array(vals, dtype="float64")
    return out


def _pairwise_within(xy, max_km, chunk=512):
    n = len(xy)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = np.sqrt(((xy[start:stop, None, :] - xy[None, :, :]) ** 2).sum(-1))
        rows, cols = np.where(d <= max_km)
        keep = (rows + start) < cols
        yield rows[keep] + start, cols[keep], d[rows[keep], cols[keep]]


def noise_ceiling_report(frame, thresholds=(0.5, 1.0, 2.0, 5.0)):
    print("\n=== NOISE CEILING: co-located sensor pairs ===")
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")
    y_ug = np.exp(y) if LOG_TARGET else y
    all_i, all_j, all_d = [], [], []
    for i, j, d in _pairwise_within(xy, max(thresholds)):
        all_i.append(i); all_j.append(j); all_d.append(d)
    if not all_i:
        print("  no pairs found"); return {}
    I = np.concatenate(all_i); J = np.concatenate(all_j); D = np.concatenate(all_d)
    report = {}
    print(f"  {'radius':>8} {'n_pairs':>9} {'pair RMSE':>10} {'sigma_sensor':>13}")
    for t in thresholds:
        m = D <= t
        if m.sum() < 10:
            print(f"  {t:>7.1f}km {int(m.sum()):>9} {'-':>10} {'(too few)':>13}"); continue
        diff = y_ug[I[m]] - y_ug[J[m]]
        pr = float(np.sqrt(np.mean(diff ** 2)))
        report[f"{t}km"] = {"n_pairs": int(m.sum()), "pair_rmse": pr, "sigma_sensor": pr/np.sqrt(2)}
        print(f"  {t:>7.1f}km {int(m.sum()):>9} {pr:>10.2f} {pr/np.sqrt(2):>13.2f}")
    return report


def linear_s5p_baseline(train_df, val_df, s5p_tr, s5p_va):
    Xtr = np.stack([s5p_tr[st] for st in S5P_STREAMS], axis=1)
    Xva = np.stack([s5p_va[st] for st in S5P_STREAMS], axis=1)
    ytr = train_df[TARGETS[0]].values.astype("float64")
    yva = val_df[TARGETS[0]].values.astype("float64")
    ok_tr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    ok_va = np.isfinite(Xva).all(1) & np.isfinite(yva)
    if ok_tr.sum() < 10 or ok_va.sum() < 10: return None
    A = np.c_[Xtr[ok_tr], np.ones(ok_tr.sum())]
    coef, *_ = np.linalg.lstsq(A, ytr[ok_tr], rcond=None)
    pred = np.c_[Xva[ok_va], np.ones(ok_va.sum())] @ coef
    p, t = (np.exp(pred), np.exp(yva[ok_va])) if LOG_TARGET else (pred, yva[ok_va])
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    sst = np.sum((t - t.mean()) ** 2)
    return {"rmse": rmse, "r2": float(1 - np.sum((t - p) ** 2) / sst) if sst > 0 else float("nan")}


def aggregate_to_grid(frame, km):
    x, y = to_km_xy(frame["lat"].values, frame["lon"].values)
    cx = np.floor(x / km).astype(int); cy = np.floor(y / km).astype(int)
    f = frame.copy(); f["_x"], f["_y"] = x, y
    f["_cell"] = [f"{a}_{b}" for a, b in zip(cx, cy)]
    rows = []
    for cell, g in f.groupby("_cell"):
        gx, gy = g["_x"].mean(), g["_y"].mean()
        d = np.sqrt((g["_x"].values - gx) ** 2 + (g["_y"].values - gy) ** 2)
        rep = g.iloc[int(np.argmin(d))].copy()
        for t in TARGETS: rep[t] = g[t].mean()
        rep["n_sensors"] = len(g)
        if "n_days_total" in g.columns:
            rep["n_days_total"] = g["n_days_total"].sum()
        rows.append(rep)
    out = pd.DataFrame(rows).reset_index(drop=True)
    dup = (out["n_sensors"] > 1).sum()
    print(f"  grid {km} km: {len(frame)} sensors -> {len(out)} cells "
          f"({dup} merge >1, max {int(out['n_sensors'].max())})")
    return out


def remove_outlier_sensors(frame, radius_km=2.0, mad_mult=2.5, min_neighbors=2):
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")
    nbrs = [[] for _ in range(len(xy))]
    for i, j, _ in _pairwise_within(xy, radius_km):
        for a, b in zip(i, j): nbrs[a].append(b); nbrs[b].append(a)
    dev = np.full(len(xy), np.nan)
    for k, nb in enumerate(nbrs):
        if len(nb) >= min_neighbors: dev[k] = y[k] - np.median(y[nb])
    have = np.isfinite(dev)
    if have.sum() < 30:
        print("  outlier filter: too few judgeable, skipped"); return frame.reset_index(drop=True)
    mad = np.median(np.abs(dev[have] - np.median(dev[have]))); sigma = 1.4826 * mad
    bad = have & (np.abs(dev) > mad_mult * sigma)
    print(f"  outlier filter (r={radius_km:g} km, {mad_mult:g}x sigma={sigma:.3f}): "
          f"{int(have.sum())}/{len(xy)} judgeable, {int(bad.sum())} dropped")
    return frame[~bad].reset_index(drop=True)


def smooth_labels(frame, radius_km=10.0, d0_km=1.0):
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    y = frame[TARGETS[0]].values.astype("float64")
    w0 = 1.0 / d0_km
    wsum = np.full(len(xy), w0); ysum = y * w0; kcnt = np.ones(len(xy), dtype=int)
    for i, j, d in _pairwise_within(xy, radius_km):
        w = 1.0 / (d + d0_km)
        np.add.at(wsum, i, w); np.add.at(ysum, i, w * y[j]); np.add.at(kcnt, i, 1)
        np.add.at(wsum, j, w); np.add.at(ysum, j, w * y[i]); np.add.at(kcnt, j, 1)
    out = frame.copy(); out[TARGETS[0]] = ysum / wsum; out["k_smooth"] = kcnt
    q = np.percentile(kcnt, [10, 50, 90])
    print(f"  smoothing (r={radius_km:g} km): k p10/50/90 = "
          f"{int(q[0])}/{int(q[1])}/{int(q[2])}, {int((kcnt==1).sum())} self-only")
    return out


def local_density(frame, radius_km=5.0):
    xy = np.stack(to_km_xy(frame["lat"].values, frame["lon"].values), axis=1)
    dens = np.ones(len(xy))
    for i, j, _ in _pairwise_within(xy, radius_km):
        np.add.at(dens, i, 1.0); np.add.at(dens, j, 1.0)
    return dens


def attach_weights(frame, mode, density_radius=5.0):
    f = frame.copy()
    if mode == "none":
        f["w"] = 1.0
    elif mode == "density":
        d = local_density(f, density_radius); w = 1.0 / d; f["w"] = w / w.mean()
    elif mode == "count":
        if "n_sensors" not in f.columns:
            raise ValueError("--weight-mode count requires --grid-km")
        w = f["n_sensors"].values.astype("float64"); f["w"] = w / w.mean()
    elif mode == "reliability":
        if "n_days_total" not in f.columns:
            raise ValueError("--weight-mode reliability needs n_days_total in the fold CSVs")
        w = np.sqrt(f["n_days_total"].values.astype("float64"))
        f["w"] = w / w.mean()
    else:
        raise ValueError(mode)
    return f


def patches_exist(loc):
    if not ((HIGH_RES / f"{loc}.npy").exists() and (LOW_RES / f"{loc}.npy").exists()):
        return False
    if not all((S5P_DIRS[st] / f"{loc}.npy").exists() for st in S5P_STREAMS):
        return False
    return all((WIDE_DIRS[st] / f"{loc}.npy").exists() for st in WIDE_STREAMS)


def filter_to_available(frame):
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
    sl = sl.dropna(subset=["fold"])
    sl_has_coords = "lat" in sl.columns
    keep_cols = ["location", "land", "fold"] + (["lat", "lon"] if sl_has_coords else [])
    frames = []
    for fold in sorted(sl["fold"].unique()):
        f = FOLD_DIR / fold.replace(" ", "_") / "annual" / "2024.csv"
        if not f.exists():
            print(f"WARNING: missing {f}"); continue
        df = pd.read_csv(f)
        fold_lat, fold_lon = _find_col(df, LAT_CANDIDATES), _find_col(df, LON_CANDIDATES)
        take = ["location"] + TARGETS
        if "n_days_total" in df.columns:
            take += ["n_days_total"]
        if not sl_has_coords and fold_lat and fold_lon:
            take += [fold_lat, fold_lon]
        df = df[take].rename(columns={fold_lat: "lat", fold_lon: "lon"}
                             if (not sl_has_coords and fold_lat) else {})
        frames.append(sl[sl["fold"] == fold][keep_cols].merge(df, on="location", how="inner"))
    out = pd.concat(frames, ignore_index=True).dropna(subset=TARGETS)
    if require_coords and "lat" not in out.columns:
        raise SystemExit("ERROR: no coordinates in sensor_land.csv or fold CSVs; pass --coords-csv")
    if "lat" in out.columns and require_coords:
        out = out.dropna(subset=["lat", "lon"])
    if LOG_TARGET:
        for t in TARGETS:
            out = out[out[t] > 0]; out[t] = np.log(out[t].values)
    out["loc_str"] = out["location"].map(_canon_loc)
    return out.reset_index(drop=True)


class PatchDataset(Dataset):
    def __init__(self, frame, tmean, tstd, s5p_stats, augment=False, aug_gain=0.02, aug_bias=2.0):
        self.augment = augment; self.aug_gain = aug_gain; self.aug_bias = aug_bias
        self.s5p_stats = s5p_stats
        self.frame = frame.reset_index(drop=True)
        if "w" not in self.frame.columns: self.frame["w"] = 1.0
        self.tmean, self.tstd = tmean, tstd

    def __len__(self): return len(self.frame)

    def _aug_arr(self, arr, k, f0, f1, gain, bias, sh):
        if k: arr = np.rot90(arr, k, axes=(0, 1)).copy()
        if f0: arr = arr[::-1].copy()
        if f1: arr = arr[:, ::-1].copy()
        if gain is not None: arr = (arr * gain + bias).astype("float32")
        if sh is not None: arr = np.roll(arr, sh, axis=(0, 1)).copy()
        return arr

    def _aug_mask(self, mask, k, f0, f1, sh):
        m = mask.astype("float32")
        if k: m = np.rot90(m, k, axes=(0, 1)).copy()
        if f0: m = m[::-1].copy()
        if f1: m = m[:, ::-1].copy()
        if sh is not None: m = np.roll(m, sh, axis=(0, 1)).copy()
        return m > 0.5

    def _load_wide(self, loc, k=0, f0=False, f1=False):
        """(n_wide, H, W) z-scored on train pixel stats, nan->0. Same rot/flip
        as the S2 streams (no shift/gain: it's a coarse geographic field)."""
        chans = []
        for st in WIDE_STREAMS:
            a = np.load(WIDE_DIRS[st] / f"{loc}.npy").astype("float32")
            m, sd = self.s5p_stats["wide:" + st]
            a = np.where(np.isfinite(a), (a - m) / sd, 0.0).astype("float32")
            if k: a = np.rot90(a, k).copy()
            if f0: a = a[::-1].copy()
            if f1: a = a[:, ::-1].copy()
            chans.append(a)
        return torch.from_numpy(np.stack(chans, axis=0))

    def __getitem__(self, i):
        r = self.frame.iloc[i]; loc = r["loc_str"]
        hi = np.load(HIGH_RES / f"{loc}.npy"); lo = np.load(LOW_RES / f"{loc}.npy")
        hi_mask = ~np.isfinite(hi) | (hi <= 0); lo_mask = ~np.isfinite(lo) | (lo <= 0)
        k = 0; f0 = f1 = False
        if self.augment:
            k = np.random.randint(4); f0 = np.random.rand() < 0.5; f1 = np.random.rand() < 0.5
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
            m, sd = self.s5p_stats[st]; s5p.append((v - m) / sd if sd > 0 else 0.0)
        xs = torch.tensor(s5p, dtype=torch.float32)
        xw = self._load_wide(loc, k, f0, f1) if WIDE_STREAMS else torch.zeros(0)
        y = ((r[TARGETS].values.astype("float32") - self.tmean) / self.tstd).astype("float32")
        w = torch.tensor(float(r["w"]), dtype=torch.float32)
        return xh, xl, xs, xw, torch.from_numpy(y), w


class ResNetRegressor(nn.Module):
    def __init__(self, n_out=1, n_s5p=1, n_wide=0, head_dropout=0.4, proj_dim=128):
        super().__init__()
        self.n_wide = n_wide
        self.backbone, feat = self._load_backbone()
        self.proj_h = nn.Linear(feat, proj_dim)
        self.low_cnn = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(True))
        low_feat = 128
        self.s5p_mlp = nn.Sequential(
            nn.Linear(n_s5p, 32), nn.ReLU(True), nn.Linear(32, 32), nn.ReLU(True))
        s5p_feat = 32
        wide_feat = 0
        if n_wide:
            self.wide_cnn = nn.Sequential(
                nn.Conv2d(n_wide, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(True))
            wide_feat = 64
            self.norm_w = nn.BatchNorm1d(wide_feat, affine=False)
        self.norm_h = nn.BatchNorm1d(proj_dim, affine=False)
        self.norm_l = nn.BatchNorm1d(low_feat, affine=False)
        self.norm_s = nn.BatchNorm1d(s5p_feat, affine=False)
        # head input includes a DIRECT skip of the raw z-scored S5P scalars: the
        # linear-on-scalars solution beats the train mean by itself, so the model
        # must be able to express it without BN/dropout in the way. The skip is
        # already z-scored on train stats -> no extra normalization.
        head_in = proj_dim + low_feat + s5p_feat + wide_feat + n_s5p
        self.head = nn.Sequential(
            nn.Dropout(head_dropout), nn.Linear(head_in, 256), nn.ReLU(True),
            nn.Dropout(head_dropout), nn.Linear(256, n_out))

    def _load_backbone(self):
        import timm
        core = timm.create_model("resnet50", pretrained=False, in_chans=10, num_classes=0)
        feat = core.num_features
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            path = hf_hub_download(repo_id="BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
                                   filename="model.safetensors")
            ckpt = load_file(path)
            if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
            core_sd = core.state_dict(); remapped, matched = {}, 0
            for ck, cv in ckpt.items():
                base = ck
                for pref in ("model.vision_encoder.", "vision_encoder.", "model.", "backbone.",
                             "network.", "encoder.", "resnet.", "timm_model."):
                    if base.startswith(pref): base = base[len(pref):]
                if base in core_sd and core_sd[base].shape == cv.shape:
                    remapped[base] = cv; matched += 1
            info = core.load_state_dict(remapped, strict=False)
            print(f"loaded {matched}/{len(core_sd)} pretrained tensors ({len(info.missing_keys)} at init)")
        except Exception as e:
            print(f"WARNING: pretrained load failed ({type(e).__name__}: {e}); RANDOM init")
        return core, feat

    def forward(self, xh, xl, xs, xw=None):
        parts = [self.norm_h(self.proj_h(self.backbone(xh))),
                 self.norm_l(self.low_cnn(xl)),
                 self.norm_s(self.s5p_mlp(xs)),
                 xs]                                    # skip connection
        if self.n_wide:
            parts.insert(3, self.norm_w(self.wide_cnn(xw)))
        return self.head(torch.cat(parts, dim=1))

    def set_backbone_trainable(self, trainable, last_block_only=False):
        for p in self.backbone.parameters(): p.requires_grad = False
        if not trainable: return
        if last_block_only and hasattr(self.backbone, "layer4"):
            for p in self.backbone.layer4.parameters(): p.requires_grad = True
        else:
            for p in self.backbone.parameters(): p.requires_grad = True


def _to_dev(xw):
    return xw.to(DEVICE) if xw.numel() else None


@torch.no_grad()
def evaluate(model, loader, tmean, tstd, tta=True, return_preds=False):
    model.eval(); P, T = [], []
    for xh, xl, xs, xw, yb, _ in loader:
        xh, xl, xs = xh.to(DEVICE), xl.to(DEVICE), xs.to(DEVICE)
        xw = _to_dev(xw)
        if tta:
            views = []
            for k in range(4):
                rh = torch.rot90(xh, k, dims=(2, 3)); rl = torch.rot90(xl, k, dims=(2, 3))
                rw = torch.rot90(xw, k, dims=(2, 3)) if xw is not None else None
                views.append(model(rh, rl, xs, rw))
                views.append(model(torch.flip(rh, dims=(3,)), torch.flip(rl, dims=(3,)), xs,
                                   torch.flip(rw, dims=(3,)) if rw is not None else None))
            out = torch.stack(views).mean(0)
        else:
            out = model(xh, xl, xs, xw)
        P.append(out.cpu().numpy()); T.append(yb.numpy())
    P = np.concatenate(P) * tstd + tmean; T = np.concatenate(T) * tstd + tmean
    if LOG_TARGET: P = np.exp(P); T = np.exp(T)
    out = {}
    for i, n in enumerate(TARGETS):
        p, t = P[:, i], T[:, i]
        rmse = float(np.sqrt(np.mean((p - t) ** 2))); mae = float(np.mean(np.abs(p - t)))
        ssr = np.sum((t - p) ** 2); sst = np.sum((t - t.mean()) ** 2)
        out[n] = {"rmse": rmse, "mae": mae, "r2": float(1 - ssr / sst) if sst > 0 else float("nan"),
                  "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}
    return (out, P, T) if return_preds else out


def run_epoch(model, loader, opt, lossf, freeze_backbone_bn=True):
    model.train()
    if freeze_backbone_bn: model.backbone.eval()
    tot, n = 0.0, 0
    for xh, xl, xs, xw, yb, wb in loader:
        xh, xl, xs, yb, wb = (xh.to(DEVICE), xl.to(DEVICE), xs.to(DEVICE),
                              yb.to(DEVICE), wb.to(DEVICE))
        xw = _to_dev(xw)
        opt.zero_grad()
        per_elem = lossf(model(xh, xl, xs, xw), yb)
        loss = (per_elem.mean(dim=1) * wb).sum() / wb.sum()
        loss.backward(); opt.step()
        tot += loss.item() * len(xh); n += len(xh)
    return tot / max(n, 1)


def fmt(m):
    return "  ".join(f"{n}: RMSE={v['rmse']:.2f} R2={v['r2']:.3f} sr={v['std_ratio']:.2f}"
                     for n, v in m.items())
def mean_rmse(m): return float(np.mean([v["rmse"] for v in m.values()]))
def metrics_from_preds(P, T):
    """Same metric dict as evaluate(), from already-de-standardized ug/m3 arrays."""
    out = {}
    for i, n in enumerate(TARGETS):
        p, t = P[:, i], T[:, i]
        rmse = float(np.sqrt(np.mean((p - t) ** 2))); mae = float(np.mean(np.abs(p - t)))
        ssr = np.sum((t - p) ** 2); sst = np.sum((t - t.mean()) ** 2)
        out[n] = {"rmse": rmse, "mae": mae, "r2": float(1 - ssr / sst) if sst > 0 else float("nan"),
                  "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}
    return out


def pooled_metrics(preds, trues):
    p = np.concatenate(preds)[:, 0]; t = np.concatenate(trues)[:, 0]
    sst = np.sum((t - t.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))), "mae": float(np.mean(np.abs(p - t))),
            "r2": float(1 - np.sum((t - p) ** 2) / sst) if sst > 0 else float("nan"),
            "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40,
                    help="also the cosine-anneal horizon; if val peaks very early, lower this")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--head-dropout", type=float, default=0.4)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=1,
                    help="train N models per fold and average their predictions; "
                         "3 is a good cost/benefit point")
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--out", default="resnet_cv_results.json")
    ap.add_argument("--s5p", nargs="+", default=["no2"],
                    help="scalar S5P streams: no2 aer co (must be fully downloaded)")
    ap.add_argument("--s5p-wide", nargs="+", default=[],
                    help="wide-context S5P streams; expects <name>_tropomi_wide/<loc>.npy grids")
    ap.add_argument("--aux-wide", nargs="+", default=[],
                    help="non-S5P wide streams by folder base name, e.g. maiac dem; "
                         "expects <name>_wide/<loc>.npy (from 05_s5p_wide.py)")
    ap.add_argument("--grid-km", type=float, default=0.0)
    ap.add_argument("--smooth-km", type=float, default=0.0)
    ap.add_argument("--outlier-mad", type=float, default=0.0)
    ap.add_argument("--weight-mode", choices=["none", "density", "count", "reliability"], default="none")
    ap.add_argument("--density-radius", type=float, default=5.0)
    ap.add_argument("--aug-gain", type=float, default=0.02)
    ap.add_argument("--aug-bias", type=float, default=2.0)
    ap.add_argument("--noise-ceiling", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--train-metric-samples", type=int, default=1500)
    args = ap.parse_args()

    global S5P_STREAMS, S5P_DIRS, WIDE_STREAMS, WIDE_DIRS
    S5P_STREAMS = [f"{s}_tropomi" for s in args.s5p]
    S5P_DIRS = {name: SAT_DIR / name for name in S5P_STREAMS}
    WIDE_STREAMS = [f"{s}_tropomi" for s in args.s5p_wide] + list(args.aux_wide)
    WIDE_DIRS = {name: SAT_DIR / f"{name}_wide" for name in WIDE_STREAMS}
    for name in S5P_STREAMS:
        n = len(list(S5P_DIRS[name].glob("*.npy"))) if S5P_DIRS[name].exists() else 0
        print(f"  S5P stream {name}: {n} patches on disk")
    for name in WIDE_STREAMS:
        n = len(list(WIDE_DIRS[name].glob("*.npy"))) if WIDE_DIRS[name].exists() else 0
        print(f"  WIDE stream {name}: {n} patches on disk"
              + ("  <- 0 patches: every sensor will be dropped!" if n == 0 else ""))

    needs_coords = (bool(args.grid_km) or bool(args.smooth_km) or bool(args.outlier_mad)
                    or args.noise_ceiling or args.weight_mode == "density")
    print(f"device: {DEVICE}  |  scalar: {S5P_STREAMS}  |  wide: {WIDE_STREAMS or '-'}")
    data = build_labeled_frame(require_coords=needs_coords)
    if args.noise_ceiling:
        rep = noise_ceiling_report(data)
        Path("noise_ceiling.json").write_text(json.dumps(rep, indent=2))
        print("saved noise_ceiling.json"); return
    data = filter_to_available(data)
    if args.outlier_mad:
        data = remove_outlier_sensors(data, mad_mult=args.outlier_mad)

    all_folds = sorted(data["fold"].unique())
    run_folds = args.folds or all_folds
    print(f"{len(data)} sensors across {len(all_folds)} folds; validating on: {run_folds}\n")

    cv, pooled_p, pooled_t, scatter = {}, [], [], []
    for val_fold in run_folds:
        print(f"########## VALIDATION FOLD: {val_fold} ##########")
        train_df = data[data["fold"] != val_fold].copy()
        val_df = data[data["fold"] == val_fold].copy()
        if args.smooth_km:
            print("  train:", end=" "); train_df = smooth_labels(train_df, args.smooth_km)
            print("  val:  ", end=" "); val_df = smooth_labels(val_df, args.smooth_km)
        if args.grid_km:
            print("  train:", end=" "); train_df = aggregate_to_grid(train_df, args.grid_km)
            print("  val:  ", end=" "); val_df = aggregate_to_grid(val_df, args.grid_km)
        train_df = attach_weights(train_df, args.weight_mode, args.density_radius)
        val_df = attach_weights(val_df, "none")
        tmean = train_df[TARGETS].values.mean(0).astype("float32")
        tstd = train_df[TARGETS].values.std(0).astype("float32")
        s5p_stats = compute_s5p_stats(train_df)
        tr_ds = PatchDataset(train_df, tmean, tstd, s5p_stats, augment=True,
                             aug_gain=args.aug_gain, aug_bias=args.aug_bias)
        va_ds = PatchDataset(val_df, tmean, tstd, s5p_stats)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4,
                               pin_memory=True)
        tm_loader = None
        if args.train_metric_samples:
            sub = train_df.sample(min(args.train_metric_samples, len(train_df)), random_state=0)
            tm_loader = DataLoader(PatchDataset(sub, tmean, tstd, s5p_stats),
                                   batch_size=args.batch, shuffle=False, num_workers=4)

        # baseline: predict the TRAIN mean (the deployable reference for skill)
        base_pred = train_df[TARGETS].values.mean(0); vy = val_df[TARGETS].values
        bp, vt = base_pred[0], vy[:, 0]
        if LOG_TARGET: bp = np.exp(bp); vt = np.exp(vt)
        base_rmse = float(np.sqrt(np.mean((vt - bp) ** 2)))
        print(f"  BASELINE (predict train mean): RMSE={base_rmse:.2f} ug/m3")
        lin = linear_s5p_baseline(train_df, val_df, s5p_scalar_table(train_df),
                                  s5p_scalar_table(val_df))
        if lin:
            print(f"  BASELINE (OLS on S5P scalars): RMSE={lin['rmse']:.2f} R2={lin['r2']:.3f}"
                  f"   <- CNN must beat this or imagery adds nothing")

        seed_P, T = [], None
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            if args.seeds > 1:
                print(f"  ----- seed {seed + 1}/{args.seeds} -----")
            model = ResNetRegressor(len(TARGETS), n_s5p=len(S5P_STREAMS), n_wide=len(WIDE_STREAMS),
                                    head_dropout=args.head_dropout, proj_dim=args.proj_dim).to(DEVICE)
            lossf = nn.SmoothL1Loss(reduction="none")
            model.set_backbone_trainable(True, last_block_only=True)
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
            opt = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr_backbone},
                {"params": head_params, "lr": args.lr_head},
            ], weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            best, best_state, bad = np.inf, None, 0
            for ep in range(1, args.epochs + 1):
                tr = run_epoch(model, tr_loader, opt, lossf)
                sched.step()
                val = evaluate(model, va_loader, tmean, tstd, tta=not args.no_tta)
                trm = evaluate(model, tm_loader, tmean, tstd, tta=False) if tm_loader else None
                r = mean_rmse(val); flag = ""
                if r < best - 1e-4:
                    best = r; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0; flag = " *"
                else:
                    bad += 1
                tr_str = (f"TRAIN R2={trm[TARGETS[0]]['r2']:.3f} RMSE={trm[TARGETS[0]]['rmse']:.2f}  "
                          if trm else "")
                skill = 1.0 - r / base_rmse
                print(f"  [{ep:02d}] loss={tr:.3f}  {tr_str}VAL {fmt(val)} skill={skill:+.3f}{flag}")
                if bad >= args.patience:
                    print(f"  early stop"); break
            model.load_state_dict(best_state)
            sfinal, P, T = evaluate(model, va_loader, tmean, tstd, tta=not args.no_tta,
                                    return_preds=True)
            seed_P.append(P)
            if args.seeds > 1:
                print(f"  seed {seed + 1}: {fmt(sfinal)} "
                      f"skill={1.0 - sfinal[TARGETS[0]]['rmse'] / base_rmse:+.3f}")
        # ensemble = mean prediction across seeds (in ug/m3 space; P is already
        # de-standardized and exponentiated by evaluate)
        P = np.mean(seed_P, axis=0)
        final = metrics_from_preds(P, T)
        skill = 1.0 - final[TARGETS[0]]["rmse"] / base_rmse
        tag = f"ENSEMBLE of {args.seeds} seeds" if args.seeds > 1 else "FINAL"
        print(f"  {tag} [{val_fold}]: {fmt(final)}  SKILL vs train-mean = {skill:+.3f}\n")
        cv[val_fold] = {"metrics": final, "baseline_rmse": base_rmse,
                        "skill": skill, "n_seeds": args.seeds,
                        "n_train": len(train_df), "n_val": len(val_df),
                        "s5p_linear_baseline": lin}
        pooled_p.append(P); pooled_t.append(T)
        scatter.append(pd.DataFrame({"fold": val_fold, "pred": P[:, 0], "true": T[:, 0]}))

    print("=" * 70); print("CROSS-VALIDATION SUMMARY")
    folds_done = [f for f in cv if not f.startswith("_")]
    for t in TARGETS:
        rmse = np.mean([cv[f]["metrics"][t]["rmse"] for f in folds_done])
        r2 = np.mean([cv[f]["metrics"][t]["r2"] for f in folds_done])
        print(f"  {t}: mean-of-folds RMSE={rmse:.2f}  mean-of-folds R2={r2:.3f}")
    skills = [cv[f]["skill"] for f in folds_done]
    print(f"  SKILL vs train-mean: mean={np.mean(skills):+.3f}  "
          f"min={np.min(skills):+.3f}  max={np.max(skills):+.3f}  "
          f"positive in {sum(s > 0 for s in skills)}/{len(skills)} folds")
    print("  (skill = 1 - RMSE_model/RMSE_trainmean; comparable across smoothing radii,")
    print("   unlike raw RMSE. This is the honest 'beats predicting the mean' number.)")
    if pooled_p:
        pm = pooled_metrics(pooled_p, pooled_t)
        print(f"  POOLED: RMSE={pm['rmse']:.2f} MAE={pm['mae']:.2f} R2={pm['r2']:.3f} "
              f"sr={pm['std_ratio']:.2f}")
        cv["_pooled"] = pm
        pd.concat(scatter, ignore_index=True).to_csv("cv_predictions.csv", index=False)
    Path(args.out).write_text(json.dumps(cv, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
