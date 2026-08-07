import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import folds, paths
from .config import CUDNN_DETERMINISTIC, MEAN, RELIEF_SCALE, SEED, STD, DISPLAY


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def load_s2(path):
    arr = np.load(path)
    nodata = ~np.isfinite(arr) | (arr <= 0)
    safe = np.where(nodata, 0.0, arr).astype("float32")
    normed = (safe - MEAN) / STD
    normed = np.where(nodata, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


def load_patch_raw(path):
    arr = np.load(path).astype("float32")
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return arr


def load_dem(path):
    raw = load_patch_raw(path)
    finite = raw[np.isfinite(raw)]
    fallback = float(finite.mean()) if len(finite) else 0.0
    h, w = raw.shape
    cb = raw[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1]
    cbf = cb[np.isfinite(cb)]
    center = float(cbf.mean()) if len(cbf) else fallback
    relief = np.where(np.isfinite(raw), (raw - center) / RELIEF_SCALE, 0.0).astype("float32")
    return relief[None], center


def buffer_exclude(train_df, val_df, buffer_km):
    if len(train_df) == 0 or len(val_df) == 0:
        return train_df
    r = 6371.0
    tlat = np.radians(train_df["lat"].values)[:, None]
    tlon = np.radians(train_df["lon"].values)[:, None]
    vlat = np.radians(val_df["lat"].values)[None, :]
    vlon = np.radians(val_df["lon"].values)[None, :]
    dphi = vlat - tlat
    dl = vlon - tlon
    a = np.sin(dphi / 2) ** 2 + np.cos(tlat) * np.cos(vlat) * np.sin(dl / 2) ** 2
    dist = 2 * r * np.arcsin(np.sqrt(a))
    nearest = dist.min(axis=1)
    return train_df[nearest >= buffer_km]


def _annual_labels():
    lab = pd.read_csv(paths.LABELS, dtype={"station_code": str})
    g = lab.groupby("station_code")
    ann = pd.DataFrame({"pm10": g["PM10"].mean(), "pm25": g["PM2.5"].mean()}).reset_index()
    ann["country"] = ann["station_code"].str[:2]
    return ann


def _with_fold_columns(ann, station_folds):
    cols = ["station_code", "fold", "lat", "lon"]
    if "land" in station_folds.columns:
        cols.append("land")
    sf = station_folds[cols].drop_duplicates("station_code")
    return ann.merge(sf, on="station_code", how="left")


def _apply_label_filters(ann, cfg):
    for p, cap in (("pm10", cfg["max_pm10"]), ("pm25", cfg["max_pm25"])):
        n_out = int((ann[p] > cap).sum())
        if n_out:
            print(f"  outlier filter {DISPLAY[p]}: {n_out} stations above {cap:g} ug/m3 -> label dropped")
        ann.loc[(ann[p] <= 0) | (ann[p] > cap), p] = np.nan
    return ann[ann[["pm10", "pm25"]].notna().any(axis=1)].reset_index(drop=True)


def _has_all_patches(code, streams, cfg):
    if not ((paths.HIGH / f"{code}.npy").exists() and (paths.LOW / f"{code}.npy").exists()):
        return False
    if not all((paths.SAT / st / f"{code}.npy").exists() for st in streams):
        return False
    extra = ([paths.AERW] if cfg["use_aer_wide"] else []) + ([paths.DEMD] if cfg["use_dem"] else [])
    return all((d / f"{code}.npy").exists() for d in extra)


def load_frame(streams, cfg):
    sf = folds.load_station_folds()
    ann = _with_fold_columns(_annual_labels(), sf)
    dropped = ann[ann["fold"].isna() | (ann["fold"] == "UNASSIGNED")]["country"].value_counts()
    ann.loc[ann["fold"].isin(["TEST", "UNASSIGNED"]), "fold"] = np.nan
    ann = ann.dropna(subset=["fold"]).reset_index(drop=True)
    ann = _apply_label_filters(ann, cfg)
    keep = ann["station_code"].map(lambda code: _has_all_patches(code, streams, cfg))
    print(f"{len(ann)} CV stations (after outlier + test/unlisted removal) -> "
          f"{int(keep.sum())} with all patches")
    if len(dropped):
        print(f"  dropped unlisted countries: {dict(dropped)}")
    ann = ann[keep].reset_index(drop=True)
    print("  stations per fold (total / with PM10 / with PM2.5):")
    for fold in folds.development_fold_names(sf):
        sub = ann[ann["fold"] == fold]
        print(f"    {fold:<16} {len(sub):>4}  /  {int(sub['pm10'].notna().sum()):>4}  "
              f"/  {int(sub['pm25'].notna().sum()):>4}")
    return ann


def load_test_frame(streams, cfg):
    sf = folds.load_station_folds()
    ann = _with_fold_columns(_annual_labels(), sf)
    ann = ann[ann["fold"] == "TEST"].reset_index(drop=True)
    ann["group"] = ann["station_code"].map(lambda code: "DE_ne" if code[:2] == "DE" else code[:2])
    ann = _apply_label_filters(ann, cfg)
    keep = ann["station_code"].map(lambda code: _has_all_patches(code, streams, cfg))
    print(f"{len(ann)} sealed-TEST stations -> {int(keep.sum())} with all patches")
    ann = ann[keep].reset_index(drop=True)
    for grp in sorted(ann["group"].unique()):
        sub = ann[ann["group"] == grp]
        print(f"    {grp:<6} {len(sub):>4}  (PM10 {int(sub['pm10'].notna().sum())}, "
              f"PM2.5 {int(sub['pm25'].notna().sum())})")
    return ann


def compute_s5p_stats(train_df, streams, cfg):
    dirs = {st: paths.SAT / st for st in streams}
    if cfg["use_aer_wide"]:
        dirs["aer_wide"] = paths.AERW
    stats = {}
    for key, d in dirs.items():
        pixels = []
        for code in train_df["station_code"]:
            raw = load_patch_raw(d / f"{code}.npy")
            valid = raw[np.isfinite(raw)]
            if len(valid):
                pixels.append(valid.astype("float64"))
        allv = np.concatenate(pixels) if pixels else np.array([0.0])
        stats[key] = (float(np.nanmean(allv)), float(np.nanstd(allv)) or 1.0)
    return stats


class EEA(Dataset):
    def __init__(self, frame, streams, tmean, tstd, s5p_stats, cfg, augment=False):
        self.f = frame.reset_index(drop=True)
        self.streams = streams
        self.tmean = tmean
        self.tstd = tstd
        self.s5p_stats = s5p_stats
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.f)

    def _norm_s5p(self, path, key):
        m, sd = self.s5p_stats[key]
        raw = load_patch_raw(path)
        safe = np.where(np.isfinite(raw), raw, m)
        return ((safe - m) / sd if sd > 0 else np.zeros_like(safe)).astype("float32")

    def __getitem__(self, i):
        r = self.f.iloc[i]
        code = r["station_code"]
        xh = torch.from_numpy(load_s2(paths.HIGH / f"{code}.npy"))
        xl = torch.from_numpy(load_s2(paths.LOW / f"{code}.npy"))
        chans = [self._norm_s5p(paths.SAT / st / f"{code}.npy", st) for st in self.streams]
        xs_patch = torch.from_numpy(np.stack(chans, axis=0))
        extras = []
        if self.cfg["use_aer_wide"]:
            w = self._norm_s5p(paths.AERW / f"{code}.npy", "aer_wide")
            xw = torch.from_numpy(w[None])
            extras.append(float(w[w.shape[0] // 2, w.shape[1] // 2]))
        else:
            xw = torch.zeros(1, 1, 1)
        if self.cfg["use_dem"]:
            relief, elev = load_dem(paths.DEMD / f"{code}.npy")
            xd = torch.from_numpy(relief)
            extras.append(elev / 1000.0)
        else:
            xd = torch.zeros(1, 1, 1)
        if self.augment:
            k = random.randint(0, 3)
            xh, xl, xs_patch, xw, xd = (torch.rot90(t, k, dims=(1, 2))
                                        for t in (xh, xl, xs_patch, xw, xd))
            if random.random() < 0.5:
                xh, xl, xs_patch, xw, xd = (torch.flip(t, dims=(2,))
                                            for t in (xh, xl, xs_patch, xw, xd))
        xs_mean = torch.tensor([float(c.mean()) for c in xs_patch] + extras, dtype=torch.float32)
        pollutants = self.cfg["pollutants"]
        y = np.zeros(len(pollutants), dtype="float32")
        mask = np.zeros(len(pollutants), dtype="float32")
        for j, p in enumerate(pollutants):
            v = r[p]
            if pd.notna(v) and v > 0:
                y[j] = (np.log(v) - self.tmean[j]) / self.tstd[j]
                mask[j] = 1.0
        return xh, xl, xs_patch, xw, xd, xs_mean, torch.from_numpy(y), torch.from_numpy(mask)
