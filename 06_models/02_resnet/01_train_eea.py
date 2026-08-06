"""
Multi-stream CNN: annual PM10 + PM2.5 from S2 + S5P + DEM, grouped country-cluster CV.
Components:
  high_res S2 (120x120x10)   -> pretrained BigEarthNet ResNet50 -> proj 128
  low_res  S2 (60x60x10)     -> small CNN -> 128
  S5P no2/co 5x5 patches     -> small CNN -> 32
  S5P aerosol 31x31 (~217km) -> small strided CNN -> 64   (regional background)
  DEM relief 60x60 (~12km)   -> small CNN -> 64           (center-relative elevation)
Scalar skips into the head: per-stream 5x5 patch means, local AER (wide-patch
center, z-scored), station elevation in km (from DEM center).
Predicts both pm10 and pm2.5 jointly (2-output head, masked loss so a station
contributes to whichever labels it has). Cross Validation with Leave-one-out. 
East/north German States are the final test set.
Training samples get a random rotation/flip (applied identically to all spatial streams, same
geography). eval uses the 8-view TTA average.
--no-aer-wide / --no-dem drop the respective stream (ablation).
Training architecture: head-only. The pretrained BigEarthNet backbone is frozen for
the whole run (weights and BN stats never updated); only the projection, the small
CNNs, and the head train. Single fixed lr_head, no scheduler.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
CONFIG = {
    "epochs": 500, "batch": 128, "patience": 15,
    "lr_head": 1e-5, "wd_head": 1e-7,
    "proj_dim": 64, "dropout": 0.8, "low_cnn_ch1": 32, "low_cnn_ch2": 128,
    "s5p_cnn_hidden": 32, "head_hidden": 64, "wide_feat": 32, "dem_feat": 32,
    "s5p_streams": ["no2", "co"],  # aerosol now enters via the wide stream
    "use_aer_wide": True, "use_dem": True,
    "pretrained": True,
    "tta": True, "folds": None, "out": "eea_cv_results.json",
    "buffer_km": 100.0,   # drop train stations within this many km of any val station (0=off)
    # annual means above these are treated as a broken sensor; label dropped
    "max_pm10": 120.0, "max_pm25": 80.0,
    "pollutants": ["pm25"],   # set to ["pm10"] or ["pm25"] for single-target
}
SEED = 123
CUDNN_DETERMINISTIC = False   # True -> exact repro on GPU, slower conv kernels
RELIEF_SCALE = 250.0  # metres; fixed DEM relief normalizer (no fold stats needed)
BASE = Path(__file__).resolve().parent.parent.parent
PROC = BASE / "data" / "processed"
SAT = PROC / "satellite_eea"
LABELS = PROC / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"
STATION_LAND = PROC / "uba" / "station_land.csv"   # station_code -> land (DEUB resolved)
HIGH = SAT / "high_res_multispec"
LOW = SAT / "low_res_multispec"
AERW = SAT / "aer_wide_tropomi"
DEMD = SAT / "dem_glo30"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POLLUTANTS = CONFIG["pollutants"]
DISPLAY = {"pm10": "PM10", "pm25": "PM2.5"}  # for anything printed to the terminal
MEAN = np.array([438.3721, 614.0557, 588.4096, 942.8433, 1769.9316,
                 2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325],
                dtype="float32").reshape(1, 1, -1)
STD = np.array([607.0269, 603.2968, 684.5688, 738.4327, 1100.4561,
                1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276],
               dtype="float32").reshape(1, 1, -1)
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def _worker_init(worker_id):
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
    """Load a single-band patch as 2D. Squeezes a trailing band dim if present."""
    arr = np.load(path).astype("float32")
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return arr
def load_dem(path):
    """DEM -> (relief patch (1,H,W), station elevation in metres)."""
    raw = load_patch_raw(path)
    finite = raw[np.isfinite(raw)]
    fallback = float(finite.mean()) if len(finite) else 0.0
    h, w = raw.shape
    cb = raw[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1]
    cbf = cb[np.isfinite(cb)]
    center = float(cbf.mean()) if len(cbf) else fallback
    relief = np.where(np.isfinite(raw), (raw - center) / RELIEF_SCALE, 0.0).astype("float32")
    return relief[None], center
FOLDS = {
    "fold1_iberia": ["PT", "ES", "AD"],
    "fold2_france": ["FR", "IE"],
    "fold3_italy": ["IT", "MT"],
    "fold4_alpine": ["DE", "CH", "AT"],   # DE = western/southern Laender only; east/north DE is TEST
    "fold5_north": ["NL", "BE", "LU", "DK", "SE", "NO", "FI", "IS"],
    "fold6_balkan_e": ["CZ", "SK", "HU", "SI", "HR", "BA", "RS", "XK", "ME", "RO"],
    "fold7_balkan_s": ["BG", "AL", "GR", "CY", "TR", "MK"],
    "fold8_poland": ["PL", "LT", "LV", "EE"],   # now a normal CV fold
}
# Sealed test set is east/north German Laender ONLY -- labelled "TEST" in
# station_fold.csv by 00_assign_folds.py, dropped from CV here. There are no
# whole-country test members anymore.
COUNTRY_TO_FOLD = {cc: f for f, ccs in FOLDS.items() for cc in ccs}
# Fold membership is NOT computed here -- it is read from station_fold.csv, which
# 00_assign_folds.py writes (and plots a QC map for). That keeps the split in one
# place so the CV trainer, final trainer, and test inference can't disagree. Run
# 00_assign_folds.py first. "TEST" rows (east/north DE) and unassigned rows are
# dropped here exactly as before.
STATION_FOLD = PROC / "eea" / "station_fold.csv"
def _load_fold_map():
    if not STATION_FOLD.exists():
        raise SystemExit(
            f"ERROR: {STATION_FOLD} not found. Run 00_assign_folds.py first to "
            f"generate the fold assignment (and its QC map).")
    return pd.read_csv(STATION_FOLD, dtype={"station_code": str})
def buffer_exclude(train_df, val_df, buffer_km):
    """Drop training stations whose location is within buffer_km of ANY validation
    station (great-circle). Removes cross-border spatial leakage from overlapping
    wide-patch footprints. Requires lat/lon columns."""
    if len(train_df) == 0 or len(val_df) == 0:
        return train_df
    R = 6371.0
    tlat = np.radians(train_df["lat"].values)[:, None]
    tlon = np.radians(train_df["lon"].values)[:, None]
    vlat = np.radians(val_df["lat"].values)[None, :]
    vlon = np.radians(val_df["lon"].values)[None, :]
    dphi = vlat - tlat
    dl = vlon - tlon
    a = np.sin(dphi / 2) ** 2 + np.cos(tlat) * np.cos(vlat) * np.sin(dl / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a))          # (n_train, n_val) km
    nearest = dist.min(axis=1)
    return train_df[nearest >= buffer_km]
def load_frame(s5p_streams, cfg):
    lab = pd.read_csv(LABELS, dtype={"station_code": str})
    g = lab.groupby("station_code")
    ann = pd.DataFrame({"pm10": g["PM10"].mean(), "pm25": g["PM2.5"].mean()}).reset_index()
    ann["country"] = ann["station_code"].str[:2]
    sf = _load_fold_map()
    fold_map = dict(zip(sf["station_code"], sf["fold"]))
    ann["fold"] = ann["station_code"].map(fold_map)
    coord = sf.set_index("station_code")[["lat", "lon"]]
    ann = ann.join(coord, on="station_code")
    # "TEST" (east/north DE only) and NaN (unlisted / not in the fold file) are
    # both non-CV -> dropped. Genuinely-unlisted = no fold AND not TEST.
    dropped = ann[ann["fold"].isna()]["country"].value_counts()
    ann.loc[ann["fold"] == "TEST", "fold"] = np.nan
    ann = ann.dropna(subset=["fold"]).reset_index(drop=True)
    for p, cap in (("pm10", cfg["max_pm10"]), ("pm25", cfg["max_pm25"])):
        n_out = int((ann[p] > cap).sum())
        if n_out:
            print(f"  outlier filter {DISPLAY[p]}: {n_out} stations above {cap:g} ug/m3 -> label dropped")
        ann.loc[(ann[p] <= 0) | (ann[p] > cap), p] = np.nan
    ann = ann[ann[["pm10", "pm25"]].notna().any(axis=1)].reset_index(drop=True)
    extra_dirs = ([AERW] if cfg["use_aer_wide"] else []) + ([DEMD] if cfg["use_dem"] else [])
    def has_all(code):
        if not ((HIGH / f"{code}.npy").exists() and (LOW / f"{code}.npy").exists()):
            return False
        if not all((SAT / st / f"{code}.npy").exists() for st in s5p_streams):
            return False
        return all((d / f"{code}.npy").exists() for d in extra_dirs)
    keep = ann["station_code"].map(has_all)
    print(f"{len(ann)} CV stations (after outlier + test/unlisted removal) -> "
          f"{int(keep.sum())} with all patches")
    if len(dropped):
        print(f"  dropped unlisted countries: {dict(dropped)}")
    ann = ann[keep].reset_index(drop=True)
    print("  stations per fold (total / with PM10 / with PM2.5):")
    for f in FOLDS:
        sub = ann[ann["fold"] == f]
        print(f"    {f:<16} {len(sub):>4}  /  {int(sub['pm10'].notna().sum()):>4}  "
              f"/  {int(sub['pm25'].notna().sum()):>4}")
    return ann
class EEA(Dataset):
    def __init__(self, frame, streams, tmean, tstd, s5p_stats, cfg, augment=False):
        self.f = frame.reset_index(drop=True)
        self.streams, self.tmean, self.tstd, self.s5p_stats = streams, tmean, tstd, s5p_stats
        self.cfg, self.augment = cfg, augment
    def __len__(self): return len(self.f)
    def _norm_s5p(self, path, key):
        m, sd = self.s5p_stats[key]
        raw = load_patch_raw(path)
        safe = np.where(np.isfinite(raw), raw, m)
        return ((safe - m) / sd if sd > 0 else np.zeros_like(safe)).astype("float32")
    def __getitem__(self, i):
        r = self.f.iloc[i]; code = r["station_code"]
        xh = torch.from_numpy(load_s2(HIGH / f"{code}.npy"))
        xl = torch.from_numpy(load_s2(LOW / f"{code}.npy"))
        chans = [self._norm_s5p(SAT / st / f"{code}.npy", st) for st in self.streams]
        xs_patch = torch.from_numpy(np.stack(chans, axis=0))  # (n_streams, 5, 5)
        extras = []
        if self.cfg["use_aer_wide"]:
            w = self._norm_s5p(AERW / f"{code}.npy", "aer_wide")
            xw = torch.from_numpy(w[None])                     # (1, 31, 31)
            extras.append(float(w[w.shape[0] // 2, w.shape[1] // 2]))  # local AER, z-scored
        else:
            xw = torch.zeros(1, 1, 1)
        if self.cfg["use_dem"]:
            relief, elev = load_dem(DEMD / f"{code}.npy")
            xd = torch.from_numpy(relief)                      # (1, 60, 60)
            extras.append(elev / 1000.0)                       # station elevation, km
        else:
            xd = torch.zeros(1, 1, 1)
        if self.augment:
            k = random.randint(0, 3)
            xh, xl, xs_patch, xw, xd = (torch.rot90(t, k, dims=(1, 2))
                                        for t in (xh, xl, xs_patch, xw, xd))
            if random.random() < 0.5:
                xh, xl, xs_patch, xw, xd = (torch.flip(t, dims=(2,))
                                            for t in (xh, xl, xs_patch, xw, xd))
        xs_mean = torch.tensor([float(c.mean()) for c in xs_patch] + extras,
                               dtype=torch.float32)
        y = np.zeros(len(POLLUTANTS), dtype="float32")
        mask = np.zeros(len(POLLUTANTS), dtype="float32")
        for j, p in enumerate(POLLUTANTS):
            v = r[p]
            if pd.notna(v) and v > 0:
                y[j] = (np.log(v) - self.tmean[j]) / self.tstd[j]
                mask[j] = 1.0
        return xh, xl, xs_patch, xw, xd, xs_mean, torch.from_numpy(y), torch.from_numpy(mask)
class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out, pretrained=True):
        super().__init__()
        import timm
        proj, dropout = cfg["proj_dim"], cfg["dropout"]
        c1, c2 = cfg["low_cnn_ch1"], cfg["low_cnn_ch2"]
        s5p_hidden, head_hidden = cfg["s5p_cnn_hidden"], cfg["head_hidden"]
        wf, df = cfg["wide_feat"], cfg["dem_feat"]
        self.use_wide, self.use_dem = cfg["use_aer_wide"], cfg["use_dem"]
        self.backbone = timm.create_model("resnet50", pretrained=False, in_chans=10, num_classes=0)
        feat = self.backbone.num_features
        if pretrained:
            self._load_pretrained()
        self.proj_h = nn.Linear(feat, proj)
        self.low_cnn = nn.Sequential(
            nn.Conv2d(10, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, 128), nn.ReLU(True))
        self.s5p_cnn = nn.Sequential(
            nn.Conv2d(n_s5p, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
            nn.Conv2d(s5p_hidden, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.norm_h = nn.BatchNorm1d(proj, affine=False)
        self.norm_l = nn.BatchNorm1d(128, affine=False)
        self.norm_s = nn.BatchNorm1d(s5p_hidden, affine=False)
        n_scalars = n_s5p + int(self.use_wide) + int(self.use_dem)
        head_in = proj + 128 + s5p_hidden + n_scalars
        if self.use_wide:
            self.wide_cnn = nn.Sequential(  # 31x31 -> strided convs -> global pool
                nn.Conv2d(1, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
                nn.Conv2d(wf, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.norm_w = nn.BatchNorm1d(wf, affine=False)
            head_in += wf
        if self.use_dem:
            self.dem_cnn = nn.Sequential(   # 60x60 relief
                nn.Conv2d(1, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(df, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.norm_d = nn.BatchNorm1d(df, affine=False)
            head_in += df
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, head_hidden), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(head_hidden, n_out))
    def _load_pretrained(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0", "model.safetensors")
        ckpt = load_file(path); sd = self.backbone.state_dict()
        remap, n = {}, 0
        for k, v in ckpt.items():
            b = k
            for p in ("model.vision_encoder.", "vision_encoder.", "model.", "backbone.",
                      "network.", "encoder.", "resnet.", "timm_model."):
                if b.startswith(p): b = b[len(p):]
            if b in sd and sd[b].shape == v.shape:
                remap[b] = v; n += 1
        self.backbone.load_state_dict(remap, strict=False)
        print(f"  loaded {n}/{len(sd)} pretrained tensors")
    def forward(self, xh, xl, xs_patch, xw, xd, xs_mean):
        parts = [self.norm_h(self.proj_h(self.backbone(xh))),
                 self.norm_l(self.low_cnn(xl)),
                 self.norm_s(self.s5p_cnn(xs_patch))]
        if self.use_wide:
            parts.append(self.norm_w(self.wide_cnn(xw)))
        if self.use_dem:
            parts.append(self.norm_d(self.dem_cnn(xd)))
        parts.append(xs_mean)
        return self.head(torch.cat(parts, dim=1))
@torch.no_grad()
def evaluate(model, loader, tmean, tstd, tta=False):
    model.eval(); P, T, M = [], [], []
    for xh, xl, xs_patch, xw, xd, xs_mean, y, m in loader:
        xh, xl, xs_patch, xw, xd, xs_mean = (
            xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE),
            xw.to(DEVICE), xd.to(DEVICE), xs_mean.to(DEVICE))
        if tta:
            views = []
            for k in range(4):
                r = [torch.rot90(t, k, dims=(2, 3)) for t in (xh, xl, xs_patch, xw, xd)]
                views.append(model(*r, xs_mean))
                f = [torch.flip(t, dims=(3,)) for t in r]
                views.append(model(*f, xs_mean))
            out = torch.stack(views).mean(0)
        else:
            out = model(xh, xl, xs_patch, xw, xd, xs_mean)
        P.append(out.cpu().numpy()); T.append(y.numpy()); M.append(m.numpy())
    P = np.concatenate(P); T = np.concatenate(T); M = np.concatenate(M)
    res, arrs = {}, {}
    for j, p in enumerate(POLLUTANTS):
        sel = M[:, j] > 0
        if sel.sum() == 0:
            res[p] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
            continue
        pred = np.exp(P[sel, j] * tstd[j] + tmean[j])
        true = np.exp(T[sel, j] * tstd[j] + tmean[j])
        err = pred - true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        sst = np.sum((true - true.mean()) ** 2)
        r2 = float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan")
        res[p] = {"rmse": rmse, "mae": mae, "r2": r2, "n": int(sel.sum())}
        arrs[p] = (pred, true)
    return res, arrs
def masked_loss(pred, y, mask, lossf):
    per = lossf(pred, y)
    return (per * mask).sum() / mask.sum().clamp(min=1)
def fmt_metrics(res):
    parts = []
    for p in POLLUTANTS:
        s = f"{DISPLAY[p]}: RMSE={res[p]['rmse']:.2f} MAE={res[p]['mae']:.2f} R2={res[p]['r2']:.3f}"
        parts.append(s)
    return "  |  ".join(parts)
def compute_s5p_stats(train_df, streams, cfg):
    """Pooled per-pixel train stats for each 5x5 stream and (if enabled) aer_wide."""
    dirs = {st: SAT / st for st in streams}
    if cfg["use_aer_wide"]:
        dirs["aer_wide"] = AERW
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
def train_one_fold(train_df, val_df, streams, cfg):
    seed_everything()
    tmean = np.array([np.nanmean(np.log(train_df[p].values)) for p in POLLUTANTS], "float64")
    tstd = np.array([np.nanstd(np.log(train_df[p].values)) or 1.0 for p in POLLUTANTS], "float64")
    s5p_stats = compute_s5p_stats(train_df, streams, cfg)
    tr = DataLoader(EEA(train_df, streams, tmean, tstd, s5p_stats, cfg, augment=True),
                    batch_size=cfg["batch"], shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True, worker_init_fn=_worker_init)
    va = DataLoader(EEA(val_df, streams, tmean, tstd, s5p_stats, cfg), batch_size=cfg["batch"],
                    shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=_worker_init)
    tsub = train_df.sample(min(1000, len(train_df)), random_state=0)
    tm = DataLoader(EEA(tsub, streams, tmean, tstd, s5p_stats, cfg), batch_size=cfg["batch"],
                    shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=_worker_init)
    base = {}
    for p in POLLUTANTS:
        tr_mean = np.nanmean(train_df[p].values)
        vv = val_df[p].values; vv = vv[~np.isnan(vv)]
        base[p] = float(np.sqrt(np.mean((vv - tr_mean) ** 2))) if len(vv) else float("nan")
    print("  baseline (train mean): " + "  ".join(f"{DISPLAY[p]} RMSE={base[p]:.2f}" for p in POLLUTANTS))
    model = Net(len(streams), cfg, n_out=len(POLLUTANTS), pretrained=cfg["pretrained"]).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad = False
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": hd, "lr": cfg["lr_head"], "weight_decay": cfg["wd_head"]}])
    lossf = nn.SmoothL1Loss(reduction="none")
    def mean_val_rmse(res):
        vs = [res[p]["rmse"] for p in POLLUTANTS if not np.isnan(res[p]["rmse"])]
        return float(np.mean(vs)) if vs else np.inf
    best, best_state, bad = np.inf, None, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); model.backbone.eval()
        tot = 0.0
        for xh, xl, xs_patch, xw, xd, xs_mean, y, m in tr:
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
                xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
                xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
            opt.zero_grad()
            loss = masked_loss(model(xh, xl, xs_patch, xw, xd, xs_mean), y, m, lossf)
            loss.backward(); opt.step(); tot += loss.item() * len(xh)
        trm, _ = evaluate(model, tm, tmean, tstd)
        val, _ = evaluate(model, va, tmean, tstd, tta=cfg["tta"])
        r = mean_val_rmse(val); flag = ""
        if r < best - 1e-4:
            best = r; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0; flag = " *"
        else:
            bad += 1
        print(f"  [{ep:02d}] loss={tot/len(train_df):.3f}")
        print(f"       TRAIN  {fmt_metrics(trm)}")
        print(f"       VAL    {fmt_metrics(val)}{flag}")
        if bad >= cfg["patience"]:
            print("  early stop"); break
    model.load_state_dict(best_state)
    val, arrs = evaluate(model, va, tmean, tstd, tta=cfg["tta"])
    out = {"n_train": len(train_df), "n_val": len(val_df)}
    for p in POLLUTANTS:
        out[p] = {"rmse": val[p]["rmse"], "mae": val[p]["mae"], "r2": val[p]["r2"],
                  "n": val[p]["n"], "baseline": base[p]}
    return out, arrs
def build_cfg_from_args(args):
    cfg = dict(CONFIG)
    for k in ("epochs", "batch", "lr_head", "wd_head",
              "patience", "tta", "out", "max_pm10", "max_pm25", "buffer_km"):
        cfg[k] = getattr(args, k)
    cfg["s5p_streams"] = args.s5p
    cfg["pretrained"] = not args.scratch
    cfg["use_aer_wide"] = args.use_aer_wide
    cfg["use_dem"] = args.use_dem
    cfg["folds"] = args.folds
    cfg["pollutants"] = list(POLLUTANTS)
    return cfg
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--batch", type=int, default=CONFIG["batch"])
    ap.add_argument("--lr-head", type=float, default=CONFIG["lr_head"])
    ap.add_argument("--wd-head", type=float, default=CONFIG["wd_head"])
    ap.add_argument("--patience", type=int, default=CONFIG["patience"])
    ap.add_argument("--max-pm10", type=float, default=CONFIG["max_pm10"],
                    help="drop stations with annual PM10 above this (malfunctioning sensor)")
    ap.add_argument("--max-pm25", type=float, default=CONFIG["max_pm25"],
                    help="drop stations with annual PM2.5 above this (malfunctioning sensor)")
    ap.add_argument("--s5p", nargs="+", default=CONFIG["s5p_streams"],
                    help="5x5 S5P streams (aerosol enters via --aer-wide instead)")
    ap.add_argument("--no-aer-wide", dest="use_aer_wide", action="store_false",
                    default=CONFIG["use_aer_wide"], help="drop the 31x31 aerosol stream")
    ap.add_argument("--no-dem", dest="use_dem", action="store_false",
                    default=CONFIG["use_dem"], help="drop the DEM relief stream")
    ap.add_argument("--buffer-km", type=float, default=CONFIG["buffer_km"],
                    help="exclude train stations within this many km of any val station (0=off)")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--tta", action="store_true", default=CONFIG["tta"])
    ap.add_argument("--folds", nargs="+", default=CONFIG["folds"])
    ap.add_argument("--out", default=CONFIG["out"])
    args = ap.parse_args()
    seed_everything()
    cfg = build_cfg_from_args(args)
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]
    df = load_frame(streams, cfg)
    run_folds = cfg["folds"] or list(FOLDS)
    print(f"\ndevice: {DEVICE}  |  seed: {SEED}  |  running folds: {run_folds}")
    print(f"config: {json.dumps({k: v for k, v in cfg.items() if k != 'folds'})}\n")
    cv, scatter = {}, []
    for fold in run_folds:
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        if len(val_df) < 10:
            print(f"########## {fold}: only {len(val_df)} stations, SKIPPING ##########\n")
            continue
        if cfg["buffer_km"] > 0:
            n_before = len(train_df)
            train_df = buffer_exclude(train_df, val_df, cfg["buffer_km"]).reset_index(drop=True)
            print(f"  buffer {cfg['buffer_km']:g}km: dropped {n_before - len(train_df)}"
                  f"/{n_before} train stations near val border")
        print(f"########## VAL FOLD: {fold}  (train {len(train_df)}, val {len(val_df)}) ##########")
        res, arrs = train_one_fold(train_df, val_df, streams, cfg)
        print(f"  FINAL [{fold}]:")
        print(f"    {fmt_metrics({p: res[p] for p in POLLUTANTS})}\n")
        cv[fold] = res
        for p in POLLUTANTS:
            if p in arrs:
                pred, true = arrs[p]
                scatter.append(pd.DataFrame({"fold": fold, "pollutant": p,
                                             "pred": pred, "true": true}))
    print("=" * 60); print("CROSS-VALIDATION SUMMARY")
    done = list(cv)
    if done:
        overall_model_rmses = []
        for p in POLLUTANTS:
            rmses = [cv[f][p]["rmse"] for f in done if not np.isnan(cv[f][p]["rmse"])]
            maes = [cv[f][p]["mae"] for f in done if not np.isnan(cv[f][p]["mae"])]
            r2s = [cv[f][p]["r2"] for f in done if not np.isnan(cv[f][p]["r2"])]
            bases = [cv[f][p]["baseline"] for f in done if not np.isnan(cv[f][p]["baseline"])]
            overall_model_rmses.extend(rmses)
            print(f"  {DISPLAY[p]}: mean-of-folds RMSE={np.mean(rmses):.2f}  "
                  f"(baseline RMSE={np.mean(bases):.2f})  MAE={np.mean(maes):.2f}  "
                  f"R2 mean={np.mean(r2s):+.3f} "
                  f"(positive in {sum(r>0 for r in r2s)}/{len(r2s)})")
            for f in done:
                print(f"    {f:<16} RMSE={cv[f][p]['rmse']:.2f}  "
                      f"(baseline={cv[f][p]['baseline']:.2f})  MAE={cv[f][p]['mae']:.2f}  "
                      f"R2={cv[f][p]['r2']:+.3f}  (n={cv[f][p]['n']})")
        print(f"\n  OVERALL (mean of folds, both pollutants pooled): "
              f"RMSE={np.mean(overall_model_rmses):.2f}")
        pd.concat(scatter, ignore_index=True).to_csv("eea_cv_predictions.csv", index=False)
    cv["_config"] = {k: v for k, v in cfg.items() if k != "folds"}
    Path(cfg["out"]).write_text(json.dumps(cv, indent=2))
    print(f"saved {cfg['out']}")
if __name__ == "__main__":
    main()
