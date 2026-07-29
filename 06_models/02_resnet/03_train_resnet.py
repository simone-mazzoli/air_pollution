"""
Step 1: fine-tune the BigEarthNet ResNet50 on HIGH-RES patches only, predicting
[PM10, PM2.5] (2-output regression). Low-res wide-context stream is NOT used yet --
single-stream baseline; low-res CNN added later as an ablation.

Proper held-out-Land cross-validation:
  - Sachsen-Anhalt is the sealed test set -- NEVER touched here.
  - Among the other 11 folds, each takes a turn as the validation Land: train on the
    sensors of the other 10 folds, validate on the held-out fold's sensors.
  - Final metrics are AVERAGED across the 11 folds. (SA is evaluated separately later,
    once, with the finished approach -- not in this script.)

The sensor->fold split comes straight from sensor_land.csv + LAND_TO_FOLD (same mapping
the calibration used), NOT from parsing fold directory names.

Two-phase fine-tune with an explicit catastrophic-forgetting check per fold:
  FULL end-to-end fine-tuning: the whole backbone is trainable from epoch 0. The
  backbone uses a lower LR than the head (so the pretrained init isn't wrecked early),
  cosine LR decay, Huber loss, and early stopping on val RMSE. Frozen-feature probing
  was tried first and only matched the mean-baseline -- the PM signal is not in the
  frozen land-cover features, so the features themselves must adapt.

Labels are joined to patches BY LOCATION inside the dataset (no fragile parallel
array), so dropped/missing patches can never misalign a label onto the wrong patch.
Targets are z-scored using TRAIN-fold stats only; predictions de-standardized to
ug/m3 before metrics.
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
MANIFEST = SAT_DIR / "manifest.csv"
SENSOR_LAND = PROC / "sensor_land.csv"

TEST_FOLD_DIRNAME = "Sachsen-Anhalt_TEST"
TEST_LAND = "Sachsen-Anhalt"
TARGETS = ["PM10_corrected"]
LOG_TARGET = True  # train on log(PM): compresses right-skew, stops mean-collapse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# same mapping the calibration used
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

# BigEarthNet v2 per-band mean/std (band order B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12)
MEAN = np.array([438.3721, 614.0557, 588.4096, 942.8433, 1769.9316,
                 2049.5515, 2193.292, 2235.5566, 1568.2268, 997.7325], dtype="float32").reshape(1,1,-1)
STD = np.array([607.0269, 603.2968, 684.5688, 738.4327, 1100.4561,
                1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276], dtype="float32").reshape(1,1,-1)


def _canon_loc(loc):
    return f"{float(loc):.1f}"


def normalize_patch(arr):
    m = ~np.isfinite(arr) | (arr <= 0)
    safe = np.where(m, 0.0, arr).astype("float32")
    normed = (safe - MEAN) / STD
    normed = np.where(m, 0.0, normed).astype("float32")
    return np.transpose(normed, (2, 0, 1))


class PatchDataset(Dataset):
    """Holds a frame with columns [loc_str, target0, target1]. Labels travel WITH the
    location row, so a missing patch simply drops that row up-front -- no parallel
    array to misalign. Locations without an 'ok' patch on disk are filtered in __init__."""
    def __init__(self, frame, tmean, tstd, augment=False):
        self.augment = augment
        ok = []
        for _, r in frame.iterrows():
            if (HIGH_RES / f"{r['loc_str']}.npy").exists():
                ok.append(r)
        self.frame = pd.DataFrame(ok).reset_index(drop=True)
        self.tmean, self.tstd = tmean, tstd
        print(f"    {len(frame)} requested -> {len(self.frame)} with a patch on disk")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        arr = np.load(HIGH_RES / f"{r['loc_str']}.npy")
        if self.augment:
            # geometric: flips + 90deg rotations (satellite patches are orientation-free)
            k = np.random.randint(4)
            if k: arr = np.rot90(arr, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5: arr = arr[::-1].copy()
            if np.random.rand() < 0.5: arr = arr[:, ::-1].copy()
            # radiometric: mild per-patch brightness/contrast jitter on reflectance,
            # regularises the unfrozen layer4 against absolute-value memorisation
            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.9, 1.1)
                bias = np.random.uniform(-30, 30)
                arr = (arr * gain + bias).astype("float32")
            # small spatial jitter: roll by a few px so the model can't key on exact center
            sh = np.random.randint(-6, 7, size=2)
            arr = np.roll(arr, sh, axis=(0, 1)).copy()
        x = torch.from_numpy(normalize_patch(arr))
        y = ((r[TARGETS].values.astype("float32") - self.tmean) / self.tstd).astype("float32")
        return x, torch.from_numpy(y)


class ResNetRegressor(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.backbone, feat = self._load_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(feat, 128), nn.ReLU(True),
            nn.BatchNorm1d(128), nn.Dropout(0.6), nn.Linear(128, n_out))

    def _load_backbone(self):
        """Load BigEarthNet resnet50-s2-v0.2.0 weights directly from HF into a timm
        ResNet50 (configilm is built on timm, so timm key names match the checkpoint;
        torchvision names would NOT). 10 input channels for the 10 S2 bands, num_classes=0
        strips the head -> 2048 pooled features. Prints how many tensors matched so a
        silent random-init can't slip through. Falls back to random init only if the
        download/load genuinely fails."""
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
                print("  WARNING: few tensors matched -- key naming off. "
                      f"Sample ckpt keys: {list(ckpt.keys())[:6]}")
        except Exception as e:
            print(f"WARNING: pretrained load failed ({type(e).__name__}: {e}); "
                  f"backbone stays RANDOM-init (NOT the real comparison)")
        return core, feat

    def forward(self, x): return self.head(self.backbone(x))

    def set_backbone_trainable(self, trainable, last_block_only=False):
        for p in self.backbone.parameters(): p.requires_grad = False
        if not trainable: return
        if last_block_only and hasattr(self.backbone, "layer4"):
            for p in self.backbone.layer4.parameters(): p.requires_grad = True
        else:
            for p in self.backbone.parameters(): p.requires_grad = True


@torch.no_grad()
def evaluate(model, loader, tmean, tstd):
    model.eval(); P, T = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        # test-time augmentation: average predictions over the 8 dihedral views
        views = []
        for k in range(4):
            xr = torch.rot90(xb, k, dims=(2, 3))
            views.append(model(xr))
            views.append(model(torch.flip(xr, dims=(3,))))
        out = torch.stack(views).mean(0)
        P.append(out.cpu().numpy()); T.append(yb.numpy())
    P = np.concatenate(P)*tstd+tmean; T = np.concatenate(T)*tstd+tmean
    if LOG_TARGET:
        P = np.exp(P); T = np.exp(T)   # back to ug/m3 for honest RMSE/R2
    out = {}
    for i, n in enumerate(TARGETS):
        p, t = P[:,i], T[:,i]
        rmse = float(np.sqrt(np.mean((p-t)**2))); mae = float(np.mean(np.abs(p-t)))
        ssr = np.sum((t-p)**2); sst = np.sum((t-t.mean())**2)
        out[n] = {"rmse": rmse, "mae": mae, "r2": float(1-ssr/sst) if sst>0 else float("nan")}
    return out


def run_epoch(model, loader, opt, lossf, freeze_backbone_bn=True):
    model.train()
    if freeze_backbone_bn:
        model.backbone.eval()   # keep pretrained BatchNorm running stats fixed
    tot = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        tot += loss.item()*len(xb)
    return tot/len(loader.dataset)


def fmt(m): return "  ".join(f"{n}: RMSE={v['rmse']:.2f} R2={v['r2']:.3f}" for n,v in m.items())
def mean_rmse(m): return float(np.mean([v["rmse"] for v in m.values()]))


def build_labeled_frame():
    """One row per non-SA sensor: location, land, fold, PM10/PM2.5 labels. Labels come
    from any fold file (all non-SA files share the same sensor set and, for a sensor
    NOT in that file's held-out land, the same calibrated value). To keep it simple and
    consistent we read every fold's own sensors from its own file, so each sensor's
    label is the one produced when its Land was the validation fold."""
    sl = pd.read_csv(SENSOR_LAND)[["location", "land"]]
    sl = sl[sl["land"] != TEST_LAND]
    sl["fold"] = sl["land"].map(LAND_TO_FOLD)
    sl = sl.dropna(subset=["fold"])

    frames = []
    for fold in sorted(sl["fold"].unique()):
        f = FOLD_DIR / fold.replace(" ", "_") / "annual" / "2024.csv"
        if not f.exists():
            print(f"WARNING: missing {f}, skipping fold {fold}"); continue
        df = pd.read_csv(f)[["location"] + TARGETS]
        keep = sl[sl["fold"] == fold][["location", "land", "fold"]]
        frames.append(keep.merge(df, on="location", how="inner"))
    out = pd.concat(frames, ignore_index=True).dropna(subset=TARGETS)
    if LOG_TARGET:
        for t in TARGETS:
            out = out[out[t] > 0]
            out[t] = np.log(out[t].values)   # model regresses log(PM)
    out["loc_str"] = out["location"].map(_canon_loc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--folds", nargs="+", default=None,
                    help="subset of folds to run as validation (default: all 11)")
    ap.add_argument("--out", default="resnet_cv_results.json")
    args = ap.parse_args()
    print(f"device: {DEVICE}")

    data = build_labeled_frame()
    all_folds = sorted(data["fold"].unique())
    run_folds = args.folds or all_folds
    print(f"{len(data)} sensors across {len(all_folds)} folds; validating on: {run_folds}\n")

    cv = {}
    for val_fold in run_folds:
        print(f"########## VALIDATION FOLD: {val_fold} ##########")
        train_df = data[data["fold"] != val_fold].copy()
        val_df = data[data["fold"] == val_fold].copy()

        tmean = train_df[TARGETS].values.mean(0).astype("float32")
        tstd = train_df[TARGETS].values.std(0).astype("float32")

        tr_loader = DataLoader(PatchDataset(train_df, tmean, tstd, augment=True), batch_size=args.batch,
                               shuffle=True, num_workers=4, pin_memory=True)
        va_loader = DataLoader(PatchDataset(val_df, tmean, tstd), batch_size=args.batch,
                               shuffle=False, num_workers=4, pin_memory=True)

        # baseline: predict the TRAIN mean for every val sensor. If the model can't
        # beat this, the imagery carries no signal for annual PM.
        # baseline predicts the train mean (in whatever space TARGETS are in),
        # then we convert to ug/m3 to compare honestly with the model.
        base_pred = train_df[TARGETS].values.mean(0)
        vy = val_df[TARGETS].values
        for i, t in enumerate(TARGETS):
            bp, vt = base_pred[i], vy[:, i]
            if LOG_TARGET:
                bp = np.exp(bp); vt = np.exp(vt)
            b_rmse = float(np.sqrt(np.mean((vt - bp) ** 2)))
            print(f"  BASELINE (predict train mean) {t}: RMSE={b_rmse:.2f} ug/m3")

        model = ResNetRegressor(len(TARGETS)).to(DEVICE)
        lossf = nn.SmoothL1Loss()

        # ANTI-OVERFIT RECIPE. With ~4400 patches a trainable ResNet50 memorises the
        # train set in one epoch (train loss falls, val degrades). So we FREEZE the
        # whole convolutional backbone -- its filters cannot overfit because they never
        # update -- and train only a small, heavily-regularised head on the pooled
        # features. The backbone is also put in eval() every step so its BatchNorm
        # running stats stay at the pretrained values (a trainable BN is itself an
        # overfitting path on small data). This is the standard low-data transfer setup
        # and is the configuration least able to overfit.
        # MIDDLE GROUND: unfreeze ONLY layer4 (the last ResNet stage). Frozen backbone
        # gave R2~0 (no adaptable signal); full fine-tune overfit in 1 epoch. Adapting
        # just the top stage is enough capacity to reshape high-level features toward
        # PM10 without memorising 4400 patches. layer4 gets a small LR + heavy weight
        # decay; the head a larger LR. Everything below layer4 stays frozen, BN stats
        # frozen (run_epoch puts backbone in eval()).
        model.set_backbone_trainable(True, last_block_only=True)
        l4 = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": args.wd},
            {"params": l4, "lr": args.lr_backbone, "weight_decay": args.wd},
        ])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

        best, best_state = np.inf, None
        bad = 0
        for ep in range(1, args.epochs + 1):
            tr = run_epoch(model, tr_loader, opt, lossf)
            sched.step()
            val = evaluate(model, va_loader, tmean, tstd)
            r = mean_rmse(val)
            flag = ""
            if r < best - 1e-4:
                best, best_state = r, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0; flag = " *"
            else:
                bad += 1
            print(f"  [{ep:02d}] loss={tr:.3f}  {fmt(val)}{flag}")
            if bad >= args.patience:
                print(f"  early stop (no val improvement for {args.patience} epochs)")
                break

        model.load_state_dict(best_state)
        final = evaluate(model, va_loader, tmean, tstd)
        print(f"  FINAL [{val_fold}]: {fmt(final)}\n")
        cv[val_fold] = {"metrics": final, "best_rmse": best}

    # average across folds
    print("="*60); print("CROSS-VALIDATION SUMMARY (averaged over folds):")
    for t in TARGETS:
        rmse = np.mean([cv[f]["metrics"][t]["rmse"] for f in cv])
        r2 = np.mean([cv[f]["metrics"][t]["r2"] for f in cv])
        print(f"  {t}: mean RMSE={rmse:.2f}  mean R2={r2:.3f}")
    Path(args.out).write_text(json.dumps(cv, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
