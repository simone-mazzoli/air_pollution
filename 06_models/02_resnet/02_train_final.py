"""
Train final model on all CV data (all folds, NE DE test fold still excluded)
and
save a checkpoint bundle for test-set inference. No held-out fold, no early
stopping, fixed epoch budget. Reuses the architecture,
dataset, stats, and seeding from the CV training script.

Buffer leakage control (same as the CV script): training stations within
CONFIG buffer_km of any NE-Germany test station are dropped before training,
so the wide-patch footprints do not overlap the sealed test region.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader

EPOCHS = 25   # fixed budget (set near the CV folds' median peak epoch)
OUT = "final_model.pt"

TRAIN_MODULE = Path(__file__).resolve().parent / "01_train_eea.py"
spec = importlib.util.spec_from_file_location("train_eea", TRAIN_MODULE)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
DEVICE = T.DEVICE


def load_test_frame(streams, cfg):
    """NE-Germany test stations (fold == TEST in station_fold.csv), with lat/lon,
    same outlier + patch-completeness filtering as the CV frame. Used only as the
    reference set for the buffer exclusion."""
    sf = T._load_fold_map()
    test_codes = set(sf.loc[sf["fold"] == "TEST", "station_code"])
    lab = pd.read_csv(T.LABELS, dtype={"station_code": str})
    g = lab.groupby("station_code")
    ann = pd.DataFrame({"pm10": g["PM10"].mean(), "pm25": g["PM2.5"].mean()}).reset_index()
    ann = ann[ann["station_code"].isin(test_codes)].reset_index(drop=True)
    coord = sf.set_index("station_code")[["lat", "lon"]]
    ann = ann.join(coord, on="station_code")
    for p, cap in (("pm10", cfg["max_pm10"]), ("pm25", cfg["max_pm25"])):
        ann.loc[(ann[p] <= 0) | (ann[p] > cap), p] = np.nan
    ann = ann[ann[["pm10", "pm25"]].notna().any(axis=1)].reset_index(drop=True)
    extra = ([T.AERW] if cfg["use_aer_wide"] else []) + ([T.DEMD] if cfg["use_dem"] else [])
    def has_all(code):
        if not ((T.HIGH / f"{code}.npy").exists() and (T.LOW / f"{code}.npy").exists()):
            return False
        if not all((T.SAT / st / f"{code}.npy").exists() for st in streams):
            return False
        return all((d / f"{code}.npy").exists() for d in extra)
    ann = ann[ann["station_code"].map(has_all)].reset_index(drop=True)
    return ann


def main():
    T.seed_everything()                      # SEED in 01_train_eea.py
    cfg = dict(T.CONFIG)
    cfg["epochs"] = EPOCHS
    cfg["pollutants"] = list(T.POLLUTANTS)
    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]

    df = T.load_frame(streams, cfg)          # all CV stations (TEST excluded)
    test_df = load_test_frame(streams, cfg)  # NE-Germany, reference for the buffer
    if cfg["buffer_km"] > 0:
        n_before = len(df)
        df = T.buffer_exclude(df, test_df, cfg["buffer_km"]).reset_index(drop=True)
        print(f"buffer {cfg['buffer_km']:g}km: dropped {n_before - len(df)}/{n_before} "
              f"train stations near NE-Germany")
    print(f"\ndevice: {DEVICE}  |  seed: {T.SEED}  |  final model on ALL {len(df)} "
          f"CV stations, {cfg['epochs']} epochs (no held-out fold)\n")

    tmean = np.array([np.nanmean(np.log(df[p].values)) for p in T.POLLUTANTS], "float64")
    tstd = np.array([np.nanstd(np.log(df[p].values)) or 1.0 for p in T.POLLUTANTS], "float64")
    s5p_stats = T.compute_s5p_stats(df, streams, cfg)
    tr = DataLoader(T.EEA(df, streams, tmean, tstd, s5p_stats, cfg, augment=True),
                    batch_size=cfg["batch"], shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True, worker_init_fn=T._worker_init)

    model = T.Net(len(streams), cfg, n_out=len(T.POLLUTANTS), pretrained=cfg["pretrained"]).to(DEVICE)
    for p in model.backbone.parameters():
        p.requires_grad = False
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": hd, "lr": cfg["lr_head"], "weight_decay": cfg["wd_head"]}])
    lossf = nn.SmoothL1Loss(reduction="none")

    for ep in range(1, cfg["epochs"] + 1):
        model.train(); model.backbone.eval()
        tot = 0.0
        for xh, xl, xs_patch, xw, xd, xs_mean, y, m in tr:
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
                xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
                xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
            opt.zero_grad()
            loss = T.masked_loss(model(xh, xl, xs_patch, xw, xd, xs_mean), y, m, lossf)
            loss.backward(); opt.step(); tot += loss.item() * len(xh)
        print(f"  [{ep:02d}] loss={tot/len(df):.3f}")

    bundle = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": cfg, "streams": streams, "pollutants": list(T.POLLUTANTS),
        "tmean": tmean, "tstd": tstd, "s5p_stats": s5p_stats,
    }
    torch.save(bundle, OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
