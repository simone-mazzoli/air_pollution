"""
Train final model on all CV data (all folds, NE DE test fold still excluded)
and
save a checkpoint bundle for test-set inference. No held-out fold, no early
stopping, fixed epoch budget. Reuses the architecture,
dataset, stats, and seeding from the CV training script.

"""
import argparse, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

TRAIN_MODULE = Path(__file__).resolve().parent / "01_train_eea.py"
spec = importlib.util.spec_from_file_location("train_eea", TRAIN_MODULE)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
DEVICE = T.DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20, #N EPOCHS
                    help="fixed budget (CV folds early-stop at various epochs; "
                         "after the new CV run, set this near their median)")
    ap.add_argument("--batch", type=int, default=T.CONFIG["batch"])
    ap.add_argument("--s5p", nargs="+", default=T.CONFIG["s5p_streams"])
    ap.add_argument("--no-aer-wide", dest="use_aer_wide", action="store_false",
                    default=T.CONFIG["use_aer_wide"])
    ap.add_argument("--no-dem", dest="use_dem", action="store_false",
                    default=T.CONFIG["use_dem"])
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--unfreeze-epoch", type=int, default=T.CONFIG["unfreeze_epoch"])
    ap.add_argument("--out", default="final_model.pt")
    args = ap.parse_args()
    T.seed_everything()                      # SEED=0 in 01_train_eea.py

    cfg = dict(T.CONFIG)
    cfg["epochs"] = args.epochs; cfg["batch"] = args.batch
    cfg["s5p_streams"] = args.s5p; cfg["pretrained"] = not args.scratch
    cfg["use_aer_wide"] = args.use_aer_wide; cfg["use_dem"] = args.use_dem
    cfg["unfreeze_epoch"] = args.unfreeze_epoch; cfg["freeze_backbone"] = False
    cfg["pollutants"] = list(T.POLLUTANTS)

    streams = [f"{s}_tropomi" for s in cfg["s5p_streams"]]
    df = T.load_frame(streams, cfg)          # all CV stations, test countries already excluded
    print(f"\ndevice: {DEVICE}  |  seed: {T.SEED}  |  final model on ALL {len(df)} "
          f"CV stations, {args.epochs} epochs (no held-out fold)\n")

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
    unfrozen = False

    for ep in range(1, cfg["epochs"] + 1):
        if (not unfrozen) and ep > cfg["unfreeze_epoch"]:
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True
            bb = [p for p in model.backbone.parameters() if p.requires_grad]
            opt.add_param_group({"params": bb, "lr": cfg["lr_backbone"],
                                 "weight_decay": cfg["wd_backbone"]})
            unfrozen = True
            print(f"  epoch {ep}: unfreezing backbone.layer4")
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
    torch.save(bundle, args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
