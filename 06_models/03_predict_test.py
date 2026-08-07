import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from shared import data, evaluation
from shared.config import BATCH_SIZE, CACHE_PATCHES, DEVICE, DISPLAY, MODEL, NUM_WORKERS, USE_TTA, result_paths
from shared.models import selected_model


@torch.no_grad()
def predict(model, loader, tmean, tstd, cfg, tta):
    model.eval()
    pred_batches, true_batches, mask_batches = [], [], []
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
        pred_batches.append(out.cpu().numpy())
        true_batches.append(y.numpy())
        mask_batches.append(m.numpy())
    pred_norm = np.concatenate(pred_batches)
    true_norm = np.concatenate(true_batches)
    mask = np.concatenate(mask_batches)
    pred, true = evaluation.denormalize(pred_norm, true_norm, tmean, tstd)
    return pred, true, mask


def baseline_report(true, mask, baseline_concentration, cfg):
    for j, p in enumerate(cfg["pollutants"]):
        valid = mask[:, j] > 0
        if valid.sum() == 0:
            continue
        pred_const = baseline_concentration[p]
        err = pred_const - true[valid, j]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        print(f"  {'baseline':<10} {DISPLAY[p]}: RMSE={rmse:.2f} MAE={mae:.2f} "
              f"(train-mean, n={int(valid.sum())})")


def report(tag, pred, true, mask, cfg):
    line = []
    for j, p in enumerate(cfg["pollutants"]):
        valid = mask[:, j] > 0
        r = evaluation.metric_values(pred[valid, j], true[valid, j]) if valid.sum() else {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
        line.append(f"{DISPLAY[p]}: RMSE={r['rmse']:.2f} MAE={r['mae']:.2f} "
                    f"R2={r['r2']:+.3f} (n={r['n']})")
    print(f"  {tag:<10} " + "  |  ".join(line))


def main():
    _, model_config = selected_model(MODEL)
    result = result_paths(model_config["experiment"])
    checkpoint = result["final_checkpoint"]
    if not checkpoint.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}. Run 06_models/02_train_final.py first.")
    bundle = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    cfg, streams = bundle["cfg"], bundle["streams"]
    cfg["pollutants"] = bundle.get("pollutants", cfg["pollutants"])
    if "baseline_concentration" not in bundle:
        raise SystemExit("ERROR: checkpoint is missing arithmetic train-mean baseline metadata. Re-run final training.")
    build_model, _ = selected_model(MODEL)
    print(f"loaded {checkpoint}  |  model={MODEL}  |  streams={streams}  "
          f"aer_wide={cfg['use_aer_wide']} dem={cfg['use_dem']}  "
          f"pollutants={cfg['pollutants']}  tta={USE_TTA}\n")
    print(f"patch cache: {'enabled' if cfg.get('cache_patches', CACHE_PATCHES) else 'disabled'}")
    df = data.load_test_frame(streams, cfg)
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = False
    model = build_model(len(streams), model_cfg, len(cfg["pollutants"])).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    ds = data.EEA(df, streams, bundle["tmean"], bundle["tstd"], bundle["s5p_stats"], cfg, augment=False)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=cfg.get("num_workers", NUM_WORKERS), pin_memory=True)
    pred, true, mask = predict(model, ld, bundle["tmean"], bundle["tstd"], cfg, USE_TTA)
    stats = data.patch_cache_stats(cfg.get("cache_patches", CACHE_PATCHES))
    if stats["enabled"]:
        print(f"patch cache final: items={stats['items']} hits={stats['hits']} misses={stats['misses']}")

    print("\n" + "=" * 60 + "\nSEALED TEST SET (east/north German Laender)")
    baseline_report(true, mask, bundle["baseline_concentration"], cfg)
    for grp in sorted(df["group"].unique()):
        sel = (df["group"] == grp).values
        if sel.sum():
            report(grp, pred[sel], true[sel], mask[sel], cfg)

    keep_cols = [c for c in ["station_code", "country", "land", "fold", "lat", "lon", "group"] if c in df.columns]
    out = df[keep_cols].copy()
    for j, p in enumerate(cfg["pollutants"]):
        valid = mask[:, j] > 0
        out[f"pred_{p}"] = pred[:, j]
        out[f"true_{p}"] = np.where(valid, true[:, j], np.nan)
    result["test_predictions"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(result["test_predictions"], index=False)
    print(f"\nsaved {result['test_predictions']}")


if __name__ == "__main__":
    main()
