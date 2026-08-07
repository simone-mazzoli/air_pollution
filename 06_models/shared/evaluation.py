import numpy as np
import torch

from .config import DEVICE, DISPLAY


@torch.no_grad()
def evaluate(model, loader, tmean, tstd, cfg, tta=False):
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
    p = np.concatenate(pred_batches)
    t = np.concatenate(true_batches)
    m = np.concatenate(mask_batches)
    return metrics_by_pollutant(p, t, m, tmean, tstd, cfg), arrays_by_pollutant(p, t, m, tmean, tstd, cfg)


def arrays_by_pollutant(pred_norm, true_norm, mask, tmean, tstd, cfg):
    out = {}
    for j, pollutant in enumerate(cfg["pollutants"]):
        sel = mask[:, j] > 0
        if sel.sum():
            pred = np.exp(pred_norm[sel, j] * tstd[j] + tmean[j])
            true = np.exp(true_norm[sel, j] * tstd[j] + tmean[j])
            out[pollutant] = (pred, true)
    return out


def metrics_by_pollutant(pred_norm, true_norm, mask, tmean, tstd, cfg):
    out = {}
    for j, pollutant in enumerate(cfg["pollutants"]):
        sel = mask[:, j] > 0
        if sel.sum() == 0:
            out[pollutant] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
            continue
        pred = np.exp(pred_norm[sel, j] * tstd[j] + tmean[j])
        true = np.exp(true_norm[sel, j] * tstd[j] + tmean[j])
        out[pollutant] = metric_values(pred, true)
    return out


def metric_values(pred, true):
    err = pred - true
    sst = np.sum((true - true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"),
        "n": int(len(true)),
    }


def constant_baseline(train_df, eval_df, cfg):
    out = {}
    for p in cfg["pollutants"]:
        train_values = train_df[p].values
        mean_value = float(np.nanmean(train_values))
        eval_values = eval_df[p].values
        valid = ~np.isnan(eval_values)
        metrics = metric_values(np.full(valid.sum(), mean_value), eval_values[valid]) if valid.sum() else {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
        out[p] = {"mean": mean_value, **metrics}
    return out


def prediction_table(frame, arrs, cfg):
    rows = []
    meta_cols = [c for c in ["station_code", "country", "land", "lat", "lon", "fold"] if c in frame.columns]
    for p in cfg["pollutants"]:
        if p not in arrs:
            continue
        pred, true = arrs[p]
        valid_frame = frame.loc[frame[p].notna() & (frame[p] > 0), meta_cols].reset_index(drop=True)
        if len(valid_frame) != len(pred):
            raise ValueError(f"prediction alignment failed for {p}: {len(valid_frame)} rows vs {len(pred)} predictions")
        block = valid_frame.copy()
        block["pollutant"] = p
        block["true"] = true
        block["pred"] = pred
        rows.append(block)
    return rows


def metrics_from_prediction_table(predictions, cfg):
    out = {}
    for p in cfg["pollutants"]:
        sub = predictions[predictions["pollutant"] == p]
        out[p] = metric_values(sub["pred"].to_numpy(), sub["true"].to_numpy()) if len(sub) else {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    return out


def denormalize(pred_norm, true_norm, tmean, tstd):
    return np.exp(pred_norm * tstd + tmean), np.exp(true_norm * tstd + tmean)


def fmt_metrics(res, cfg):
    return "  |  ".join(
        f"{DISPLAY[p]}: RMSE={res[p]['rmse']:.2f} MAE={res[p]['mae']:.2f} R2={res[p]['r2']:.3f}"
        for p in cfg["pollutants"]
    )


def baseline_rmse(train_df, val_df, cfg):
    return {p: v["rmse"] for p, v in constant_baseline(train_df, val_df, cfg).items()}
