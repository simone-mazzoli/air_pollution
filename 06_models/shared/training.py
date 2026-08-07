import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data, evaluation
from .config import DEVICE, DISPLAY


def masked_loss(pred, y, mask, lossf):
    per = lossf(pred, y)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def trainable_parameters(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    return params


def optimizer_parameter_groups(model, cfg):
    if hasattr(model, "optimizer_parameter_groups"):
        groups = model.optimizer_parameter_groups(cfg)
    else:
        groups = [{"params": trainable_parameters(model),
                   "lr": cfg["lr_head"], "weight_decay": cfg["wd_head"]}]
    grouped = [p for g in groups for p in g["params"]]
    trainable = trainable_parameters(model)
    if len({id(p) for p in grouped}) != len(grouped):
        raise ValueError("optimizer parameter groups contain duplicate parameters")
    if {id(p) for p in grouped} != {id(p) for p in trainable}:
        raise ValueError("optimizer parameter groups do not match trainable parameters")
    return groups


def model_metadata(model, cfg):
    meta = {
        "model": cfg.get("model"),
        "experiment": cfg.get("experiment"),
    }
    if hasattr(model, "parameter_metadata"):
        meta.update(model.parameter_metadata())
    return meta


def train_loader(dataset, cfg):
    n = len(dataset)
    batch = cfg["batch"]
    if n == 0:
        raise ValueError("training dataset has zero samples")
    if batch <= 1 or n == 1:
        raise ValueError("BatchNorm training needs batches with at least two samples")
    drop_last = n % batch == 1
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=True, drop_last=drop_last, worker_init_fn=data.worker_init)
    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError(f"training would have zero batches: n={n}, batch={batch}, drop_last={drop_last}")
    return loader, {
        "n_train_samples": n,
        "n_train_batches": n_batches,
        "effective_drop_last": drop_last,
    }


def final_epochs_from_cv(cv_results_path, expected_folds):
    path = cv_results_path
    if not path.exists():
        raise SystemExit(f"ERROR: CV results not found: {path}. Run 06_models/01_train_cv.py first.")
    cv = json.loads(path.read_text())
    missing = [f for f in expected_folds if f not in cv or "best_epoch" not in cv[f]]
    if missing:
        raise SystemExit(f"ERROR: CV results are incomplete; missing best_epoch for: {missing}")
    best_epochs = [int(cv[f]["best_epoch"]) for f in expected_folds]
    median = float(np.median(best_epochs))
    final_epochs = int(math.ceil(median))
    return final_epochs, best_epochs, "median_cv_best_epoch_ceil"


def train_one_fold(train_df, val_df, streams, cfg, build_model):
    data.seed_everything()
    pollutants = cfg["pollutants"]
    tmean = np.array([np.nanmean(np.log(train_df[p].values)) for p in pollutants], "float64")
    tstd = np.array([np.nanstd(np.log(train_df[p].values)) or 1.0 for p in pollutants], "float64")
    s5p_stats = data.compute_s5p_stats(train_df, streams, cfg)
    tr, loader_info = train_loader(data.EEA(train_df, streams, tmean, tstd, s5p_stats, cfg, augment=True), cfg)
    va = DataLoader(data.EEA(val_df, streams, tmean, tstd, s5p_stats, cfg),
                    batch_size=cfg["batch"], shuffle=False, num_workers=cfg["num_workers"],
                    pin_memory=True, worker_init_fn=data.worker_init)
    tsub = train_df.sample(min(1000, len(train_df)), random_state=0)
    tm = DataLoader(data.EEA(tsub, streams, tmean, tstd, s5p_stats, cfg),
                    batch_size=cfg["batch"], shuffle=False, num_workers=cfg["num_workers"],
                    pin_memory=True, worker_init_fn=data.worker_init)
    base = evaluation.constant_baseline(train_df, val_df, cfg)
    print("  baseline (train mean): " + "  ".join(
        f"{DISPLAY[p]} mean={base[p]['mean']:.2f} RMSE={base[p]['rmse']:.2f}" for p in pollutants))
    model = build_model(len(streams), cfg, len(pollutants)).to(DEVICE)
    counts = parameter_counts(model)
    meta = model_metadata(model, cfg)
    print(f"  parameters: total={counts['total']} trainable={counts['trainable']} frozen={counts['frozen']}")
    print(f"  batches: n_train_samples={loader_info['n_train_samples']} "
          f"n_train_batches={loader_info['n_train_batches']} "
          f"effective_drop_last={loader_info['effective_drop_last']}")
    opt = torch.optim.AdamW(optimizer_parameter_groups(model, cfg))
    lossf = nn.SmoothL1Loss(reduction="none")

    def mean_val_rmse(res):
        vals = [res[p]["rmse"] for p in pollutants if not np.isnan(res[p]["rmse"])]
        return float(np.mean(vals)) if vals else np.inf

    best, best_state, best_epoch, bad, epochs_run = np.inf, None, None, 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        epochs_run = ep
        model.train()
        tot = 0.0
        for xh, xl, xs_patch, xw, xd, xs_mean, y, m in tr:
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
                xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
                xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
            opt.zero_grad()
            loss = masked_loss(model(xh, xl, xs_patch, xw, xd, xs_mean), y, m, lossf)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xh)
        trm, _ = evaluation.evaluate(model, tm, tmean, tstd, cfg)
        val, _ = evaluation.evaluate(model, va, tmean, tstd, cfg, tta=cfg["tta"])
        r = mean_val_rmse(val)
        flag = ""
        if r < best - 1e-4:
            best = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
            flag = " *"
        else:
            bad += 1
        print(f"  [{ep:02d}] loss={tot/len(train_df):.3f}")
        print(f"       TRAIN  {evaluation.fmt_metrics(trm, cfg)}")
        print(f"       VAL    {evaluation.fmt_metrics(val, cfg)}{flag}")
        if bad >= cfg["patience"]:
            print("  early stop")
            break
    model.load_state_dict(best_state)
    val, arrs = evaluation.evaluate(model, va, tmean, tstd, cfg, tta=cfg["tta"])
    out = {"n_train": len(train_df), "n_val": len(val_df),
           "best_epoch": best_epoch, "epochs_run": epochs_run,
           "best_validation_metric": best,
           "epoch_numbering": "one_based", **loader_info,
           "parameter_counts": counts, **meta}
    for p in pollutants:
        out[p] = {
            "rmse": val[p]["rmse"],
            "mae": val[p]["mae"],
            "r2": val[p]["r2"],
            "n": val[p]["n"],
            "baseline": base[p]["rmse"],
            "baseline_concentration": base[p]["mean"],
            "baseline_metrics": {k: base[p][k] for k in ("rmse", "mae", "r2", "n")},
        }
    return out, arrs
