import json
import math
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data, evaluation, experiment
from .config import CACHE_PATCHES, DEVICE, DISPLAY


def masked_loss(pred, y, mask, lossf):
    per = lossf(pred, y)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def masked_loss_sum_count(pred, y, mask, lossf):
    per = lossf(pred, y)
    return (per * mask).sum(), mask.sum()


@torch.no_grad()
def objective_loss(model, loader, lossf):
    model.eval()
    loss_sum = torch.tensor(0.0, device=DEVICE)
    valid_count = torch.tensor(0.0, device=DEVICE)
    for xh, xl, xs_patch, xw, xd, xs_mean, y, m in loader:
        xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
            xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
            xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
        batch_sum, batch_count = masked_loss_sum_count(
            model(xh, xl, xs_patch, xw, xd, xs_mean), y, m, lossf)
        loss_sum += batch_sum
        valid_count += batch_count
    return float((loss_sum / valid_count.clamp(min=1)).cpu())


def sync_if_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def elapsed(start):
    return time.perf_counter() - start


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
        lr = cfg["lr"] if "lr" in cfg else cfg["lr_head"]
        weight_decay = cfg["weight_decay"] if "weight_decay" in cfg else cfg["wd_head"]
        groups = [{"params": trainable_parameters(model),
                   "lr": lr, "weight_decay": weight_decay}]
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


def final_epoch_summary_from_cv(cv_results_path, expected_folds):
    path = cv_results_path
    if not path.exists():
        raise SystemExit(f"ERROR: CV results not found: {path}. Run 06_models/01_train_cv.py first.")
    cv = json.loads(path.read_text())
    missing = [f for f in expected_folds if f not in cv or "best_epoch" not in cv[f]]
    if missing:
        raise SystemExit(f"ERROR: CV results are incomplete; missing best_epoch for: {missing}")
    fold_best_epochs = [(f, int(cv[f]["best_epoch"])) for f in expected_folds]
    best_epochs = [epoch for _, epoch in fold_best_epochs]
    median = float(np.median(best_epochs))
    final_epochs = int(math.ceil(median))
    return {
        "final_epochs": final_epochs,
        "best_epochs": best_epochs,
        "fold_best_epochs": fold_best_epochs,
        "median_best_epoch": median,
        "epoch_selection_rule": "median_cv_best_epoch_ceil",
    }


def final_epochs_from_cv(cv_results_path, expected_folds):
    summary = final_epoch_summary_from_cv(cv_results_path, expected_folds)
    return summary["final_epochs"], summary["best_epochs"], summary["epoch_selection_rule"]


def train_one_fold(train_df, val_df, streams, cfg, build_model, *, fold=None, history_path=None):
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
    print(f"  patch cache: {'enabled' if cfg.get('cache_patches', CACHE_PATCHES) else 'disabled'}")
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
        sync_if_cuda()
        epoch_start = time.perf_counter()
        model.train()
        train_start = time.perf_counter()
        train_detail = {k: 0.0 for k in ("data_wait", "to_device", "forward", "backward", "optimizer")}
        train_iter = iter(tr)
        train_loss_sum = 0.0
        train_loss_count = 0.0
        for _ in range(len(tr)):
            start = time.perf_counter()
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = next(train_iter)
            train_detail["data_wait"] += elapsed(start)

            sync_if_cuda()
            start = time.perf_counter()
            xh, xl, xs_patch, xw, xd, xs_mean, y, m = (
                xh.to(DEVICE), xl.to(DEVICE), xs_patch.to(DEVICE), xw.to(DEVICE),
                xd.to(DEVICE), xs_mean.to(DEVICE), y.to(DEVICE), m.to(DEVICE))
            sync_if_cuda()
            train_detail["to_device"] += elapsed(start)

            start = time.perf_counter()
            opt.zero_grad()
            sync_if_cuda()
            train_detail["optimizer"] += elapsed(start)

            start = time.perf_counter()
            pred = model(xh, xl, xs_patch, xw, xd, xs_mean)
            batch_loss_sum, batch_loss_count = masked_loss_sum_count(pred, y, m, lossf)
            loss = batch_loss_sum / batch_loss_count.clamp(min=1)
            sync_if_cuda()
            train_detail["forward"] += elapsed(start)

            start = time.perf_counter()
            loss.backward()
            sync_if_cuda()
            train_detail["backward"] += elapsed(start)

            start = time.perf_counter()
            opt.step()
            sync_if_cuda()
            train_detail["optimizer"] += elapsed(start)
            train_loss_sum += float(batch_loss_sum.detach().cpu())
            train_loss_count += float(batch_loss_count.detach().cpu())
        sync_if_cuda()
        train_seconds = time.perf_counter() - train_start
        train_eval_start = time.perf_counter()
        trm, _ = evaluation.evaluate(model, tm, tmean, tstd, cfg)
        sync_if_cuda()
        train_eval_seconds = time.perf_counter() - train_eval_start
        val_loss_start = time.perf_counter()
        val_loss = objective_loss(model, va, lossf)
        sync_if_cuda()
        val_loss_seconds = time.perf_counter() - val_loss_start
        val_start = time.perf_counter()
        val, _ = evaluation.evaluate(model, va, tmean, tstd, cfg, tta=cfg["tta"])
        sync_if_cuda()
        val_seconds = time.perf_counter() - val_start
        total_seconds = time.perf_counter() - epoch_start
        r = mean_val_rmse(val)
        flag = ""
        best_so_far = False
        if r < best - 1e-4:
            best = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad = 0
            flag = " *"
            best_so_far = True
        else:
            bad += 1
        train_loss = train_loss_sum / max(train_loss_count, 1.0)
        if history_path is not None:
            row = experiment.epoch_history_row(
                cfg, fold, ep, train_loss, trm, val, opt,
                {
                    "train_seconds": train_seconds,
                    "train_batch_avg_seconds": train_seconds / len(tr),
                    "train_eval_seconds": train_eval_seconds,
                    "val_loss_seconds": val_loss_seconds,
                    "val_seconds": val_seconds,
                    "total_seconds": total_seconds,
                    "data_wait_seconds": train_detail["data_wait"],
                    "to_device_seconds": train_detail["to_device"],
                    "forward_seconds": train_detail["forward"],
                    "backward_seconds": train_detail["backward"],
                    "optimizer_seconds": train_detail["optimizer"],
                },
                best_so_far,
                bad,
            )
            row["val_loss"] = val_loss
            experiment.append_csv_row(history_path, row)
        print(f"  [{ep:02d}] loss={train_loss:.3f} val_loss={val_loss:.3f}")
        print(f"       TRAIN  {evaluation.fmt_metrics(trm, cfg)}")
        print(f"       VAL    {evaluation.fmt_metrics(val, cfg)}{flag}")
        print(f"       timing: train={train_seconds:.1f}s "
              f"train_batch_avg={train_seconds / len(tr):.2f}s "
              f"train_eval={train_eval_seconds:.1f}s "
              f"val={val_seconds:.1f}s total={total_seconds:.1f}s")
        print("       train detail: " + " ".join(
            f"{k}={v:.1f}s/{v / len(tr):.2f}s"
            for k, v in train_detail.items()))
        stats = data.patch_cache_stats(cfg.get("cache_patches", CACHE_PATCHES))
        if stats["enabled"]:
            print(f"       patch cache: items={stats['items']} hits={stats['hits']} misses={stats['misses']}")
        if bad >= cfg["patience"]:
            print("  early stop")
            break
    model.load_state_dict(best_state)
    val, arrs = evaluation.evaluate(model, va, tmean, tstd, cfg, tta=cfg["tta"])
    stats = data.patch_cache_stats(cfg.get("cache_patches", CACHE_PATCHES))
    if stats["enabled"]:
        print(f"  patch cache final: items={stats['items']} hits={stats['hits']} misses={stats['misses']}")
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
