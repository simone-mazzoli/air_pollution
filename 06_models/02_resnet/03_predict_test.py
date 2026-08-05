"""
Inference-only: load a final_model.pt bundle and evaluate on the final NE Germany test set.
No training. Same normalization, de-normalization, and 8-view TTA as CV. 
Prints overall + per-group metrics and writes per-station predictions.

Run order: 00_assign_folds.py -> 01b_train_final.py -> 02_predict_test.py
  python3 02_predict_test.py --model final_model.pt
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

TRAIN_MODULE = Path(__file__).resolve().parent / "01_train_eea.py"
spec = importlib.util.spec_from_file_location("train_eea", TRAIN_MODULE)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
DEVICE = T.DEVICE


def load_test_frame(streams, cfg):
    """Annual PM for every station labelled TEST in station_fold.csv, outlier-
    filtered, with all patches present. Adds a 'group' column for per-region
    reporting: country code, except east/north German stations get 'DE_ne'."""
    if not T.STATION_FOLD.exists():
        raise SystemExit(
            f"ERROR: {T.STATION_FOLD} not found. Run 00_assign_folds.py first.")
    sf = pd.read_csv(T.STATION_FOLD, dtype={"station_code": str})
    test_codes = set(sf.loc[sf["fold"] == "TEST", "station_code"])
    # land per code (for splitting the German test stations out in the report)
    code_land = dict(zip(sf["station_code"], sf.get("land", pd.Series(dtype=str))))

    lab = pd.read_csv(T.LABELS, dtype={"station_code": str})
    g = lab.groupby("station_code")
    ann = pd.DataFrame({"pm10": g["PM10"].mean(), "pm25": g["PM2.5"].mean()}).reset_index()
    ann = ann[ann["station_code"].isin(test_codes)].reset_index(drop=True)
    ann["country"] = ann["station_code"].str[:2]

    def group_of(code):
        cc = code[:2]
        if cc == "DE":
            return "DE_ne"          # east/north DE (only DE in TEST)
        return cc
    ann["group"] = ann["station_code"].map(group_of)

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

    keep = ann["station_code"].map(has_all)
    print(f"{len(ann)} sealed-TEST stations -> {int(keep.sum())} with all patches")
    ann = ann[keep].reset_index(drop=True)
    for grp in sorted(ann["group"].unique()):
        sub = ann[ann["group"] == grp]
        print(f"    {grp:<6} {len(sub):>4}  (PM10 {int(sub['pm10'].notna().sum())}, "
              f"PM2.5 {int(sub['pm25'].notna().sum())})")
    return ann


@torch.no_grad()
def predict(model, loader, tmean, tstd, tta):
    model.eval(); P, T_, M = [], [], []
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
        P.append(out.cpu().numpy()); T_.append(y.numpy()); M.append(m.numpy())
    P = np.concatenate(P); T_ = np.concatenate(T_); M = np.concatenate(M)
    pred = np.exp(P * tstd + tmean)                      # (N, 2) ug/m3
    true = np.exp(T_ * tstd + tmean)
    return pred, true, M


def metrics(pred, true, valid):
    if valid.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    e = pred[valid] - true[valid]
    sst = np.sum((true[valid] - true[valid].mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "r2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan"),
            "n": int(valid.sum())}


def report(tag, pred, true, M):
    line = []
    for j, p in enumerate(T.POLLUTANTS):
        r = metrics(pred[:, j], true[:, j], M[:, j] > 0)
        line.append(f"{T.DISPLAY[p]}: RMSE={r['rmse']:.2f} MAE={r['mae']:.2f} "
                    f"R2={r['r2']:+.3f} (n={r['n']})")
    print(f"  {tag:<10} " + "  |  ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="final_model.pt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=True)
    ap.add_argument("--out", default="test_predictions.csv")
    args = ap.parse_args()

    bundle = torch.load(args.model, map_location=DEVICE, weights_only=False)
    cfg, streams = bundle["cfg"], bundle["streams"]
    tmean, tstd = bundle["tmean"], bundle["tstd"]
    s5p_stats = bundle["s5p_stats"]
    T.POLLUTANTS = bundle.get("pollutants", T.POLLUTANTS)   # match trained target
    print(f"loaded {args.model}  |  streams={streams}  "
          f"aer_wide={cfg['use_aer_wide']} dem={cfg['use_dem']}  "
          f"pollutants={T.POLLUTANTS}  tta={args.tta}\n")

    df = load_test_frame(streams, cfg)

    model = T.Net(len(streams), cfg, n_out=len(T.POLLUTANTS),
                  pretrained=False).to(DEVICE)   # weights come from the bundle
    model.load_state_dict(bundle["state_dict"])

    ds = T.EEA(df, streams, tmean, tstd, s5p_stats, cfg, augment=False)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    pred, true, M = predict(model, ld, tmean, tstd, args.tta)

    print("\n" + "=" * 60 + "\nSEALED TEST SET (east/north German Laender)")
    report("overall", pred, true, M)
    for grp in sorted(df["group"].unique()):
        sel = (df["group"] == grp).values
        if sel.sum():
            report(grp, pred[sel], true[sel], M[sel])

    out = df[["station_code", "country", "group"]].copy()
    for j, p in enumerate(T.POLLUTANTS):
        v = M[:, j] > 0
        out[f"pred_{p}"] = pred[:, j]
        out[f"true_{p}"] = np.where(v, true[:, j], np.nan)
    out.to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
