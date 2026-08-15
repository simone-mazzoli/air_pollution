"""
Runs the selected trained experiment's final checkpoint over the dense grid
(built by 04_GEE/02_download_dense_grid_patches.py) and
writes a continuous pollution-map PNG + a predictions CSV.
Model selection matches 03_predict_test.py: pass --experiment (and possibly
--wide.
Patch loading/normalization reuses shared/data.py's existing
load_s2_station / load_patch_raw_station / load_dem_station directly.
Grid points come from data/processed/eea/grid_points.csv (grid_id, lat, lon,
land), produced by 04_download_grid_patches.py. Patches live in
data/processed/satellite_grid/<stream>/<grid_id>.npy, mirroring
satellite_eea/ folder layout.
Target de-normalization mirrors shared/data.py's EEA.__getitem__ label
transform (y = (log(v) - tmean) / tstd) in reverse -- exp(pred*tstd+tmean).

Run order: 02_train_final.py -> 04_download_grid_patches.py -> this.
    python3 05_predict_grid.py --experiment cnn_deep --wide

outputs:
    grid_results/cnn_deep_wide_grid_predictions.csv 
    grid_results/cnn_deep_wide_grid_map.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_models"))

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from shared import data, paths, runtime
from shared.config import BATCH_SIZE, CACHE_PATCHES, DEVICE, DISPLAY, MODEL, NUM_WORKERS, USE_TTA, result_paths
from shared.models import SUPPORTED_EXPERIMENTS, require_single_experiment, selected_model


GRID_SAT = paths.PROC / "satellite_grid"
GRID_CSV = paths.PROC / "eea" / "grid_points.csv"
GHIGH = GRID_SAT / "high_res_multispec"
GLOW = GRID_SAT / "low_res_multispec"
GAERW = GRID_SAT / "aer_wide_tropomi"
GDEM = GRID_SAT / "dem_glo30"
def _has_all_grid_patches(gid, streams, cfg):
    if not ((GHIGH / f"{gid}.npy").exists() and (GLOW / f"{gid}.npy").exists()):
        return False
    if not all((GRID_SAT / st / f"{gid}.npy").exists() for st in streams):
        return False
    extra = ([GAERW] if cfg["use_aer_wide"] else []) + ([GDEM] if cfg["use_dem"] else [])
    return all((d / f"{gid}.npy").exists() for d in extra)
def load_grid_frame(streams, cfg):
    if not GRID_CSV.exists():
        raise SystemExit(f"ERROR: {GRID_CSV} not found. Run 04_download_grid_patches.py first.")
    pts = pd.read_csv(GRID_CSV, dtype={"grid_id": str})
    keep = pts["grid_id"].map(lambda gid: _has_all_grid_patches(gid, streams, cfg))
    print(f"{len(pts)} grid points -> {int(keep.sum())} with all patches")
    return pts[keep].reset_index(drop=True)
class GridEEA(Dataset):
    """Same patch loading/normalization as shared.data.EEA, keyed by grid_id
    instead of station_code, no labels/augmentation -- inference over new
    unlabeled locations."""
    def __init__(self, frame, streams, s5p_stats, cfg):
        self.f = frame.reset_index(drop=True)
        self.streams = streams
        self.s5p_stats = s5p_stats
        self.cfg = cfg
    def __len__(self):
        return len(self.f)
    def _norm_s5p(self, directory, modality, gid, key):
        m, sd = self.s5p_stats[key]
        cache_patches = self.cfg.get("cache_patches", CACHE_PATCHES)
        raw = data.load_patch_raw_station(directory, modality, gid, cache_patches)
        safe = np.where(np.isfinite(raw), raw, m)
        return ((safe - m) / sd if sd > 0 else np.zeros_like(safe)).astype("float32")
    def __getitem__(self, i):
        r = self.f.iloc[i]
        gid = r["grid_id"]
        cache_patches = self.cfg.get("cache_patches", CACHE_PATCHES)
        xh = torch.from_numpy(data.load_s2_station(GHIGH, "grid_high", gid, cache_patches))
        xl = torch.from_numpy(data.load_s2_station(GLOW, "grid_low", gid, cache_patches))
        chans = [self._norm_s5p(GRID_SAT / st, f"grid_{st}", gid, st) for st in self.streams]
        xs_patch = torch.from_numpy(np.stack(chans, axis=0))
        extras = []
        if self.cfg["use_aer_wide"]:
            w = self._norm_s5p(GAERW, "grid_aer_wide", gid, "aer_wide")
            xw = torch.from_numpy(w[None])
            extras.append(float(w[w.shape[0] // 2, w.shape[1] // 2]))
        else:
            xw = torch.zeros(1, 1, 1)
        if self.cfg["use_dem"]:
            relief, elev = data.load_dem_station(GDEM, "grid_dem", gid, cache_patches)
            xd = torch.from_numpy(relief)
            extras.append(elev / 1000.0)
        else:
            xd = torch.zeros(1, 1, 1)
        xs_mean = torch.tensor([float(c.mean()) for c in xs_patch] + extras, dtype=torch.float32)
        return gid, xh, xl, xs_patch, xw, xd, xs_mean
@torch.no_grad()
def predict_grid(model, loader, tmean, tstd, tta):
    model.eval()
    gids, preds = [], []
    for gid, xh, xl, xs_patch, xw, xd, xs_mean in loader:
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
        preds.append(out.cpu().numpy())
        gids.extend(gid)
    P = np.concatenate(preds)
    pred = np.exp(P * tstd + tmean)  # inverse of EEA.__getitem__'s log-normalized label transform
    return gids, pred
def make_map(df, pollutant, path):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(9, 9))

    # drop the longest-edge triangles so only real coverage gets colored
    tri = mtri.Triangulation(df["lon"].values, df["lat"].values)
    lon, lat = df["lon"].values, df["lat"].values
    edges = np.stack([
        np.hypot(lon[tri.triangles[:, i]] - lon[tri.triangles[:, j]],
                 lat[tri.triangles[:, i]] - lat[tri.triangles[:, j]])
        for i, j in ((0, 1), (1, 2), (2, 0))
    ], axis=1)
    max_edge = edges.max(axis=1)
    tri.set_mask(max_edge > np.percentile(max_edge, 95))

    sc = ax.tricontourf(tri, df[f"pred_{pollutant}"], levels=20, cmap="viridis", zorder=1)

    geojson = json.loads((paths.PROC / "germany_states.geojson").read_text())
    borders = []
    for feature in geojson["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            borders.append(Polygon(polygon[0], closed=True))
    ax.add_collection(PatchCollection(borders, facecolor="none", edgecolor="#555555",
                                      linewidths=0.6, zorder=2))

    fig.colorbar(sc, label=f"predicted {DISPLAY.get(pollutant, pollutant)} (ug/m3)")
    ax.set_title(f"Continuous {DISPLAY.get(pollutant, pollutant)} map -- sealed-test Laender")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(df["lat"].median()))))
    margin = 0.3
    ax.set_xlim(lon.min() - margin, lon.max() + margin)
    ax.set_ylim(lat.min() - margin, lat.max() + margin)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path}")
def parse_args():
    choices = "{" + ",".join(SUPPORTED_EXPERIMENTS) + "}"
    ap = argparse.ArgumentParser(description="Run the selected final model over the dense inference grid.")
    ap.add_argument("--experiment", default=MODEL, metavar=choices)
    ap.add_argument("--wide", action="store_true",
                    help="use wider scratch-CNN channels with --experiment cnn or cnn_deep")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=USE_TTA)
    ap.add_argument("--out", default="grid_results/grid_predictions.csv")
    ap.add_argument("--plot", default="grid_results/grid_map.png")
    return ap.parse_args()
def main():
    runtime.apply_runtime_config()
    print(runtime.runtime_summary())
    args = parse_args()
    experiment_name = require_single_experiment(args.experiment, "Grid inference", wide=args.wide)
    _, model_config = selected_model(experiment_name, wide=args.wide)
    result = result_paths(model_config["experiment"])
    checkpoint = result["final_checkpoint"]
    if not checkpoint.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}. Run 02_train_final.py first.")
    bundle = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    cfg, streams = bundle["cfg"], bundle["streams"]
    cfg["pollutants"] = bundle.get("pollutants", cfg["pollutants"])
    tmean, tstd = bundle["tmean"], bundle["tstd"]
    s5p_stats = bundle["s5p_stats"]
    pollutants = cfg["pollutants"]
    build_model, _ = selected_model(experiment_name, wide=args.wide)
    print(f"loaded {checkpoint}  |  experiment={model_config['experiment']}  |  streams={streams}  "
          f"aer_wide={cfg['use_aer_wide']} dem={cfg['use_dem']}  "
          f"pollutants={pollutants}  tta={args.tta}\n")
    frame = load_grid_frame(streams, cfg)
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = False
    model = build_model(len(streams), model_cfg, len(pollutants)).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    ds = GridEEA(frame, streams, s5p_stats, cfg)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=cfg.get("num_workers", NUM_WORKERS), pin_memory=True)
    gids, pred = predict_grid(model, ld, tmean, tstd, args.tta)
    stats = data.patch_cache_stats(cfg.get("cache_patches", CACHE_PATCHES))
    if stats["enabled"]:
        print(f"patch cache: {stats['items']} unique patches loaded "
              f"({stats['hits']} reused, {stats['misses']} loaded from disk)")
    out = frame.set_index("grid_id").loc[gids].reset_index()
    for j, p in enumerate(pollutants):
        out[f"pred_{p}"] = pred[:, j]
    out_name = Path(args.out).parent / f"{model_config['experiment']}_{Path(args.out).name}"
    out_name.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_name, index=False)
    print(f"saved {out_name}")
    plot_name = Path(args.plot).parent / f"{model_config['experiment']}_{Path(args.plot).name}"
    make_map(out, pollutants[0], plot_name)
if __name__ == "__main__":
    main()
