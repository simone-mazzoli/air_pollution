#!/usr/bin/env python3
"""Post-hoc diagnostics for the frozen cnn_deep_wide sealed-TEST predictions."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "06_models" / "results" / "cnn_deep_wide" / "test_predictions.csv"
CHECKPOINT = ROOT / "06_models" / "results" / "cnn_deep_wide" / "final_model.pt"
OUT = ROOT / "07_prediction_analysis" / "outputs"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
BOUNDARIES = ROOT / "07_prediction_analysis" / "boundaries"
COUNTRY_BOUNDARIES = BOUNDARIES / "ne_50m_admin_0_countries.geojson"
STATE_BOUNDARIES = BOUNDARIES / "geoboundaries_deu_adm1.geojson"
POLLUTANT = "pm25"
UNITS = "µg/m³"
BOUNDARY_SOURCES = {
    "countries": {
        "path": COUNTRY_BOUNDARIES,
        "label": "Natural Earth 1:50m Admin 0 Countries",
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
    },
    "german_states": {
        "path": STATE_BOUNDARIES,
        "label": "geoBoundaries Open DEU ADM1",
        "url": "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/DEU/ADM1/geoBoundaries-DEU-ADM1.geojson",
    },
}
MAP_FIGURE_NAMES = {
    "test_observed_predicted_maps.png",
    "test_observed_map.png",
    "test_predicted_map.png",
    "test_residual_map.png",
    "test_absolute_error_map.png",
}


def first_existing(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def prediction_columns(df):
    cols = set(df.columns)
    return {
        "observed": first_existing(cols, [f"true_{POLLUTANT}", f"observed_{POLLUTANT}", POLLUTANT, "true", "observed"]),
        "predicted": first_existing(cols, [f"pred_{POLLUTANT}", f"predicted_{POLLUTANT}", "pred", "predicted"]),
        "lat": first_existing(cols, ["lat", "latitude"]),
        "lon": first_existing(cols, ["lon", "longitude"]),
        "station_id": first_existing(cols, ["station_code", "station_id", "location", "id"]),
        "state": first_existing(cols, ["land", "state", "region", "bundesland"]),
        "station_type": first_existing(cols, ["station_type", "type", "station_category", "site_type"]),
    }


def require_columns(found):
    missing = [name for name in ("observed", "predicted") if found[name] is None]
    if missing:
        raise SystemExit(f"ERROR: prediction file is missing required columns: {missing}")


def metric_values(pred, obs):
    err = pred - obs
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": int(len(obs)),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"),
    }


def project_lonlat(lon, lat):
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    radius_km = 6371.0
    lon0 = np.deg2rad(10.0)
    lat0 = np.deg2rad(52.0)
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    denom = 1 + np.sin(lat0) * np.sin(phi) + np.cos(lat0) * np.cos(phi) * np.cos(lam - lon0)
    k = np.sqrt(2 / np.maximum(denom, 1e-12))
    x = radius_km * k * np.cos(phi) * np.sin(lam - lon0)
    y = radius_km * k * (np.cos(lat0) * np.sin(phi) - np.sin(lat0) * np.cos(phi) * np.cos(lam - lon0))
    return x, y


def download_boundary_files():
    BOUNDARIES.mkdir(parents=True, exist_ok=True)
    for source in BOUNDARY_SOURCES.values():
        path = source["path"]
        if path.exists():
            continue
        print(f"downloading boundary data once: {source['label']}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                tmp.write_bytes(response.read())
            tmp.replace(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            print(f"warning: could not download {source['label']}: {exc}")


def geometry_rings(geometry):
    if not geometry:
        return
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"]:
            yield ring
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            for ring in polygon:
                yield ring


def ring_overlaps_bbox(ring, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    arr = np.asarray(ring, dtype="float64")
    if arr.size == 0:
        return False
    return (
        arr[:, 0].max() >= lon_min
        and arr[:, 0].min() <= lon_max
        and arr[:, 1].max() >= lat_min
        and arr[:, 1].min() <= lat_max
    )


def geojson_segments(path, bbox):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    segments = []
    for feature in data.get("features", []):
        for ring in geometry_rings(feature.get("geometry")):
            if len(ring) < 2 or not ring_overlaps_bbox(ring, bbox):
                continue
            arr = np.asarray(ring, dtype="float64")
            x, y = project_lonlat(arr[:, 0], arr[:, 1])
            segments.append(np.column_stack([x, y]))
    return segments


def lonlat_bbox(geo):
    lon_pad = max((geo["lon"].max() - geo["lon"].min()) * 0.25, 1.6)
    lat_pad = max((geo["lat"].max() - geo["lat"].min()) * 0.25, 1.0)
    return (
        float(geo["lon"].min() - lon_pad),
        float(geo["lon"].max() + lon_pad),
        float(geo["lat"].min() - lat_pad),
        float(geo["lat"].max() + lat_pad),
    )


def load_map_context(geo, *, download_missing=True):
    if download_missing and (not COUNTRY_BOUNDARIES.exists() or not STATE_BOUNDARIES.exists()):
        download_boundary_files()
    bbox = lonlat_bbox(geo)
    countries = geojson_segments(COUNTRY_BOUNDARIES, bbox)
    states = geojson_segments(STATE_BOUNDARIES, bbox)
    if not countries and not states:
        print("warning: no boundary GeoJSON files available; station maps will not include geographic boundaries")
    else:
        used = []
        if countries:
            used.append(BOUNDARY_SOURCES["countries"]["label"])
        if states:
            used.append(BOUNDARY_SOURCES["german_states"]["label"])
        print("map boundary source: " + "; ".join(used))
    return {"countries": countries, "states": states}


def remove_stale_map_figures(reason):
    removed = []
    for name in MAP_FIGURE_NAMES:
        path = FIGURES / name
        if path.exists():
            path.unlink()
            removed.append(path)
    print(f"skipping geographic maps: {reason}")
    if removed:
        print("removed stale map figures:")
        for path in removed:
            try:
                display_path = path.relative_to(ROOT)
            except ValueError:
                display_path = path
            print(f"  {display_path}")


def load_training_mean_baseline():
    if not CHECKPOINT.exists():
        return None
    try:
        import torch

        bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"warning: could not read training-mean baseline from {CHECKPOINT}: {exc}")
        return None
    baseline = bundle.get("baseline_concentration", {})
    value = baseline.get(POLLUTANT)
    return float(value) if value is not None else None


def finite_analysis_frame(df, found):
    out = df.copy()
    out["observed"] = pd.to_numeric(out[found["observed"]], errors="coerce")
    out["predicted"] = pd.to_numeric(out[found["predicted"]], errors="coerce")
    out = out.dropna(subset=["observed", "predicted"]).copy()
    out["residual"] = out["predicted"] - out["observed"]
    out["absolute_error"] = out["residual"].abs()
    return out


def save_scatter(df, metrics):
    path = FIGURES / "test_observed_vs_predicted.png"
    lo = float(min(df["observed"].min(), df["predicted"].min()))
    hi = float(max(df["observed"].max(), df["predicted"].max()))
    pad = max((hi - lo) * 0.06, 0.25)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    ax.scatter(df["observed"], df["predicted"], s=34, alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Observed annual PM2.5 [{UNITS}]")
    ax.set_ylabel(f"Predicted annual PM2.5 [{UNITS}]")
    ax.text(
        0.04,
        0.96,
        f"n={metrics['n']}\nRMSE={metrics['rmse']:.2f}\nMAE={metrics['mae']:.2f}\nR²={metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_distributions(df):
    path = FIGURES / "test_prediction_distributions.png"
    values = pd.concat([df["observed"], df["predicted"]])
    bins = np.linspace(values.min(), values.max(), 18)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(df["observed"], bins=bins, alpha=0.55, label="Observed", color="#2f6f9f")
    ax.hist(df["predicted"], bins=bins, alpha=0.55, label="Predicted", color="#d9822b")
    ax.set_xlabel(f"Annual PM2.5 [{UNITS}]")
    ax.set_ylabel("Number of TEST stations")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_residual_distribution(df):
    path = FIGURES / "test_residual_distribution.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(df["residual"], bins=20, color="#6d5f9f", alpha=0.75)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel(f"Residual: prediction - observation [{UNITS}]")
    ax.set_ylabel("Number of TEST stations")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def projected_extent(geo):
    x, y = project_lonlat(geo["lon"], geo["lat"])
    x_pad = max((x.max() - x.min()) * 0.25, 80.0)
    y_pad = max((y.max() - y.min()) * 0.25, 80.0)
    return (float(x.min() - x_pad), float(x.max() + x_pad), float(y.min() - y_pad), float(y.max() + y_pad))


def draw_boundaries(ax, map_context):
    if map_context.get("countries"):
        ax.add_collection(LineCollection(map_context["countries"], colors="#b8b8b8", linewidths=0.65, zorder=1))
    if map_context.get("states"):
        ax.add_collection(LineCollection(map_context["states"], colors="#5f6368", linewidths=0.55, zorder=2))


def plot_map_panel(ax, df, value_col, title, *, cmap, colorbar_label, extent, map_context, vmin=None, vmax=None, norm=None):
    draw_boundaries(ax, map_context)
    x, y = project_lonlat(df["lon"], df["lat"])
    sc = ax.scatter(
        x,
        y,
        c=df[value_col],
        s=38,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        edgecolor="black",
        linewidth=0.25,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def save_station_map(df, value_col, title, path, *, cmap, colorbar_label, extent, map_context, vmin=None, vmax=None, norm=None):
    fig, ax = plt.subplots(figsize=(6.4, 6.8), dpi=160)
    sc = plot_map_panel(
        ax,
        df,
        value_col,
        title,
        cmap=cmap,
        colorbar_label=colorbar_label,
        extent=extent,
        map_context=map_context,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_observed_predicted_pair(df, conc_min, conc_max, extent, map_context):
    path = FIGURES / "test_observed_predicted_maps.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), dpi=160, constrained_layout=True)
    last = None
    for ax, value_col, title in [
        (axes[0], "observed", "Observed annual PM2.5"),
        (axes[1], "predicted", "Predicted annual PM2.5"),
    ]:
        last = plot_map_panel(
            ax,
            df,
            value_col,
            title,
            cmap="viridis",
            colorbar_label=f"Annual PM2.5 [{UNITS}]",
            extent=extent,
            map_context=map_context,
            vmin=conc_min,
            vmax=conc_max,
        )
    cb = fig.colorbar(last, ax=axes, shrink=0.78)
    cb.set_label(f"Annual PM2.5 [{UNITS}]")
    fig.suptitle("Sealed TEST stations: observed vs predicted", y=1.02)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_maps(df, found, *, map_context=None, download_missing_boundaries=True):
    if found["lat"] is None or found["lon"] is None:
        remove_stale_map_figures("prediction file has no station coordinates")
        return []
    geo = df.copy()
    geo["lat"] = pd.to_numeric(geo[found["lat"]], errors="coerce")
    geo["lon"] = pd.to_numeric(geo[found["lon"]], errors="coerce")
    geo = geo.dropna(subset=["lat", "lon"])
    if geo.empty:
        remove_stale_map_figures("station coordinates are empty after numeric parsing")
        return []

    map_context = map_context if map_context is not None else load_map_context(geo, download_missing=download_missing_boundaries)
    if not map_context.get("countries") and not map_context.get("states"):
        remove_stale_map_figures("no cached boundary GeoJSON files are available")
        return []
    extent = projected_extent(geo)
    conc_min = float(min(geo["observed"].min(), geo["predicted"].min()))
    conc_max = float(max(geo["observed"].max(), geo["predicted"].max()))
    residual_abs = max(float(np.nanmax(np.abs(geo["residual"]))), 1e-6)
    ae_max = float(geo["absolute_error"].max())
    return [
        save_observed_predicted_pair(geo, conc_min, conc_max, extent, map_context),
        save_station_map(
            geo,
            "observed",
            "Observed TEST station annual PM2.5",
            FIGURES / "test_observed_map.png",
            cmap="viridis",
            colorbar_label=f"Annual PM2.5 [{UNITS}]",
            extent=extent,
            map_context=map_context,
            vmin=conc_min,
            vmax=conc_max,
        ),
        save_station_map(
            geo,
            "predicted",
            "Predicted TEST station annual PM2.5",
            FIGURES / "test_predicted_map.png",
            cmap="viridis",
            colorbar_label=f"Annual PM2.5 [{UNITS}]",
            extent=extent,
            map_context=map_context,
            vmin=conc_min,
            vmax=conc_max,
        ),
        save_station_map(
            geo,
            "residual",
            "TEST residual: positive = overprediction",
            FIGURES / "test_residual_map.png",
            cmap="coolwarm",
            colorbar_label=f"Residual: prediction - observation [{UNITS}]",
            extent=extent,
            map_context=map_context,
            norm=TwoSlopeNorm(vmin=-residual_abs, vcenter=0, vmax=residual_abs),
        ),
        save_station_map(
            geo,
            "absolute_error",
            "TEST absolute error",
            FIGURES / "test_absolute_error_map.png",
            cmap="magma",
            colorbar_label=f"Absolute error [{UNITS}]",
            extent=extent,
            map_context=map_context,
            vmin=0,
            vmax=ae_max,
        ),
    ]


def summary_row(df, metrics, training_mean):
    obs = df["observed"]
    pred = df["predicted"]
    residual = df["residual"]
    if training_mean is None:
        baseline_mae = baseline_rmse = baseline_improve_abs = baseline_improve_pct = float("nan")
    else:
        b_err = training_mean - obs
        baseline_mae = float(np.mean(np.abs(b_err)))
        baseline_rmse = float(np.sqrt(np.mean(b_err ** 2)))
        baseline_improve_abs = baseline_rmse - metrics["rmse"]
        baseline_improve_pct = 100 * baseline_improve_abs / baseline_rmse if baseline_rmse else float("nan")
    test_mean = float(obs.mean())
    tm_err = test_mean - obs
    return {
        "n": metrics["n"],
        "observed_mean": float(obs.mean()),
        "observed_std": float(obs.std(ddof=1)),
        "observed_min": float(obs.min()),
        "observed_max": float(obs.max()),
        "predicted_mean": float(pred.mean()),
        "predicted_std": float(pred.std(ddof=1)),
        "predicted_min": float(pred.min()),
        "predicted_max": float(pred.max()),
        "bias": metrics["bias"],
        "median_residual": float(residual.median()),
        "residual_std": float(residual.std(ddof=1)),
        "fraction_overpredicted": float((residual > 0).mean()),
        "fraction_underpredicted": float((residual < 0).mean()),
        "fraction_exact": float((residual == 0).mean()),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "training_mean_baseline_value": training_mean,
        "training_mean_baseline_mae": baseline_mae,
        "training_mean_baseline_rmse": baseline_rmse,
        "training_mean_baseline_rmse_improvement": baseline_improve_abs,
        "training_mean_baseline_rmse_improvement_pct": baseline_improve_pct,
        "test_mean_reference_value": test_mean,
        "test_mean_reference_mae": float(np.mean(np.abs(tm_err))),
        "test_mean_reference_rmse": float(np.sqrt(np.mean(tm_err ** 2))),
    }


def save_summary(df, metrics, training_mean):
    row = summary_row(df, metrics, training_mean)
    path = TABLES / "test_summary.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path, row


def save_by_state(df, found):
    if found["state"] is None:
        return None
    rows = []
    for state, sub in df.groupby(found["state"], dropna=True):
        if pd.isna(state) or str(state).strip() == "":
            continue
        m = metric_values(sub["predicted"].to_numpy(), sub["observed"].to_numpy())
        rows.append({
            "state": state,
            "n": m["n"],
            "observed_mean": float(sub["observed"].mean()),
            "predicted_mean": float(sub["predicted"].mean()),
            "bias": m["bias"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "r2": m["r2"] if m["n"] >= 2 else float("nan"),
        })
    if not rows:
        return None
    path = TABLES / "test_metrics_by_state.csv"
    pd.DataFrame(rows).sort_values(["state"]).to_csv(path, index=False)
    return path


def save_largest_errors(df, found, n=20):
    keep = []
    for key in ("station_id", "lat", "lon", "state", "station_type"):
        if found[key] is not None and found[key] not in keep:
            keep.append(found[key])
    renamed = df.sort_values("absolute_error", ascending=False).head(n).copy()
    renamed = renamed[keep + ["observed", "predicted", "residual", "absolute_error"]]
    rename = {
        "observed": f"observed_{POLLUTANT}",
        "predicted": f"predicted_{POLLUTANT}",
        "residual": "residual_prediction_minus_observation",
    }
    path = TABLES / "largest_test_errors.csv"
    renamed.rename(columns=rename).to_csv(path, index=False)
    return path


def print_column_report(found, df):
    print("Prediction columns found")
    print("------------------------")
    for label in ("observed", "predicted", "lat", "lon", "station_id", "state", "station_type"):
        print(f"{label:>12}: {found[label] or 'not available'}")
    print(f"\nall columns: {', '.join(df.columns)}\n")


def run_analysis(predictions=PREDICTIONS, *, download_missing_boundaries=True):
    if not predictions.exists():
        raise SystemExit(f"ERROR: prediction file not found: {predictions}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(predictions)
    found = prediction_columns(raw)
    print_column_report(found, raw)
    require_columns(found)
    df = finite_analysis_frame(raw, found)
    if df.empty:
        raise SystemExit("ERROR: no rows have both observed and predicted PM2.5 values")

    metrics = metric_values(df["predicted"].to_numpy(), df["observed"].to_numpy())
    training_mean = load_training_mean_baseline()

    figure_paths = [
        save_scatter(df, metrics),
        save_distributions(df),
        save_residual_distribution(df),
        *save_maps(df, found, download_missing_boundaries=download_missing_boundaries),
    ]
    summary_path, summary = save_summary(df, metrics, training_mean)
    table_paths = [summary_path, save_largest_errors(df, found), save_by_state(df, found)]
    table_paths = [p for p in table_paths if p is not None]

    print("Summary")
    print("-------")
    print(f"n={metrics['n']}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}")
    print(f"observed mean/std/range: {summary['observed_mean']:.2f} / {summary['observed_std']:.2f} / "
          f"{summary['observed_min']:.2f}-{summary['observed_max']:.2f} {UNITS}")
    print(f"predicted mean/std/range: {summary['predicted_mean']:.2f} / {summary['predicted_std']:.2f} / "
          f"{summary['predicted_min']:.2f}-{summary['predicted_max']:.2f} {UNITS}")
    print(f"mean residual (bias)={summary['bias']:.2f}  median residual={summary['median_residual']:.2f}  "
          f"residual std={summary['residual_std']:.2f} {UNITS}")
    print(f"overpredicted={100 * summary['fraction_overpredicted']:.1f}%  "
          f"underpredicted={100 * summary['fraction_underpredicted']:.1f}%  "
          f"exact={100 * summary['fraction_exact']:.1f}%")
    if training_mean is None or math.isnan(summary["training_mean_baseline_rmse"]):
        print("training-mean baseline: not available; final_model.pt with baseline_concentration was not read")
    else:
        print(f"training-mean baseline: RMSE={summary['training_mean_baseline_rmse']:.2f} "
              f"MAE={summary['training_mean_baseline_mae']:.2f}")
        print(f"model improvement vs training-mean baseline: "
              f"{summary['training_mean_baseline_rmse_improvement']:.2f} RMSE "
              f"({summary['training_mean_baseline_rmse_improvement_pct']:.1f}%)")
    print(f"TEST-mean reference: RMSE={summary['test_mean_reference_rmse']:.2f} "
          f"MAE={summary['test_mean_reference_mae']:.2f} (R² reference, not deployable)")
    print("\nGenerated figures")
    for path in figure_paths:
        print(f"  {path.relative_to(ROOT)}")
    print("\nGenerated tables")
    for path in table_paths:
        print(f"  {path.relative_to(ROOT)}")
    return found, summary, figure_paths, table_paths


def self_check():
    import tempfile

    df = pd.DataFrame({
        "station_code": ["a", "b", "c"],
        "land": ["X", "X", "Y"],
        "lat": [52.0, 53.0, 54.0],
        "lon": [12.0, 13.0, 14.0],
        "true_pm25": [5.0, 7.0, 9.0],
        "pred_pm25": [6.0, 6.0, 10.0],
    })
    found = prediction_columns(df)
    assert found["observed"] == "true_pm25"
    assert found["predicted"] == "pred_pm25"
    assert found["state"] == "land"
    clean = finite_analysis_frame(df, found)
    m = metric_values(clean["predicted"].to_numpy(), clean["observed"].to_numpy())
    assert m["n"] == 3
    assert round(m["mae"], 6) == 1.0
    row = summary_row(clean, m, training_mean=7.0)
    assert row["training_mean_baseline_rmse"] > 0
    assert row["test_mean_reference_rmse"] > 0
    assert math.isclose(row["fraction_overpredicted"], 2 / 3)
    assert math.isclose(row["fraction_underpredicted"], 1 / 3)

    ring_lon = np.array([9.0, 15.0, 15.0, 9.0, 9.0])
    ring_lat = np.array([51.0, 51.0, 55.0, 55.0, 51.0])
    x, y = project_lonlat(ring_lon, ring_lat)
    map_context = {"countries": [np.column_stack([x, y])], "states": [np.column_stack([x, y])]}
    global FIGURES
    old_figures = FIGURES
    with tempfile.TemporaryDirectory() as td:
        try:
            FIGURES = Path(td)
            paths = save_maps(clean, found, map_context=map_context, download_missing_boundaries=False)
            assert {p.name for p in paths} == {
                "test_observed_predicted_maps.png",
                "test_observed_map.png",
                "test_predicted_map.png",
                "test_residual_map.png",
                "test_absolute_error_map.png",
            }
            stale = FIGURES / "test_observed_map.png"
            assert stale.exists()
            assert save_maps(clean, found, map_context={"countries": [], "states": []},
                             download_missing_boundaries=False) == []
            assert not stale.exists()
        finally:
            FIGURES = old_figures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze frozen cnn_deep_wide sealed-TEST predictions.")
    parser.add_argument("--self-test", action="store_true", help="run lightweight internal checks and exit")
    parser.add_argument("--download-boundaries", action="store_true",
                        help="download/cache map boundary GeoJSON files and exit")
    parser.add_argument("--no-boundary-download", action="store_true",
                        help="use only already cached boundary files")
    args = parser.parse_args()
    if args.self_test:
        self_check()
    elif args.download_boundaries:
        download_boundary_files()
    else:
        run_analysis(download_missing_boundaries=not args.no_boundary_download)
