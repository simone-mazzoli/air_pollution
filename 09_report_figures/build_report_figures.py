#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import report_plot_style

REPORT = ROOT / "Air_pollution_report"
GENERATED = REPORT / "Figures" / "generated"
MANIFEST = REPORT / "Figures" / "FIGURE_SOURCES.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(command: list[str], *, optional: bool = False) -> tuple[bool, str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    if optional:
        return False, proc.stdout.strip()
    raise SystemExit(proc.stdout)


def copy_figure(
    rows: list[dict[str, str]],
    source: Path,
    output_name: str,
    *,
    script: str,
    inputs: str,
    status: str,
) -> None:
    out = GENERATED / output_name
    if not source.exists():
        rows.append(
            {
                "figure": output_name,
                "script": script,
                "inputs": inputs,
                "source": rel(source),
                "output": rel(out),
                "status": f"blocked: missing source figure {rel(source)}",
            }
        )
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out)
    rows.append(
        {
            "figure": output_name,
            "script": script,
            "inputs": inputs,
            "source": rel(source),
            "output": rel(out),
            "status": status,
        }
    )


def add_generated_row(
    rows: list[dict[str, str]],
    output_name: str,
    *,
    script: str,
    inputs: str,
    source: str,
    status: str,
) -> None:
    rows.append(
        {
            "figure": output_name,
            "script": script,
            "inputs": inputs,
            "source": source,
            "output": rel(GENERATED / output_name),
            "status": status,
        }
    )


def compose_existing_figures(
    rows: list[dict[str, str]],
    sources: list[Path],
    output_name: str,
    titles: list[str],
    *,
    figsize: tuple[float, float],
    script: str,
    inputs: str,
    status: str,
) -> None:
    missing = [path for path in sources if not path.exists()]
    if missing:
        add_generated_row(
            rows,
            output_name,
            script=script,
            inputs=inputs,
            source=", ".join(rel(path) for path in sources),
            status="blocked: missing source figure(s) " + ", ".join(rel(path) for path in missing),
        )
        return

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageChops

    report_plot_style.apply()
    fig, axes = plt.subplots(1, len(sources), figsize=figsize)
    if len(sources) == 1:
        axes = [axes]
    for idx, (ax, source, title) in enumerate(zip(axes, sources, titles)):
        image = Image.open(source).convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bbox = ImageChops.difference(image, white).getbbox()
        if bbox:
            left, upper, right, lower = bbox
            pad = 12
            image = image.crop(
                (
                    max(0, left - pad),
                    max(0, upper - pad),
                    min(image.width, right + pad),
                    min(image.height, lower + pad),
                )
            )
        ax.imshow(np.asarray(image))
        ax.axis("off")
        report_plot_style.panel_label(ax, f"({chr(97 + idx)})", x=0.0, y=0.98)
    fig.tight_layout(w_pad=0.5)
    report_plot_style.savefig(fig, GENERATED / output_name)
    plt.close(fig)
    add_generated_row(
        rows,
        output_name,
        script=script,
        inputs=inputs,
        source=", ".join(rel(path) for path in sources),
        status=status,
    )


def write_pipeline_figure(rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    report_plot_style.apply()
    out_png = GENERATED / "pipeline_overview_compact.png"
    out_pdf = GENERATED / "pipeline_overview_compact.pdf"
    fig, ax = plt.subplots(figsize=(9.2, 3.05))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box_fc = "#F7F8FA"
    accent_fc = "#E8F1F8"
    edge = report_plot_style.BOUNDARY_COLOR
    w = 2.15
    h = 0.54
    boxes = {
        "sensor": (0.45, 4.85, w, h, "Sensor\ndata"),
        "station": (4.15, 4.85, w, h, "Reference-station\ndata"),
        "satellite": (7.85, 4.85, w, h, "Satellite\npatches"),
        "socio": (13.30, 4.85, w, h, "Socioeconomic\ndata"),
        "verify": (1.10, 3.55, 2.40, h, "LC sensor\nvalidation"),
        "process": (6.60, 3.55, 2.40, h, "Co-location /\nprocessing"),
        "scratch": (4.95, 2.40, 2.25, h, "Scratch CNN"),
        "resnet": (8.40, 2.40, 2.25, h, "Pretrained /\nfrozen ResNet"),
        "test": (6.60, 1.28, 2.40, h, "Held-out\nevaluation"),
        "map": (6.60, 0.25, 2.40, h, "Continuous\nPM map"),
        "social": (10.95, 0.25, 2.40, h, "Socioeconomic\nanalysis"),
    }

    bounds = {}
    for key, (x, y, w, h, label) in boxes.items():
        face = accent_fc if key in {"scratch", "resnet", "map", "social"} else box_fc
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.4,
            color=report_plot_style.TEXT_COLOR,
        )
        bounds[key] = (x, y, w, h)

    def anchor(key: str, side: str, frac: float = 0.5) -> tuple[float, float]:
        x, y, w, h = bounds[key]
        if side == "top":
            return x + w * frac, y + h
        if side == "bottom":
            return x + w * frac, y
        if side == "left":
            return x, y + h * frac
        if side == "right":
            return x + w, y + h * frac
        raise ValueError(side)

    def line(points: list[tuple[float, float]]) -> None:
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=edge, linewidth=0.8, solid_capstyle="round", zorder=0)

    def route(points: list[tuple[float, float]]) -> None:
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=edge,
                linewidth=0.8,
                solid_capstyle="round",
                zorder=0,
            )
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=edge,
                shrinkA=0,
                shrinkB=5,
                zorder=0,
            )
        )

    fork = (7.80, 3.02)
    merge = (7.80, 2.02)
    line([anchor("process", "bottom"), fork])
    line([anchor("scratch", "bottom"), (6.08, 2.02), merge])
    line([anchor("resnet", "bottom"), (9.52, 2.02), merge])

    start = anchor("sensor", "bottom", 0.58)
    end = anchor("verify", "top", 0.28)
    route([start, (start[0], 4.45), (end[0], 4.45), end])
    start = anchor("station", "bottom", 0.22)
    end = anchor("verify", "top", 0.72)
    route([start, (start[0], 4.62), (end[0], 4.62), end])
    start = anchor("station", "bottom", 0.78)
    end = anchor("process", "top", 0.25)
    route([start, (start[0], 4.38), (end[0], 4.38), end])
    start = anchor("satellite", "bottom", 0.42)
    end = anchor("process", "top", 0.75)
    route([start, (start[0], 4.38), (end[0], 4.38), end])
    route([anchor("verify", "right"), anchor("process", "left")])
    route([fork, (6.08, 3.02), anchor("scratch", "top", 0.50)])
    route([fork, (9.52, 3.02), anchor("resnet", "top", 0.50)])
    route([merge, anchor("test", "top")])
    route([anchor("test", "bottom"), anchor("map", "top")])
    route([anchor("map", "right"), anchor("social", "left")])
    route([anchor("socio", "bottom"), (14.38, 0.52), anchor("social", "right")])

    fig.tight_layout(pad=0.2)
    report_plot_style.savefig(fig, out_png)
    report_plot_style.savefig(fig, out_pdf)
    plt.close(fig)
    add_generated_row(
        rows,
        "pipeline_overview_compact.pdf",
        script="09_report_figures/build_report_figures.py",
        inputs="project pipeline stages documented in `PIPELINE_OVERVIEW.md`",
        source="PIPELINE_OVERVIEW.md",
        status="redesigned with fixed node coordinates and orthogonal routes; PNG preview also exported",
    )


def write_data_size_summary(rows: list[dict[str, str]]) -> None:
    source = ROOT / "06_models/data_size_ablation/results/data_size_summary_by_fraction.csv"
    out = GENERATED / "data_size_ablation_summary.png"
    if not source.exists():
        add_generated_row(
            rows,
            out.name,
            script="09_report_figures/build_report_figures.py",
            inputs=rel(source),
            source=rel(source),
            status="blocked: missing data-size summary CSV",
        )
        return

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    report_plot_style.apply()
    summary = pd.read_csv(source)
    fig, axes = plt.subplots(1, 2, figsize=report_plot_style.FIGSIZE_TWO_PANEL, sharex=True)
    for idx, (ax, (metric, label)) in enumerate(zip(axes, [("rmse", "RMSE"), ("mae", "MAE")])):
        for model, sub in summary.groupby("model"):
            sub = sub.sort_values("fraction")
            x = sub["fraction"] * 100
            y = sub[f"pooled_oof_{metric}_mean"]
            yerr = sub[f"pooled_oof_{metric}_std"]
            sns.lineplot(
                x=x,
                y=y,
                marker="o",
                linewidth=1.8,
                color=report_plot_style.model_color(model),
                label=report_plot_style.model_label(model),
                ax=ax,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=report_plot_style.model_color(model),
                elinewidth=1.0,
                capsize=3,
                alpha=0.9,
            )
        ax.set_xlabel("Training fraction [%]")
        ax.set_ylabel(f"Pooled OOF {label} [µg/m³]")
        report_plot_style.clean_axis(ax)
        report_plot_style.panel_label(ax, f"({chr(97 + idx)})")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0)
    report_plot_style.savefig(fig, out)
    plt.close(fig)
    add_generated_row(
        rows,
        out.name,
        script="09_report_figures/build_report_figures.py",
        inputs="`06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`",
        source=rel(source),
        status="regenerated from saved ablation summary; same RMSE/MAE values as the separate source figures",
    )


def station_gdf(df, *, crs: str):
    import geopandas as gpd

    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(crs)


def write_coverage_validation_geography(rows: list[dict[str, str]]) -> None:
    dev_path = ROOT / "06_models/results/cnn_deep_wide/eea_cv_predictions.csv"
    test_path = ROOT / "06_models/results/cnn_deep_wide/test_predictions.csv"
    countries_path = ROOT / "07_prediction_analysis/boundaries/ne_50m_admin_0_countries.geojson"
    out = GENERATED / "coverage_validation_geography.png"
    out_pdf = GENERATED / "coverage_validation_geography.pdf"
    missing = [p for p in (dev_path, test_path, countries_path) if not p.exists()]
    if missing:
        add_generated_row(
            rows,
            out.name,
            script="09_report_figures/build_report_figures.py",
            inputs=f"`{rel(dev_path)}`, `{rel(test_path)}`, `{rel(countries_path)}`",
            source=", ".join(rel(p) for p in (dev_path, test_path, countries_path)),
            status="blocked: missing " + ", ".join(rel(p) for p in missing),
        )
        return

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    report_plot_style.apply()
    crs = "EPSG:3035"
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    dev_geo = station_gdf(dev, crs=crs)
    test_geo = station_gdf(test, crs=crs)
    from shapely.geometry import box

    countries = gpd.read_file(countries_path)
    countries["station_iso2"] = countries["ISO_A2"].where(
        countries["ISO_A2"] != "-99",
        countries["ISO_A2_EH"],
    )
    domain = gpd.GeoSeries([box(-25.0, 30.0, 45.0, 72.5)], crs="EPSG:4326")
    countries = countries.explode(index_parts=False).reset_index(drop=True)
    countries = countries[countries.intersects(domain.iloc[0])].copy()
    station_countries = sorted(set(dev["country"].dropna()) | set(test["country"].dropna()))
    station_country_polys = countries[countries["station_iso2"].isin(station_countries)].copy()
    missing_countries = sorted(set(station_countries) - set(station_country_polys["station_iso2"]))
    if missing_countries:
        raise SystemExit("Missing country polygons for station countries: " + ", ".join(missing_countries))
    station_country_polys = station_country_polys.to_crs(crs)
    countries = countries.to_crs(crs)
    xmin, ymin, xmax, ymax = station_country_polys.total_bounds
    xpad = (xmax - xmin) * 0.035
    ypad = (ymax - ymin) * 0.035
    extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    fold_labels = {
        "fold1_iberia": "Iberia",
        "fold2_france": "France & Benelux",
        "fold3_italy": "Italy",
        "fold4_alpine": "Alpine",
        "fold5_north": "North & Baltics",
        "fold6_balkan_e": "Balkans (east)",
        "fold7_balkan_s": "Balkans (south)",
        "fold8_poland": "Poland & al.",
    }
    fold_colors = {
        "fold1_iberia": "#CC6677",
        "fold2_france": "#4477AA",
        "fold3_italy": "#228833",
        "fold4_alpine": "#AA3377",
        "fold5_north": "#66CCEE",
        "fold6_balkan_e": "#EE7733",
        "fold7_balkan_s": "#EE3377",
        "fold8_poland": "#BBBB44",
    }
    country_fold_counts = dev.groupby("country")["fold"].nunique()
    mixed_countries = sorted(country_fold_counts[country_fold_counts > 1].index)
    country_fold = (
        dev.groupby("country")["fold"]
        .first()
        .loc[lambda s: s.index.isin(country_fold_counts[country_fold_counts == 1].index)]
        .to_dict()
    )
    country_fold.pop("DE", None)

    states = gpd.read_file(ROOT / "07_prediction_analysis/boundaries/geoboundaries_deu_adm1.geojson").to_crs(crs)
    land_name = {
        "Baden-Wuerttemberg": "Baden-Württemberg",
        "Thueringen": "Thüringen",
    }
    dev_de = dev[dev["country"] == "DE"].copy()
    dev_de["state_name"] = dev_de["land"].replace(land_name)
    state_fold_counts = dev_de.groupby("state_name")["fold"].nunique()
    state_fold = (
        dev_de.groupby("state_name")["fold"]
        .first()
        .loc[lambda s: s.index.isin(state_fold_counts[state_fold_counts == 1].index)]
        .to_dict()
    )
    test_states = set(test["land"].replace(land_name).dropna())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharex=True, sharey=True)
    for idx, ax in enumerate(axes):
        report_plot_style.plot_country_context(ax, countries)
        report_plot_style.plot_outer_outline(ax, station_country_polys, zorder=2)
        report_plot_style.style_map_axis(ax, extent)
        report_plot_style.panel_label(ax, f"({chr(97 + idx)})", x=0.0, y=0.99)

    dev_geo.plot(
        ax=axes[0],
        markersize=3.2,
        color=report_plot_style.REFERENCE_COLOR,
        alpha=0.58,
        linewidth=0,
        zorder=2,
    )
    test_geo.plot(
        ax=axes[0],
        markersize=12,
        color="#4B5563",
        marker="^",
        alpha=0.9,
        linewidth=0,
        zorder=3,
    )
    station_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=report_plot_style.REFERENCE_COLOR, markeredgecolor="none", markersize=4.2, label=f"Development EEA ({len(dev):,})"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#4B5563", markeredgecolor="none", markersize=5.8, label=f"Sealed TEST ({len(test):,})"),
    ]

    for fold, color in fold_colors.items():
        report_plot_style.plot_data_polygons(
            ax=axes[1],
            polygons=countries[countries["station_iso2"].map(country_fold).eq(fold)],
            facecolor=color,
            alpha=0.86,
            zorder=2,
        )
    report_plot_style.plot_germany_state_context(axes[1], states, zorder=3)
    for fold, color in fold_colors.items():
        sub_states = states[states["shapeName"].map(state_fold).eq(fold)]
        if sub_states.empty:
            continue
        report_plot_style.plot_data_polygons(
            ax=axes[1],
            polygons=sub_states,
            facecolor=color,
            alpha=0.86,
            zorder=4,
        )
    report_plot_style.plot_data_polygons(
        ax=axes[1],
        polygons=states[states["shapeName"].isin(test_states)],
        facecolor="#4B5563",
        alpha=0.92,
        zorder=5,
    )
    report_plot_style.plot_outer_outline(axes[1], station_country_polys, zorder=6)
    report_plot_style.plot_outer_outline(axes[1], states, linewidth=0.5, zorder=7)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=fold_colors[fold], markeredgecolor="none", markersize=5, label=fold_labels[fold])
        for fold in fold_labels
        if fold in set(country_fold.values()) | set(state_fold.values())
    ]
    handles.append(
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#4B5563", markeredgecolor="none", markersize=5, label="TEST")
    )
    legend_a = axes[0].legend(
        handles=station_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=axes[0].transAxes,
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        borderpad=0.25,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=report_plot_style.REPORT_LEGEND_FONT,
    )
    legend_b = axes[1].legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        bbox_transform=axes[1].transAxes,
        ncol=1,
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        borderpad=0.25,
        handletextpad=0.4,
        labelspacing=0.12,
        fontsize=5.2,
    )
    axes[0].add_artist(legend_a)
    axes[1].add_artist(legend_b)

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.985, wspace=0.04)
    report_plot_style.savefig(fig, out)
    report_plot_style.savefig(fig, out_pdf)
    plt.close(fig)
    add_generated_row(
        rows,
        out.name,
        script="09_report_figures/build_report_figures.py::write_coverage_validation_geography",
        inputs=f"`{rel(dev_path)}`, `{rel(test_path)}`, `{rel(countries_path)}`",
        source=f"{rel(dev_path)} ({len(dev)} rows), {rel(test_path)} ({len(test)} rows), {rel(countries_path)}",
        status=(
            f"regenerated from numerical station data; CRS {crs}; "
            f"development stations={len(dev)}, TEST stations={len(test)}; "
            f"country extent from station-country polygon union; mixed countries={mixed_countries or 'none'}"
        ),
    )


def write_dense_prediction_residual_maps(rows: list[dict[str, str]]) -> None:
    grid_path = ROOT / "07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv"
    test_path = ROOT / "06_models/results/cnn_deep_wide/test_predictions.csv"
    states_path = ROOT / "07_prediction_analysis/boundaries/geoboundaries_deu_adm1.geojson"
    out = GENERATED / "dense_prediction_residual_maps.png"
    missing = [p for p in (grid_path, test_path, states_path) if not p.exists()]
    if missing:
        add_generated_row(
            rows,
            out.name,
            script="09_report_figures/build_report_figures.py",
            inputs=f"`{rel(grid_path)}`, `{rel(test_path)}`, `{rel(states_path)}`",
            source=", ".join(rel(p) for p in (grid_path, test_path, states_path)),
            status="blocked: missing " + ", ".join(rel(p) for p in missing),
        )
        return

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    import pandas as pd
    from matplotlib.colors import TwoSlopeNorm

    report_plot_style.apply()
    crs = "EPSG:25832"
    grid = pd.read_csv(grid_path)
    test = pd.read_csv(test_path)
    grid_geo = station_gdf(grid, crs=crs)
    test_geo = station_gdf(test, crs=crs)
    states = gpd.read_file(states_path).to_crs(crs)
    test_geo["residual"] = test_geo["pred_pm25"] - test_geo["true_pm25"]

    bounds = np.array(
        [
            grid_geo.total_bounds,
            test_geo.total_bounds,
        ]
    )
    xmin, ymin = bounds[:, [0, 1]].min(axis=0)
    xmax, ymax = bounds[:, [2, 3]].max(axis=0)
    xpad = max((xmax - xmin) * 0.055, 20_000)
    ypad = max((ymax - ymin) * 0.055, 20_000)
    extent = (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)

    x = grid_geo.geometry.x.to_numpy()
    y = grid_geo.geometry.y.to_numpy()
    tri = mtri.Triangulation(x, y)
    edges = np.stack(
        [
            np.hypot(x[tri.triangles[:, i]] - x[tri.triangles[:, j]], y[tri.triangles[:, i]] - y[tri.triangles[:, j]])
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=1,
    )
    tri.set_mask(edges.max(axis=1) > np.percentile(edges.max(axis=1), 95))

    fig = plt.figure(figsize=(7.2, 3.65))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.035, 0.16, 1, 0.035], wspace=0.05)
    ax_pm = fig.add_subplot(gs[0, 0])
    cax_pm = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[0, 3])
    cax_res = fig.add_subplot(gs[0, 4])

    for idx, ax in enumerate((ax_pm, ax_res)):
        report_plot_style.plot_germany_state_context(
            ax,
            states,
        )
        report_plot_style.style_map_axis(ax, extent)
        report_plot_style.panel_label(ax, f"({chr(97 + idx)})", x=0.0, y=0.99)

    pm = ax_pm.tricontourf(
        tri,
        grid["pred_pm25"].to_numpy(),
        levels=18,
        cmap=report_plot_style.PM_CMAP,
        zorder=2,
    )
    states.boundary.plot(ax=ax_pm, color=report_plot_style.MAP_INTERNAL_EDGE, linewidth=0.32, zorder=3)
    report_plot_style.plot_outer_outline(ax_pm, states, linewidth=0.55, zorder=4)
    cbar_pm = fig.colorbar(pm, cax=cax_pm)
    cbar_pm.set_label("PM$_{2.5}$ [µg/m³]", fontsize=7.5)
    cbar_pm.ax.tick_params(labelsize=6.5)

    residual_abs = max(float(np.nanmax(np.abs(test_geo["residual"]))), 1e-6)
    res = ax_res.scatter(
        test_geo.geometry.x,
        test_geo.geometry.y,
        c=test_geo["residual"],
        cmap=report_plot_style.RESIDUAL_CMAP,
        norm=TwoSlopeNorm(vmin=-residual_abs, vcenter=0, vmax=residual_abs),
        s=18,
        edgecolor="#4B5563",
        linewidth=0.35,
        zorder=3,
    )
    states.boundary.plot(ax=ax_res, color=report_plot_style.MAP_INTERNAL_EDGE, linewidth=0.32, zorder=2)
    report_plot_style.plot_outer_outline(ax_res, states, linewidth=0.55, zorder=4)
    cbar_res = fig.colorbar(res, cax=cax_res)
    cbar_res.set_label("Prediction − observation [µg/m³]", fontsize=7.5)
    cbar_res.ax.tick_params(labelsize=6.5)

    report_plot_style.savefig(fig, out)
    plt.close(fig)
    add_generated_row(
        rows,
        out.name,
        script="09_report_figures/build_report_figures.py::write_dense_prediction_residual_maps",
        inputs=f"`{rel(grid_path)}`, `{rel(test_path)}`, `{rel(states_path)}`",
        source=f"{rel(grid_path)} ({len(grid)} rows), {rel(test_path)} ({len(test)} rows), {rel(states_path)}",
        status=f"regenerated from numerical prediction data; CRS {crs}; identical extent/projection for both panels",
    )


def write_manifest(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Figure Sources",
        "",
        "Generated report-facing figures live in `Air_pollution_report/Figures/generated/`.",
        "Static/manual assets remain directly under `Air_pollution_report/Figures/`.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['figure']}",
                "",
                f"- script: `{row['script']}`",
                f"- inputs: {row['inputs']}",
                f"- source: `{row['source']}`",
                f"- report output: `{row['output']}`",
                f"- status: {row['status']}",
                "",
            ]
        )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    GENERATED.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "06_models/plot_learning_curves.py", "--experiment", "cnn_deep_wide"])
    run([sys.executable, "06_models/plot_learning_curves.py", "--experiment", "resnet_frozen"])
    run([sys.executable, "06_models/data_size_ablation/plot_results.py", "--data-size-only"])
    run([sys.executable, "07_prediction_analysis/01_analyze_test_predictions.py", "--no-boundary-download"])
    run([sys.executable, "08_kreislevel_data/visualize.py"])
    maps_ok, maps_output = run(
        [sys.executable, "08_kreislevel_data/03_map_pollution_inequality.py"],
        optional=True,
    )

    copy_figure(
        rows,
        ROOT / "06_models/results/cnn_deep_wide/figures/learning_curves_summary_objective_loss.png",
        "learning_curves_cnn_deep_wide_objective_loss.png",
        script="06_models/plot_learning_curves.py --experiment cnn_deep_wide",
        inputs="`06_models/results/cnn_deep_wide/cv_history.csv`, `06_models/results/cnn_deep_wide/cv_folds.csv`",
        status="regenerated from saved CV history; objective-loss values come from the same CSV inputs",
    )
    copy_figure(
        rows,
        ROOT / "06_models/results/resnet_frozen/figures/learning_curves_summary_objective_loss.png",
        "learning_curves_resnet_frozen_objective_loss.png",
        script="06_models/plot_learning_curves.py --experiment resnet_frozen",
        inputs="`06_models/results/resnet_frozen/cv_history.csv`, `06_models/results/resnet_frozen/cv_folds.csv`",
        status="regenerated from saved CV history; objective-loss values come from the same CSV inputs",
    )
    copy_figure(
        rows,
        ROOT / "06_models/data_size_ablation/results/figures/data_size_rmse_learning_curve.png",
        "data_size_rmse_learning_curve.png",
        script="06_models/data_size_ablation/plot_results.py --data-size-only",
        inputs="`06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`",
        status="regenerated from saved ablation summary; plotted values unchanged",
    )
    copy_figure(
        rows,
        ROOT / "06_models/data_size_ablation/results/figures/data_size_mae_learning_curve.png",
        "data_size_mae_learning_curve.png",
        script="06_models/data_size_ablation/plot_results.py --data-size-only",
        inputs="`06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`",
        status="regenerated from saved ablation summary; plotted values unchanged",
    )
    copy_figure(
        rows,
        ROOT / "07_prediction_analysis/grid_results/cnn_deep_wide_grid_map.png",
        "cnn_deep_wide_grid_map.png",
        script="07_prediction_analysis/02_predict_grid.py",
        inputs="`07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv` plus previously generated map output; grid patch arrays/checkpoint are not present here",
        status="copied existing grid map; prediction CSV is present but grid patch arrays needed to regenerate are absent",
    )
    copy_figure(
        rows,
        ROOT / "07_prediction_analysis/outputs/figures/test_residual_map.png",
        "test_residual_map.png",
        script="07_prediction_analysis/01_analyze_test_predictions.py --no-boundary-download",
        inputs="`06_models/results/cnn_deep_wide/test_predictions.csv`, cached boundary GeoJSON files",
        status="regenerated from saved TEST predictions; plotted values unchanged",
    )
    copy_figure(
        rows,
        ROOT / "08_kreislevel_data/figures/socioeconomic/01_correlation_matrix.png",
        "socioeconomic_correlation_matrix.png",
        script="08_kreislevel_data/visualize.py",
        inputs="`08_kreislevel_data/socioeconomic_kreis_2024.csv`",
        status="regenerated from saved socioeconomic table; correlations unchanged",
    )
    copy_figure(
        rows,
        ROOT / "08_kreislevel_data/figures/socioeconomic/02_indicator_boxplots.png",
        "socioeconomic_indicator_boxplots.png",
        script="08_kreislevel_data/visualize.py",
        inputs="`08_kreislevel_data/socioeconomic_kreis_2024.csv`",
        status="regenerated from saved socioeconomic table; plotted values unchanged",
    )
    if maps_ok:
        copy_figure(
            rows,
            ROOT / "08_kreislevel_data/figures/pollution_inequality_maps/00_summary_maps.png",
            "socioeconomic_summary_maps.png",
            script="08_kreislevel_data/03_map_pollution_inequality.py",
            inputs="`08_kreislevel_data/kreis_exposure_socioeconomic.csv`, Kreis boundary geometry",
            status="regenerated from numerical Kreis-level data and Kreis boundary geometry",
        )
    else:
        add_generated_row(
            rows,
            "socioeconomic_summary_maps.png",
            script="08_kreislevel_data/03_map_pollution_inequality.py",
            inputs="`08_kreislevel_data/kreis_exposure_socioeconomic.csv`, Kreis boundary geometry",
            source="08_kreislevel_data/kreis_exposure_socioeconomic.csv; missing local Kreis boundary geometry",
            status="not regenerated: " + maps_output.splitlines()[-1],
        )
    write_pipeline_figure(rows)
    write_data_size_summary(rows)
    write_coverage_validation_geography(rows)
    write_dense_prediction_residual_maps(rows)

    add_generated_row(
        rows,
        "urban_rural_pm25_distribution.png",
        script="09_report_figures/build_preliminary_analysis_figures.py",
        inputs="saved development CV predictions, daily EEA PM2.5 counts and station-area metadata",
        source="analysis_outputs/preliminary_analysis/urban_rural_pm25_summary_collapsed.csv",
        status="generated by the preliminary-analysis figure script from development stations",
    )
    add_generated_row(
        rows,
        "high_low_pm25_patch_examples.png",
        script="09_report_figures/build_preliminary_analysis_figures.py",
        inputs="local processed EEA labels and Sentinel-2 high-resolution arrays",
        source="analysis_outputs/preliminary_analysis/high_low_pm25_selected_stations.csv",
        status="generated from the three lowest and three highest eligible German development stations",
    )
    add_generated_row(
        rows,
        "high_low_pm25_patch_examples_lowres.png",
        script="09_report_figures/build_preliminary_analysis_figures.py",
        inputs="same selected stations and low-resolution Sentinel-2 arrays",
        source="analysis_outputs/preliminary_analysis/high_low_pm25_selected_stations.csv",
        status="generated as wider-context companion figure",
    )

    write_manifest(rows)
    print(f"wrote {rel(MANIFEST)}")
    print(f"wrote report figures to {rel(GENERATED)}")


if __name__ == "__main__":
    main()
