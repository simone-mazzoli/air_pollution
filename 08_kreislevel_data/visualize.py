from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import report_plot_style

report_plot_style.apply()
savefig = report_plot_style.savefig

DEFAULT_INPUT = ROOT / "08_kreislevel_data" / "socioeconomic_kreis_2024.csv"
DEFAULT_OUTDIR = ROOT / "08_kreislevel_data" / "figures" / "socioeconomic"


CORR_VARIABLES = [
    "disposable_income_2023_eur_per_capita",
    "unemployment_rate_2024_pct",
    "no_vocational_qualification_2022_pct",
    "university_degree_2022_pct",
    "immigration_history_2022_pct",
    "population_density_2024_per_km2",
]

BOX_VARIABLES = [
    "disposable_income_2023_eur_per_capita",
    "unemployment_rate_2024_pct",
    "no_vocational_qualification_2022_pct",
    "university_degree_2022_pct",
    "immigration_history_2022_pct",
    "population_density_2024_per_km2",
    "share_65plus_2024_pct",
]

LABELS = {
    "disposable_income_2023_eur_per_capita": "Disposable income",
    "unemployment_rate_2024_pct": "Unemployment",
    "no_vocational_qualification_2022_pct": "No vocational qualification",
    "university_degree_2022_pct": "University degree",
    "immigration_history_2022_pct": "Immigration history",
    "population_density_2024_per_km2": "Population density",
    "share_65plus_2024_pct": "Age 65+",
}

SHORT_LABELS = {
    "disposable_income_2023_eur_per_capita": "Income",
    "unemployment_rate_2024_pct": "Unemployment",
    "no_vocational_qualification_2022_pct": "No vocational\nqualification",
    "university_degree_2022_pct": "University\ndegree",
    "immigration_history_2022_pct": "Immigration\nhistory",
    "population_density_2024_per_km2": "Log population\ndensity",
}

UNITS = {
    "disposable_income_2023_eur_per_capita": "EUR per capita (2023)",
    "unemployment_rate_2024_pct": "% (2024)",
    "no_vocational_qualification_2022_pct": "% (2022)",
    "university_degree_2022_pct": "% (2022)",
    "immigration_history_2022_pct": "% (2022)",
    "population_density_2024_per_km2": "persons per km² (2024)",
    "share_65plus_2024_pct": "% (2024)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, dtype={"AGS": str})
    for col in set(CORR_VARIABLES + BOX_VARIABLES):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def existing_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    cols = [c for c in requested if c in df.columns]
    if len(cols) < 2:
        raise ValueError("Too few expected socioeconomic variables found.")
    return cols


def save_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    out_path: Path,
) -> None:
    data = df[columns].copy()

    density_col = "population_density_2024_per_km2"
    if density_col in data.columns:
        density = data[density_col]
        data[density_col] = np.where(
            density > 0,
            np.log10(density),
            np.nan,
        )

    corr = data.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    labels = [SHORT_LABELS[c] for c in columns]
    sns.heatmap(
        corr,
        ax=ax,
        cmap=report_plot_style.RESIDUAL_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9},
        linewidths=0.25,
        linecolor="#F3F4F6",
        square=True,
        cbar_kws={"label": "Pearson r", "fraction": 0.048, "pad": 0.04},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.grid(False)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="both", length=0)
    ax.set_title("")
    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def save_indicator_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    out_path: Path,
) -> None:
    n = len(columns)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5, 2.3 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    rng = np.random.default_rng(123)

    for ax, col in zip(axes, columns):
        values = df[col].dropna().to_numpy()

        ax.boxplot(
            values,
            vert=False,
            widths=0.45,
            showfliers=False,
            whis=(5, 95),
            patch_artist=True,
            boxprops={
                "facecolor": "0.90",
                "edgecolor": "0.35",
                "linewidth": 1.2,
            },
            whiskerprops={"color": "0.45", "linewidth": 1.0},
            capprops={"color": "0.45", "linewidth": 1.0},
            medianprops={"color": "0.10", "linewidth": 2.0},
        )

        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            values,
            np.ones(len(values)) + jitter,
            s=8,
            alpha=0.14,
            edgecolors="none",
        )

        if col == "population_density_2024_per_km2":
            ax.set_xscale("log")

        ax.set_yticks([])
        ax.set_title(LABELS[col], loc="left", fontsize=10.5)
        ax.set_xlabel(UNITS[col], fontsize=9)
        ax.grid(True, axis="x", alpha=0.20)
        ax.set_axisbelow(True)

        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    savefig(fig, out_path)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    corr_columns = existing_columns(df, CORR_VARIABLES)
    box_columns = existing_columns(df, BOX_VARIABLES)

    corr_path = args.outdir / "01_correlation_matrix.png"
    box_path = args.outdir / "02_indicator_boxplots.png"

    save_correlation_matrix(df, corr_columns, corr_path)
    save_indicator_boxplots(df, box_columns, box_path)

    print(f"Loaded {len(df)} Kreise from {args.input}")
    print("Generated:")
    print(f"  {corr_path}")
    print(f"  {box_path}")


if __name__ == "__main__":
    main()
