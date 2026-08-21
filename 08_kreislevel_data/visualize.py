from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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
    values = corr.to_numpy()
    n = len(columns)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(
        values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="none",
        aspect="equal",
    )

    labels = [SHORT_LABELS[c] for c in columns]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            r = values[i, j]
            ax.text(
                j,
                i,
                f"{r:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if abs(r) >= 0.55 else "black",
            )

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cbar.set_label("Pearson correlation")

    ax.set_title("Correlation of selected socioeconomic indicators", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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

    fig.suptitle(
        "Distribution of selected socioeconomic and demographic indicators",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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