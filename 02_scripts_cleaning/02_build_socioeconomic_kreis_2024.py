"""
Combines the manually collected 2024 socioeconomic source files in
data/processed/socioeconomic/ into a single Kreis-level table.

Sources (all 2024, Kreis level, joined on AGS):
  population_age_groups.xlsx  Eurostat demo_r_pjanaggr3, 4 sheets (total/<15/
                               15-64/65+), keyed on NUTS3.
  median_age.xlsx             Eurostat demo_r_pjanind3, sheet "Sheet 3" (total,
                               not the male/female breakdown), keyed on NUTS3.
  deaths.xlsx                 Eurostat demo_r_deaths, sheet "Sheet 1", keyed on
                               NUTS3. Also used to derive a death rate.
  Bundesagentur fuer Arbeit    Official unemployment rate (Kreise, Jahreszahlen),
  Arbeitslosenquote            downloaded fresh here rather than read from
                               data/processed/socioeconomic/unemployment.xlsx,
                               which only has raw headcounts (would need
                               dividing by total population, not the standard
                               civilian-labour-force denominator the official
                               rate uses).
  income_kreislevel.xlsx      VGRdL "Bruttoloehne und -gehaelter je Arbeitnehmer"
                               (sheet "5"), keyed directly on AGS via
                               Regional-schluessel. NOTE: despite the filename,
                               this is gross wages per employee, not disposable
                               household income -- kept as a wage proxy and
                               named accordingly.

The three Eurostat files are NUTS3-keyed; data/processed/admin_boundaries/
vg250_kreis.geojson already carries AGS next to NUTS side by side (see
01_scripts_gathering/sensors-related/04_get_admin_boundaries.py), so that file
is the crosswalk -- no separate NUTS-AGS table needed. One Kreis (Wartburgkreis)
has two NUTS3 codes in Eurostat's raw export straddling a 2021 boundary
revision (DEG0P superseded by DEG0R); vg250 only carries the current DEG0R, and
the old DEG0P code's 2024 value is the Eurostat missing-value marker ":" anyway,
so it drops out during parsing without special-casing.

Input:  data/processed/admin_boundaries/vg250_kreis.geojson
        data/processed/socioeconomic/population_age_groups.xlsx
        data/processed/socioeconomic/median_age.xlsx
        data/processed/socioeconomic/deaths.xlsx
        data/processed/socioeconomic/income_kreislevel.xlsx
        (Arbeitslosenquote is downloaded, cached under data/raw/ba_arbeitslosenquote/)
Output: data/processed/socioeconomic/socioeconomic_kreis_2024.csv
"""

import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
SOCIO_DIR = BASE_DIR / "data" / "processed" / "socioeconomic"
ADMIN_DIR = BASE_DIR / "data" / "processed" / "admin_boundaries"
RAW_DIR = BASE_DIR / "data" / "raw" / "ba_arbeitslosenquote"
OUT_PATH = SOCIO_DIR / "socioeconomic_kreis_2024.csv"

YEAR = 2024

ALQ_URL = (
    "https://statistik.arbeitsagentur.de/Statistikdaten/Detail/Aktuell/iiia4/"
    "kreise-arbeitslosenquoten/arbeitslosenquoten-k-0-xlsx.xlsx"
)
ALQ_ZIP_PATH = RAW_DIR / "arbeitslosenquoten-k-0-xlsx.xlsx"


def load_base() -> pd.DataFrame:
    """AGS + Name + NUTS for all 401 Kreise -- the join backbone."""
    with open(ADMIN_DIR / "vg250_kreis.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    rows = [f["properties"] for f in geo["features"]]
    df = pd.DataFrame(rows)[["AGS", "GEN", "NUTS"]].rename(columns={"GEN": "Name"})
    return df


def parse_eurostat_sheet(path: Path, sheet: str, year: int) -> pd.DataFrame:
    """Generic Eurostat single-geography-table parser.

    Eurostat's xlsx export always has a "GEO (Codes)" / "GEO (Labels)" marker
    row, with the year header one row above it (values may carry blank spacer
    columns between years, e.g. deaths.xlsx -- found by column *value*, not by
    a fixed offset, so this works regardless). Returns columns ["NUTS", value].
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    marker_rows = raw.index[raw[0] == "GEO (Codes)"]
    assert len(marker_rows) == 1, f"expected exactly one GEO (Codes) row in {path}/{sheet}"
    geo_row = marker_rows[0]
    year_row = geo_row - 1

    year_col = None
    for col in raw.columns[2:]:
        val = raw.at[year_row, col]
        if pd.notna(val) and int(float(val)) == year:
            year_col = col
            break
    assert year_col is not None, f"year {year} not found in {path}/{sheet}"

    data = raw.iloc[geo_row + 1 :][[0, year_col]].copy()
    data.columns = ["NUTS", "value"]
    data = data[data["NUTS"].astype(str).str.startswith("DE")]
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    return data.dropna(subset=["value"])


def load_population() -> pd.DataFrame:
    sheets = {
        "Sheet 1": "bevoelkerung_gesamt",
        "Sheet 2": "bevoelkerung_unter15",
        "Sheet 3": "bevoelkerung_15_64",
        "Sheet 4": "bevoelkerung_65plus",
    }
    path = SOCIO_DIR / "population_age_groups.xlsx"
    out = None
    for sheet, col in sheets.items():
        part = parse_eurostat_sheet(path, sheet, YEAR).rename(columns={"value": col})
        out = part if out is None else out.merge(part, on="NUTS", how="outer")
    return out


def load_median_age() -> pd.DataFrame:
    path = SOCIO_DIR / "median_age.xlsx"
    df = parse_eurostat_sheet(path, "Sheet 3", YEAR)
    return df.rename(columns={"value": "medianalter"})


def load_deaths() -> pd.DataFrame:
    path = SOCIO_DIR / "deaths.xlsx"
    df = parse_eurostat_sheet(path, "Sheet 1", YEAR)
    return df.rename(columns={"value": "tode_2024"})


def load_arbeitslosenquote() -> pd.DataFrame:
    if not ALQ_ZIP_PATH.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official Arbeitslosenquote from BA ({ALQ_URL})...")
        resp = requests.get(ALQ_URL, timeout=120)
        resp.raise_for_status()
        ALQ_ZIP_PATH.write_bytes(resp.content)
    else:
        print(f"Using cached download: {ALQ_ZIP_PATH}")

    df = pd.read_excel(ALQ_ZIP_PATH, sheet_name="Jahreszahlen", header=10, skiprows=[11])
    df.columns = ["Region"] + list(df.columns[1:])
    df = df.dropna(subset=["Region"])
    df["AGS"] = df["Region"].str[:5]
    df = df[df["AGS"].str.len() == 5]
    return df[["AGS", YEAR]].rename(columns={YEAR: "arbeitslosenquote_pct"})


def load_wages() -> pd.DataFrame:
    path = SOCIO_DIR / "income_kreislevel.xlsx"
    # dtype=str is required: Regional-schluessel is otherwise inferred as
    # float64, which strips leading zeros (e.g. "08111" -> 8111.0 -> "8111.0",
    # a 6-character string) and silently breaks the length==5 Kreis filter.
    raw = pd.read_excel(path, sheet_name="5", header=4, dtype=str)
    key_col = [c for c in raw.columns if "schl" in str(c).lower()][0]
    raw[key_col] = raw[key_col].astype(str).str.strip()
    kreis = raw[raw[key_col].str.len() == 5].copy()
    year_col = [c for c in raw.columns if str(c).strip() == str(YEAR)][-1]
    out = kreis[[key_col, year_col]].rename(
        columns={key_col: "AGS", year_col: "bruttolohn_je_arbeitnehmer_eur"}
    )
    out["bruttolohn_je_arbeitnehmer_eur"] = pd.to_numeric(
        out["bruttolohn_je_arbeitnehmer_eur"], errors="coerce"
    )
    return out


def report_coverage(df: pd.DataFrame, total: int) -> None:
    print(f"\nCoverage ({total} Kreise total):")
    for col in df.columns:
        if col in ("AGS", "Name", "NUTS"):
            continue
        n = df[col].notna().sum()
        print(f"  {col}: {n}/{total}")


def main() -> None:
    base = load_base()
    total = len(base)
    print(f"Base: {total} Kreise from vg250_kreis.geojson")

    merged = base
    for label, loader, join_key in [
        ("population (4 age brackets)", load_population, "NUTS"),
        ("median age", load_median_age, "NUTS"),
        ("deaths", load_deaths, "NUTS"),
        ("Arbeitslosenquote", load_arbeitslosenquote, "AGS"),
        ("wages (income proxy)", load_wages, "AGS"),
    ]:
        print(f"Loading {label}...")
        part = loader()
        merged = merged.merge(part, on=join_key, how="left")

    merged["todesrate_je_1000ew"] = (
        merged["tode_2024"] / merged["bevoelkerung_gesamt"] * 1000
    )

    report_coverage(merged, total)

    merged = merged.drop(columns=["NUTS"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(
        "\nNote: bruttolohn_je_arbeitnehmer_eur is gross wages per employee "
        "(VGRdL), not disposable household income -- treat as a wage proxy."
    )


if __name__ == "__main__":
    main()
