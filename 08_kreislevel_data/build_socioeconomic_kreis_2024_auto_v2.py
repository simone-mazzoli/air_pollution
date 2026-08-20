#!/usr/bin/env python3
"""
Automatically build the Kreis-level socioeconomic table used in the
environmental-justice analysis.

This replaces the earlier mostly-manual builder.  It downloads/caches the
public official sources itself, parses them, joins them by AGS/NUTS, validates
coverage, and writes a small report-ready table.

Primary variables
-----------------
2024:
  population_2024
  population_under15_2024
  population_15_64_2024
  population_65plus_2024
  share_under15_2024_pct
  share_15_64_2024_pct
  share_65plus_2024_pct
  median_age_2024
  population_density_2024_per_km2
  unemployment_rate_2024_pct

2023:
  disposable_income_2023_eur_per_capita

Zensus 2022:
  immigration_history_2022_pct
  foreign_national_2022_pct
  no_vocational_qualification_2022_pct
  university_degree_2022_pct

Deliberately NOT included:
  - mortality/death-rate variables
  - material-deprivation variables

Why mixed reference years?
--------------------------
The column names preserve the true reference year instead of pretending that
slow-moving structural variables from Zensus 2022 or VGRdL 2023 are 2024 data.
They can still be joined to the 2024 pollution exposure surface later.

Sources
-------
Eurostat Statistics API:
  demo_r_pjanaggr3   population by broad age group, NUTS3
  demo_r_pjanind3    population structure indicators, NUTS3
  demo_r_d3dens      population density, NUTS3

Bundesagentur fuer Arbeit:
  annual official Kreis unemployment rate workbook

VGRdL / Statistikportal:
  Reihe 2 Band 3, table 2.4: disposable household income per inhabitant

Destatis / Zensus 2022:
  public regional tables for Demografie and Bildung/Erwerbstaetigkeit.
  The script discovers the current download links from the official Destatis
  Regionaltabellen page rather than requiring a manually downloaded workbook.

Local project dependency
------------------------
data/processed/admin_boundaries/vg250_kreis.geojson

This is only used as the project's AGS <-> NUTS3 crosswalk and name source.
The final base is restricted to NUTS3 codes actually present in Eurostat's
2024 population data, which helps avoid accidentally including a later Kreis
boundary vintage.

Outputs
-------
08_kreislevel_data/socioeconomic_kreis_2024.csv
08_kreislevel_data/socioeconomic_kreis_2024_coverage.csv
08_kreislevel_data/socioeconomic_kreis_2024_sources.json

Normal run
----------
python 08_kreislevel_data/build_socioeconomic_kreis_2024.py

Refresh all web sources
-----------------------
python 08_kreislevel_data/build_socioeconomic_kreis_2024.py --refresh

Quick code/parser check without internet
----------------------------------------
python 08_kreislevel_data/build_socioeconomic_kreis_2024.py --self-check
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sqlite3
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ADMIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "admin_boundaries"
    / "vg250_kreis.geojson"
)

RAW_DIR = BASE_DIR / "data" / "raw" / "socioeconomic_auto"
BKG_DIR = RAW_DIR / "bkg"
EUROSTAT_DIR = RAW_DIR / "eurostat"
BA_DIR = RAW_DIR / "ba"
VGRDL_DIR = RAW_DIR / "vgrdl"
ZENSUS_DIR = RAW_DIR / "zensus2022"

OUT_DIR = BASE_DIR / "08_kreislevel_data"
OUT_CSV = OUT_DIR / "socioeconomic_kreis_2024.csv"
OUT_COVERAGE = OUT_DIR / "socioeconomic_kreis_2024_coverage.csv"
OUT_MANIFEST = OUT_DIR / "socioeconomic_kreis_2024_sources.json"

YEAR = 2024
INCOME_YEAR = 2023
ZENSUS_YEAR = 2022
EXPECTED_2024_KREISE = 400


# ---------------------------------------------------------------------------
# Public sources
# ---------------------------------------------------------------------------

EUROSTAT_API = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
)

BA_UNEMPLOYMENT_URL = (
    "https://statistik.arbeitsagentur.de/Statistikdaten/Detail/Aktuell/iiia4/"
    "kreise-arbeitslosenquoten/arbeitslosenquoten-k-0-xlsx.xlsx"
)

# Official current Kreis-income workbook (Reihe 2 Band 3, through 2023).
VGRDL_INCOME_URL = (
    "https://www.statistikportal.de/sites/default/files/2026-01/"
    "vgrdl_r2b3_bs2024_0.xlsx"
)

VGRDL_PUBLICATIONS_PAGE = (
    "https://www.statistikportal.de/de/vgrdl/publikationen"
)

ZENSUS_REGIONAL_TABLES_PAGE = (
    "https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/"
    "Zensus2022/Publikationen/publikationen-akkordeon-regionaltabellen.html"
)

# Official BKG VG250-EW administrative areas, reference date 31.12.2024.
# We download the GeoPackage archive rather than the human-facing Excel archive:
# GeoPackage is SQLite, so AGS/GEN/NUTS can be read robustly with Python's built-in
# sqlite3 module and no geopandas/GDAL dependency.
BKG_VG250_2024_URL = (
    "https://daten.gdz.bkg.bund.de/produkte/vg/vg250-ew_ebenen_1231/2024/"
    "vg250-ew_12-31.utm32s.gpkg.ebenen.zip"
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36 "
    "air-pollution-university-project/1.0"
)


# ---------------------------------------------------------------------------
# Source manifest
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    variable: str
    source: str
    reference_year: int
    source_id_or_url: str
    cached_file: str | None
    n_available: int
    n_total: int
    note: str = ""


SOURCE_RECORDS: list[SourceRecord] = []


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def record_source(
    df: pd.DataFrame,
    variables: Iterable[str],
    *,
    source: str,
    reference_year: int,
    source_id_or_url: str,
    cached_file: Path | None = None,
    note: str = "",
) -> None:
    for variable in variables:
        SOURCE_RECORDS.append(
            SourceRecord(
                variable=variable,
                source=source,
                reference_year=reference_year,
                source_id_or_url=source_id_or_url,
                cached_file=_relative(cached_file),
                n_available=(
                    int(df[variable].notna().sum())
                    if variable in df.columns
                    else 0
                ),
                n_total=len(df),
                note=note,
            )
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def web_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        }
    )
    return sess


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = (
        text.replace("ß", "ss")
        .replace("\u00a0", " ")
        .replace("–", "-")
        .replace("—", "-")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_ags(value: object) -> str | None:
    """Return a five-digit Kreis AGS where possible."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None

    # Explicitly reject years and other short numeric fields.
    if len(digits) < 4:
        return None

    if len(digits) == 4:
        return digits.zfill(5)
    if len(digits) == 5:
        return digits

    # Regional keys can be longer than AGS; Kreis is first five digits.
    return digits[:5]


def parse_number(value: object) -> float:
    """Parse numeric values from German/English official spreadsheets."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if text in {"", ":", "-", "—", ".", "..", "...", "nan", "NaN"}:
        return float("nan")

    # Remove common footnote markers at the end.
    text = re.sub(r"[A-Za-z*]+$", "", text)

    # German decimal comma: 1.234,5 -> 1234.5
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        # Keep one decimal point as decimal separator.  Multiple dots are most
        # likely thousands separators.
        if text.count(".") > 1:
            text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return float("nan")


def numeric_series(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(
    url: str,
    path: Path,
    *,
    refresh: bool,
    timeout: int = 180,
    referer: str | None = None,
) -> Path:
    if path.exists() and path.stat().st_size > 0 and not refresh:
        print(f"Using cached: {_relative(path)}")
        return path

    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    headers = {"Referer": referer} if referer else None
    print(f"Downloading: {url}")
    with web_session().get(
        url,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
        headers=headers,
    ) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty file from {url}")

    tmp.replace(path)
    print(f"Saved: {_relative(path)} ({path.stat().st_size:,} bytes)")
    return path


def safe_merge(
    base: pd.DataFrame,
    part: pd.DataFrame,
    *,
    key: str,
    label: str,
) -> pd.DataFrame:
    part = part.copy()
    if part[key].duplicated().any():
        examples = (
            part.loc[part[key].duplicated(keep=False), key]
            .astype(str)
            .unique()[:10]
            .tolist()
        )
        raise ValueError(f"{label}: duplicate {key}, examples={examples}")

    before = len(base)
    merged = base.merge(part, on=key, how="left", validate="one_to_one")
    if len(merged) != before:
        raise AssertionError(
            f"{label}: merge changed row count {before} -> {len(merged)}"
        )
    return merged


# ---------------------------------------------------------------------------
# Geography crosswalk
# ---------------------------------------------------------------------------

def _crosswalk_from_bkg_zip(zip_path: Path) -> pd.DataFrame:
    """Extract the 2024 Kreis AGS/NUTS/name table from BKG's GeoPackage archive.

    GeoPackage is a SQLite database.  Scan non-metadata tables for a layer that
    contains a district key (AGS/RS), GEN/name, and NUTS and whose number of
    unique Kreis keys is close to the official 2024 count.  This is more robust
    than parsing the presentation-oriented Excel archive.
    """
    extract_dir = BKG_DIR / "vg250_2024_gpkg"
    extract_dir.mkdir(parents=True, exist_ok=True)

    gpkg_files = sorted(extract_dir.rglob("*.gpkg"))
    if not gpkg_files:
        with zipfile.ZipFile(zip_path) as archive:
            gpkg_members = [
                name for name in archive.namelist()
                if name.lower().endswith(".gpkg") and not name.endswith("/")
            ]
            if not gpkg_members:
                raise RuntimeError(
                    f"BKG archive {_relative(zip_path)} contained no .gpkg file; "
                    f"members sample={archive.namelist()[:30]}"
                )
            for member in gpkg_members:
                archive.extract(member, extract_dir)
        gpkg_files = sorted(extract_dir.rglob("*.gpkg"))

    candidates: list[tuple[int, Path, str, pd.DataFrame]] = []
    metadata_prefixes = ("gpkg_", "rtree_", "sqlite_")

    for gpkg in gpkg_files:
        try:
            con = sqlite3.connect(str(gpkg))
        except sqlite3.Error:
            continue
        try:
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not str(row[0]).lower().startswith(metadata_prefixes)
            ]
            for table in tables:
                try:
                    info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
                except sqlite3.Error:
                    continue
                if not info:
                    continue

                # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
                names = [str(row[1]) for row in info]
                lower = {name.lower(): name for name in names}

                key_col = next(
                    (lower[k] for k in ("ags", "ags_0", "rs", "rs_0", "ars", "ars_0") if k in lower),
                    None,
                )
                name_col = next(
                    (lower[k] for k in ("gen", "name", "gebietsname") if k in lower),
                    None,
                )
                nuts_col = next(
                    (lower[k] for k in ("nuts", "nuts3", "nuts_id") if k in lower),
                    None,
                )
                if key_col is None or name_col is None or nuts_col is None:
                    continue

                def qident(value: str) -> str:
                    return '"' + value.replace('"', '""') + '"'

                query = (
                    f"SELECT {qident(key_col)} AS district_key, "
                    f"{qident(name_col)} AS district_name, "
                    f"{qident(nuts_col)} AS nuts FROM {qident(table)}"
                )
                try:
                    raw = pd.read_sql_query(query, con)
                except Exception:
                    continue

                part = raw.rename(
                    columns={
                        "district_key": "AGS",
                        "district_name": "Name",
                        "nuts": "NUTS",
                    }
                )
                part["AGS"] = part["AGS"].map(normalize_ags)
                part["Name"] = part["Name"].astype(str).str.strip()
                part["NUTS"] = part["NUTS"].astype(str).str.strip()
                part = part[
                    part["AGS"].notna()
                    & part["NUTS"].str.match(r"^DE[A-Z0-9]{3}$", na=False)
                ].drop_duplicates("AGS")

                n = len(part)
                if not n:
                    continue
                score = 10000 - abs(n - EXPECTED_2024_KREISE)
                table_norm = normalize_text(table)
                if "krs" in table_norm or "kreis" in table_norm:
                    score += 20000
                if 380 <= n <= 420:
                    score += 10000
                if key_col.lower() in {"ags", "ags_0"}:
                    score += 1000
                candidates.append((score, gpkg, table, part))
        finally:
            con.close()

    if not candidates:
        diagnostics = []
        for gpkg in gpkg_files:
            try:
                con = sqlite3.connect(str(gpkg))
                for (table,) in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall():
                    if str(table).lower().startswith(metadata_prefixes):
                        continue
                    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
                    diagnostics.append(f"{table}: {cols}")
                con.close()
            except Exception:
                pass
        raise RuntimeError(
            "Could not find a Kreis AGS/GEN/NUTS layer inside the official BKG "
            f"GeoPackage archive {_relative(zip_path)}. Available tables/columns: "
            + " | ".join(diagnostics[:20])
        )

    _score, gpkg, table, out = max(candidates, key=lambda item: item[0])
    print(
        f"  BKG crosswalk: {len(out)} rows from "
        f"{_relative(gpkg)} / table {table!r}"
    )
    return out.sort_values("AGS").reset_index(drop=True)

def load_geo_crosswalk(*, refresh: bool = False) -> pd.DataFrame:
    """Load the 2024 Kreis AGS/NUTS/name crosswalk.

    Prefer the project's existing GeoJSON when available.  If it is absent,
    automatically download the official BKG 31.12.2024 administrative-area
    GeoPackage archive and extract the Kreis table with built-in sqlite3.  This keeps the socioeconomic
    builder runnable on a clean server checkout without the large processed
    boundary asset.
    """
    if ADMIN_PATH.exists():
        obj = json.loads(ADMIN_PATH.read_text(encoding="utf-8"))
        rows = [feature.get("properties", {}) for feature in obj.get("features", [])]
        frame = pd.DataFrame(rows)

        required = {"AGS", "GEN", "NUTS"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{_relative(ADMIN_PATH)} missing properties {sorted(missing)}"
            )

        out = frame[["AGS", "GEN", "NUTS"]].rename(columns={"GEN": "Name"}).copy()
        out["AGS"] = out["AGS"].map(normalize_ags)
        out["NUTS"] = out["NUTS"].astype(str).str.strip()
        out = out.dropna(subset=["AGS", "NUTS"])
        out = out.sort_values(["AGS", "NUTS"]).drop_duplicates("AGS", keep="last")
        print(f"  using project boundary crosswalk: {len(out)} rows")
        return out.reset_index(drop=True)

    print(
        f"  {_relative(ADMIN_PATH)} is absent; downloading the official BKG "
        "31.12.2024 Kreis tables instead."
    )
    zip_path = download_file(
        BKG_VG250_2024_URL,
        BKG_DIR / "vg250-ew_12-31.utm32s.gpkg.ebenen.zip",
        refresh=refresh,
    )
    return _crosswalk_from_bkg_zip(zip_path)


# ---------------------------------------------------------------------------
# Eurostat JSON-stat parser
# ---------------------------------------------------------------------------

def _dimension_codes(payload: dict, dim: str) -> list[str]:
    category = payload["dimension"][dim]["category"]
    index = category["index"]
    if isinstance(index, list):
        return list(index)
    if isinstance(index, dict):
        return [
            code
            for code, _pos in sorted(index.items(), key=lambda item: item[1])
        ]
    raise TypeError(f"Unsupported JSON-stat category index for {dim!r}")


def jsonstat_to_frame(payload: dict) -> pd.DataFrame:
    dims = list(payload["id"])
    sizes = list(payload["size"])

    codes = {dim: _dimension_codes(payload, dim) for dim in dims}
    labels = {
        dim: payload["dimension"][dim]["category"].get("label", {})
        for dim in dims
    }

    values = payload.get("value", {})
    if isinstance(values, list):
        indexed_values = {
            idx: value
            for idx, value in enumerate(values)
            if value is not None
        }
    elif isinstance(values, dict):
        indexed_values = {
            int(idx): value
            for idx, value in values.items()
            if value is not None
        }
    else:
        raise TypeError(f"Unsupported JSON-stat value type {type(values)}")

    rows: list[dict] = []
    for flat_idx, value in indexed_values.items():
        coords = np.unravel_index(flat_idx, sizes)
        row = {"value": value}
        for dim, coord in zip(dims, coords):
            code = codes[dim][coord]
            row[dim] = code
            dim_labels = labels[dim]
            row[f"{dim}_label"] = (
                dim_labels.get(code, code)
                if isinstance(dim_labels, dict)
                else code
            )
        rows.append(row)

    return pd.DataFrame(rows)


def eurostat_payload(
    dataset: str,
    params: dict[str, object],
    *,
    cache_name: str,
    refresh: bool,
) -> tuple[dict, Path]:
    EUROSTAT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EUROSTAT_DIR / f"{cache_name}.json"

    if cache.exists() and not refresh:
        print(f"Using cached: {_relative(cache)}")
        return json.loads(cache.read_text(encoding="utf-8")), cache

    query = {"format": "JSON", "lang": "EN", **params}
    url = f"{EUROSTAT_API}/{dataset}"
    print(f"Eurostat {dataset}: {query}")
    response = web_session().get(url, params=query, timeout=180)
    response.raise_for_status()
    payload = response.json()

    if payload.get("class") != "dataset":
        raise RuntimeError(
            f"Unexpected Eurostat response for {dataset}: "
            f"{str(payload)[:500]}"
        )

    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload, cache


def german_nuts3(frame: pd.DataFrame) -> pd.DataFrame:
    if "geo" not in frame.columns:
        raise ValueError("Eurostat result does not contain a geo dimension")
    mask = frame["geo"].astype(str).str.match(r"^DE[A-Z0-9]{3}$")
    return frame.loc[mask].copy()


def _select_age_rows(
    frame: pd.DataFrame,
    *,
    code: str,
    label_patterns: Sequence[str],
) -> pd.DataFrame:
    if "age" in frame.columns and code in set(frame["age"].astype(str)):
        return frame[frame["age"].astype(str) == code]

    if "age_label" not in frame.columns:
        return frame.iloc[0:0]

    text = frame["age_label"].map(normalize_text)
    mask = np.zeros(len(frame), dtype=bool)
    for pattern in label_patterns:
        mask |= text.str.contains(pattern, regex=True, na=False).to_numpy()
    return frame.loc[mask]


def load_population_age(*, refresh: bool) -> tuple[pd.DataFrame, Path]:
    payload, cache = eurostat_payload(
        "demo_r_pjanaggr3",
        {
            "time": YEAR,
            "sex": "T",
            "unit": "NR",
            "geoLevel": "nuts3",
        },
        cache_name=f"demo_r_pjanaggr3_{YEAR}",
        refresh=refresh,
    )
    frame = german_nuts3(jsonstat_to_frame(payload))
    if frame.empty:
        raise RuntimeError("Eurostat population query returned no German NUTS3")

    specs = [
        (
            "TOTAL",
            [r"\btotal\b"],
            "population_2024",
        ),
        (
            "Y_LT15",
            [r"less than 15", r"under 15", r"0.*14"],
            "population_under15_2024",
        ),
        (
            "Y15-64",
            [r"15.*64"],
            "population_15_64_2024",
        ),
        (
            "Y_GE65",
            [r"65.*over", r"65.*more", r"65.*older"],
            "population_65plus_2024",
        ),
    ]

    parts: list[pd.DataFrame] = []
    for age_code, patterns, column in specs:
        selected = _select_age_rows(
            frame,
            code=age_code,
            label_patterns=patterns,
        )
        if selected.empty:
            available = (
                sorted(frame["age"].astype(str).unique())
                if "age" in frame.columns
                else []
            )
            raise RuntimeError(
                f"Could not identify age group {age_code} in "
                f"demo_r_pjanaggr3; available age codes={available[:50]}"
            )
        part = (
            selected[["geo", "value"]]
            .rename(columns={"geo": "NUTS", "value": column})
            .copy()
        )
        part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.drop_duplicates("NUTS")
        parts.append(part)

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="NUTS", how="outer", validate="one_to_one")

    out["share_under15_2024_pct"] = (
        out["population_under15_2024"] / out["population_2024"] * 100.0
    )
    out["share_15_64_2024_pct"] = (
        out["population_15_64_2024"] / out["population_2024"] * 100.0
    )
    out["share_65plus_2024_pct"] = (
        out["population_65plus_2024"] / out["population_2024"] * 100.0
    )
    return out, cache


def _indicator_match(
    frame: pd.DataFrame,
    patterns: Sequence[str],
) -> pd.Series:
    masks = []
    for column in frame.columns:
        if not column.endswith("_label"):
            continue
        if column in {"geo_label", "time_label", "sex_label"}:
            continue
        text = frame[column].map(normalize_text)
        mask = np.zeros(len(frame), dtype=bool)
        for pattern in patterns:
            mask |= text.str.contains(pattern, regex=True, na=False).to_numpy()
        if mask.any():
            masks.append(pd.Series(mask, index=frame.index))
    if not masks:
        return pd.Series(False, index=frame.index)
    result = masks[0].copy()
    for mask in masks[1:]:
        result |= mask
    return result


def load_median_age(*, refresh: bool) -> tuple[pd.DataFrame, Path]:
    payload, cache = eurostat_payload(
        "demo_r_pjanind3",
        {
            "time": YEAR,
            # demo_r_pjanind3 is an indicator table; unlike the raw
            # population table demo_r_pjanaggr3 it has no sex dimension.
            # Passing sex=T makes the Eurostat dissemination API return 400.
            "geoLevel": "nuts3",
        },
        cache_name=f"demo_r_pjanind3_{YEAR}",
        refresh=refresh,
    )
    frame = german_nuts3(jsonstat_to_frame(payload))
    if frame.empty:
        raise RuntimeError("Eurostat median-age query returned no German NUTS3")

    mask = _indicator_match(frame, [r"median age"])
    if not mask.any():
        # Code fallback.  Eurostat indicator codes can change names while
        # labels remain stable, so this is deliberately secondary.
        for code_col in ("indic_de", "indic"):
            if code_col in frame.columns:
                candidate = frame[code_col].astype(str).str.contains(
                    "MED",
                    case=False,
                    na=False,
                )
                if candidate.any():
                    mask = candidate
                    break

    if not mask.any():
        sample_labels = sorted(
            {
                str(value)
                for column in frame.columns
                if column.endswith("_label")
                for value in frame[column].dropna().unique()
            }
        )
        raise RuntimeError(
            "Could not identify median-age indicator in demo_r_pjanind3. "
            f"Label sample={sample_labels[:40]}"
        )

    out = (
        frame.loc[mask, ["geo", "value"]]
        .rename(columns={"geo": "NUTS", "value": "median_age_2024"})
        .copy()
    )
    out["median_age_2024"] = pd.to_numeric(
        out["median_age_2024"],
        errors="coerce",
    )
    return out.drop_duplicates("NUTS"), cache


def load_population_density(*, refresh: bool) -> tuple[pd.DataFrame, Path]:
    payload, cache = eurostat_payload(
        "demo_r_d3dens",
        {
            "time": YEAR,
            "geoLevel": "nuts3",
        },
        cache_name=f"demo_r_d3dens_{YEAR}",
        refresh=refresh,
    )
    frame = german_nuts3(jsonstat_to_frame(payload))
    if frame.empty:
        raise RuntimeError(
            "Eurostat population-density query returned no German NUTS3"
        )

    # If the dataset contains multiple units, prefer inhabitants/km².
    if "unit_label" in frame.columns:
        unit_text = frame["unit_label"].map(normalize_text)
        unit_mask = unit_text.str.contains(
            r"(?:square kilomet|km2|km²)",
            regex=True,
            na=False,
        )
        if unit_mask.any():
            frame = frame.loc[unit_mask]

    out = (
        frame[["geo", "value"]]
        .rename(
            columns={
                "geo": "NUTS",
                "value": "population_density_2024_per_km2",
            }
        )
        .copy()
    )
    out["population_density_2024_per_km2"] = pd.to_numeric(
        out["population_density_2024_per_km2"],
        errors="coerce",
    )
    return out.drop_duplicates("NUTS"), cache


# ---------------------------------------------------------------------------
# Bundesagentur fuer Arbeit
# ---------------------------------------------------------------------------

def load_unemployment(*, refresh: bool) -> tuple[pd.DataFrame, Path]:
    path = download_file(
        BA_UNEMPLOYMENT_URL,
        BA_DIR / "arbeitslosenquoten-k-0-xlsx.xlsx",
        refresh=refresh,
    )

    frame = pd.read_excel(
        path,
        sheet_name="Jahreszahlen",
        header=10,
        skiprows=[11],
    )
    frame.columns = ["Region"] + list(frame.columns[1:])
    frame = frame.dropna(subset=["Region"]).copy()
    frame["AGS"] = frame["Region"].astype(str).str.extract(
        r"^\s*(\d{5})",
        expand=False,
    )

    year_columns = [
        column
        for column in frame.columns
        if str(column).strip() == str(YEAR)
    ]
    if not year_columns:
        raise RuntimeError(f"BA workbook has no {YEAR} annual column")
    year_column = year_columns[-1]

    out = frame[["AGS", year_column]].rename(
        columns={year_column: "unemployment_rate_2024_pct"}
    )
    out["unemployment_rate_2024_pct"] = numeric_series(
        out["unemployment_rate_2024_pct"]
    )
    out = out.dropna(subset=["AGS"]).drop_duplicates("AGS")
    return out, path


# ---------------------------------------------------------------------------
# VGRdL disposable household income
# ---------------------------------------------------------------------------

def identify_vgrdl_income_sheet(path: Path) -> str:
    workbook = pd.ExcelFile(path)
    scored: list[tuple[int, str]] = []

    for sheet in workbook.sheet_names:
        head = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            nrows=24,
            dtype=object,
        )
        text = normalize_text(
            " ".join(head.fillna("").astype(str).to_numpy().ravel())
        )
        score = 0
        if normalize_text(sheet).startswith("2.4"):
            score += 10
        if "verfugbares einkommen" in text:
            score += 4
        if "je einwohner" in text or "je einwohnerin" in text:
            score += 4
        scored.append((score, sheet))

    score, sheet = max(scored)
    if score <= 0:
        raise RuntimeError(
            f"Could not identify VGRdL table 2.4; sheets={workbook.sheet_names}"
        )
    return sheet


def identify_header_row(path: Path, sheet: str) -> int:
    raw = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        nrows=35,
        dtype=object,
    )
    for idx in range(len(raw)):
        values = [normalize_text(value) for value in raw.iloc[idx].tolist()]
        if any(
            "regional" in value
            and ("schlussel" in value or "schl" in value)
            for value in values
        ):
            return idx
    return 4


def load_disposable_income(*, refresh: bool) -> tuple[pd.DataFrame, Path]:
    path = download_file(
        VGRDL_INCOME_URL,
        VGRDL_DIR / "vgrdl_r2b3_income_kreise.xlsx",
        refresh=refresh,
        referer=VGRDL_PUBLICATIONS_PAGE,
    )
    sheet = identify_vgrdl_income_sheet(path)
    header = identify_header_row(path, sheet)

    frame = pd.read_excel(
        path,
        sheet_name=sheet,
        header=header,
        dtype=str,
    )

    key_candidates = [
        column
        for column in frame.columns
        if "schl" in normalize_text(column)
        or "regional" in normalize_text(column)
    ]
    if not key_candidates:
        raise RuntimeError(
            f"Could not find Regional-schluessel column in VGRdL sheet {sheet}"
        )
    key_column = key_candidates[0]

    frame["AGS"] = frame[key_column].map(normalize_ags)

    year_candidates = [
        column
        for column in frame.columns
        if normalize_text(column) == str(INCOME_YEAR)
        or normalize_text(column).startswith(f"{INCOME_YEAR}.")
    ]
    if not year_candidates:
        raise RuntimeError(
            f"Could not find {INCOME_YEAR} in VGRdL table 2.4. "
            f"Columns={list(frame.columns)}"
        )

    # Duplicate year headings can be pandas-suffixed; the right-most is the
    # final value column in the table.
    year_column = year_candidates[-1]
    out = frame[["AGS", year_column]].rename(
        columns={
            year_column: "disposable_income_2023_eur_per_capita",
        }
    )
    out["disposable_income_2023_eur_per_capita"] = numeric_series(
        out["disposable_income_2023_eur_per_capita"]
    )
    out = out.dropna(subset=["AGS"]).drop_duplicates("AGS")
    return out, path


# ---------------------------------------------------------------------------
# Destatis Zensus regional-table discovery
# ---------------------------------------------------------------------------

class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append(
                (self._href, " ".join(self._text).strip())
            )
            self._href = None
            self._text = []


def discover_zensus_download(
    kind: str,
    *,
    refresh: bool,
) -> tuple[Path, str]:
    """
    Discover and cache the current public Destatis regional workbook.

    kind:
      "demografie"
      "bildung"
    """
    ZENSUS_DIR.mkdir(parents=True, exist_ok=True)
    cache = ZENSUS_DIR / f"regionaltabelle_{kind}.xlsx"
    url_cache = ZENSUS_DIR / f"regionaltabelle_{kind}.url.txt"

    if cache.exists() and cache.stat().st_size > 0 and not refresh:
        url = (
            url_cache.read_text(encoding="utf-8").strip()
            if url_cache.exists()
            else ZENSUS_REGIONAL_TABLES_PAGE
        )
        print(f"Using cached: {_relative(cache)}")
        return cache, url

    response = web_session().get(
        ZENSUS_REGIONAL_TABLES_PAGE,
        timeout=120,
        headers={"Referer": "https://www.destatis.de/"},
    )
    response.raise_for_status()

    collector = LinkCollector()
    collector.feed(response.text)

    candidates: list[tuple[int, str, str]] = []
    for href, link_text in collector.links:
        if not href:
            continue
        full_url = urljoin(
            ZENSUS_REGIONAL_TABLES_PAGE,
            html.unescape(href),
        )
        lower_url = full_url.lower().split("?")[0]
        if not lower_url.endswith((".xlsx", ".xls")):
            continue

        combined = normalize_text(link_text + " " + href)
        score = 0
        if "regionaltabelle" in combined:
            score += 4

        if kind == "demografie":
            if "demografie" in combined or "demographie" in combined:
                score += 10
        elif kind == "bildung":
            if "bildung" in combined:
                score += 7
            if "erwerb" in combined:
                score += 4
        else:
            raise ValueError(f"Unknown Zensus workbook kind {kind!r}")

        if score:
            candidates.append((score, full_url, link_text))

    if not candidates:
        raise RuntimeError(
            "Could not discover the public Destatis Zensus regional workbook "
            f"for {kind!r} from {ZENSUS_REGIONAL_TABLES_PAGE}. "
            "The official page may have changed its link markup."
        )

    score, url, text = max(candidates, key=lambda item: item[0])
    print(
        f"Discovered Zensus {kind} workbook: "
        f"{text or url} (score={score})"
    )
    download_file(
        url,
        cache,
        refresh=True,
        referer=ZENSUS_REGIONAL_TABLES_PAGE,
    )
    url_cache.write_text(url, encoding="utf-8")
    return cache, url


# ---------------------------------------------------------------------------
# Generic Zensus regional-table parser
# ---------------------------------------------------------------------------

def _find_header_column(raw: pd.DataFrame, patterns: Sequence[str], *, max_rows: int = 12) -> int | None:
    """Find a column from explicit header text near the top of a Zensus sheet."""
    limit = min(max_rows, len(raw))
    for column in raw.columns:
        values = [normalize_text(v) for v in raw.iloc[:limit, int(column)].tolist()]
        joined = " | ".join(v for v in values if v)
        if any(re.search(pattern, joined) for pattern in patterns):
            return int(column)
    return None


def zensus_kreis_rows(raw: pd.DataFrame) -> tuple[int, pd.Series, pd.Series, int]:
    """Return the official Kreis key column, 5-digit AGS codes and row mask.

    The Zensus regional workbooks mix Bund/Land/Kreis/Gemeindeverband/Gemeinde
    rows.  Do not guess the geography column by looking for values that happen
    to contain five digits: population/count columns also contain many 5-digit
    numbers.  Instead locate the explicitly labelled ARS/_RS column and the
    explicitly labelled ``Regionalebene`` column, then retain only rows whose
    level is Stadtkreis/kreisfreie Stadt/Landkreis and whose original key is
    exactly five digits.
    """
    key_col = _find_header_column(
        raw,
        [r"amtlicher.*regionalschlussel", r"_rs"],
    )
    level_col = _find_header_column(raw, [r"regionalebene"])
    if key_col is None or level_col is None:
        raise RuntimeError(
            "Could not find the explicitly labelled Zensus region-key and "
            "Regionalebene columns."
        )

    original = raw[key_col].astype(str).str.strip()
    exact_five = original.str.match(r"^\d{5}(?:\.0)?$", na=False)
    level = raw[level_col].map(normalize_text)
    kreis_level = level.str.contains(
        r"stadtkreis|kreisfreie stadt|landkreis",
        regex=True,
        na=False,
    )
    valid = exact_five & kreis_level
    codes = pd.Series(None, index=raw.index, dtype=object)
    codes.loc[valid] = original.loc[valid].map(normalize_ags)
    rows = np.flatnonzero(valid.to_numpy())
    if len(rows) < 50:
        raise RuntimeError(
            f"Zensus sheet yielded only {len(rows)} explicit Kreis rows; "
            "expected at least 50 and normally about 400."
        )
    return key_col, codes, valid, int(rows.min())


def ags_candidates(raw: pd.DataFrame) -> list[tuple[int, pd.Series, int]]:
    """Compatibility wrapper used by the generic parser/self-check."""
    try:
        key_col, codes, valid, _ = zensus_kreis_rows(raw)
    except RuntimeError:
        return []
    return [(key_col, codes, int(valid.sum()))]

def header_context(
    raw: pd.DataFrame,
    column: int,
    data_start: int,
    *,
    rows: int = 12,
) -> str:
    """Return the semantic header context for one Zensus value column.

    Destatis' formatted sheets use merged Excel cells. Pandas reads only the
    left-most value of a merged range, leaving NaN in neighbouring columns.
    Reconstruct that hierarchy by forward-filling horizontally across header
    rows before inspecting a value column. Also stop before the first actual
    geography row so Bund/Land counts are never mistaken for header text.
    """
    start = max(0, data_start - rows)
    end = data_start

    geography_labels = {
        "bund",
        "land",
        "gemeinde",
        "gemeindeverband",
        "stadtkreis/kreisfreie stadt/landkreis",
    }
    for row_idx in range(start, data_start):
        row_values = {
            normalize_text(v)
            for v in raw.iloc[row_idx].tolist()
            if normalize_text(v)
        }
        if row_values & geography_labels:
            end = row_idx
            break

    if end <= start:
        return ""

    header = raw.iloc[start:end].copy().ffill(axis=1)
    values = []
    for value in header.iloc[:, int(column)].tolist():
        norm = normalize_text(value)
        if norm and (not values or norm != values[-1]):
            values.append(norm)
    return " | ".join(values)


def all_text(frame: pd.DataFrame, *, max_rows: int = 60) -> str:
    head = frame.iloc[:max_rows].fillna("").astype(str)
    return normalize_text(
        " ".join(head.to_numpy().ravel())
    )


def target_phrase_mask(
    series: pd.Series,
    phrase_groups: Sequence[Sequence[str]],
) -> pd.Series:
    text = series.map(normalize_text)
    mask = pd.Series(True, index=series.index)
    for group in phrase_groups:
        group_mask = pd.Series(False, index=series.index)
        for phrase in group:
            phrase_norm = normalize_text(phrase)
            group_mask |= text.str.contains(
                re.escape(phrase_norm),
                regex=True,
                na=False,
            )
        mask &= group_mask
    return mask


def choose_value_column(
    raw: pd.DataFrame,
    rows: pd.Series,
    *,
    exclude: set[int],
    data_start: int,
    prefer_percent: bool,
) -> int | None:
    candidates: list[tuple[int, int]] = []

    for column in raw.columns:
        col_int = int(column)
        if col_int in exclude:
            continue

        numeric = numeric_series(raw.loc[rows, column])
        count = int(numeric.notna().sum())
        if count < 50:
            continue

        context = header_context(raw, col_int, data_start)
        score = count
        if prefer_percent and (
            "%" in context
            or "prozent" in context
            or "anteil" in context
        ):
            score += 1000
        candidates.append((score, col_int))

    return max(candidates)[1] if candidates else None


def extract_zensus_indicator_from_sheet(
    raw: pd.DataFrame,
    *,
    phrase_groups: Sequence[Sequence[str]],
    prefer_percent: bool = True,
) -> pd.DataFrame | None:
    """
    Handle two common machine-readable workbook layouts:
      - wide: one row per geography; categories are value columns
      - long: one row per geography/category; category is a text column
    """
    ags_info = ags_candidates(raw)
    if not ags_info:
        return None

    ags_column, codes, _ = ags_info[0]
    valid_geo = codes.notna()
    geo_rows = np.flatnonzero(valid_geo.to_numpy())
    if len(geo_rows) < 50:
        return None
    data_start = int(geo_rows.min())

    # Wide layout: category phrase appears in column header context.
    wide_candidates: list[tuple[int, int]] = []
    for column in raw.columns:
        col_int = int(column)
        if col_int == ags_column:
            continue
        context = header_context(raw, col_int, data_start)
        context_series = pd.Series([context])
        if bool(target_phrase_mask(context_series, phrase_groups).iloc[0]):
            score = 10
            if prefer_percent and (
                "%" in context
                or "prozent" in context
                or "anteil" in context
            ):
                score += 10
            numeric_count = int(
                numeric_series(raw.loc[valid_geo, column]).notna().sum()
            )
            score += numeric_count
            wide_candidates.append((score, col_int))

    if wide_candidates:
        _, value_column = max(wide_candidates)
        out = pd.DataFrame(
            {
                "AGS": codes.loc[valid_geo].values,
                "value": numeric_series(
                    raw.loc[valid_geo, value_column]
                ).values,
            }
        )
        return out.dropna(subset=["AGS"]).drop_duplicates("AGS")

    # Long layout: category phrase occurs in one of the row-value columns.
    for category_column in raw.columns:
        cat_int = int(category_column)
        if cat_int == ags_column:
            continue
        category_match = target_phrase_mask(
            raw[category_column],
            phrase_groups,
        )
        rows = valid_geo & category_match
        if int(rows.sum()) < 50:
            continue

        value_column = choose_value_column(
            raw,
            rows,
            exclude={ags_column, cat_int},
            data_start=data_start,
            prefer_percent=prefer_percent,
        )
        if value_column is None:
            continue

        out = pd.DataFrame(
            {
                "AGS": codes.loc[rows].values,
                "value": numeric_series(
                    raw.loc[rows, value_column]
                ).values,
            }
        )
        return out.dropna(subset=["AGS"]).drop_duplicates("AGS")

    return None


def candidate_zensus_sheets(
    workbook: Path,
    *,
    sheet_terms: Sequence[str],
) -> list[str]:
    xls = pd.ExcelFile(workbook)
    scored: list[tuple[int, str]] = []

    for sheet in xls.sheet_names:
        preview = pd.read_excel(
            workbook,
            sheet_name=sheet,
            header=None,
            nrows=60,
            dtype=object,
        )
        text = normalize_text(sheet) + " " + all_text(preview)
        score = sum(
            1 for term in sheet_terms
            if normalize_text(term) in text
        )
        if score:
            scored.append((score, sheet))

    return [
        sheet
        for _score, sheet in sorted(scored, reverse=True)
    ]


def extract_zensus_indicator(
    workbook: Path,
    *,
    output_column: str,
    sheet_terms: Sequence[str],
    phrase_groups: Sequence[Sequence[str]],
    prefer_percent: bool = True,
) -> pd.DataFrame:
    sheets = candidate_zensus_sheets(
        workbook,
        sheet_terms=sheet_terms,
    )
    if not sheets:
        raise RuntimeError(
            f"No candidate Zensus sheet for {output_column}; "
            f"sheet terms={list(sheet_terms)}"
        )

    for sheet in sheets:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet,
            header=None,
            dtype=object,
        )
        extracted = extract_zensus_indicator_from_sheet(
            raw,
            phrase_groups=phrase_groups,
            prefer_percent=prefer_percent,
        )
        if (
            extracted is not None
            and extracted["value"].notna().sum() >= 50
        ):
            return extracted.rename(
                columns={"value": output_column}
            )

    raise RuntimeError(
        f"Found candidate Zensus sheets for {output_column} but could not "
        f"safely parse the requested category. Candidates={sheets}. "
        f"Inspect cached workbook {_relative(workbook)} if Destatis changed "
        "the machine-readable layout."
    )


def _formatted_zensus_sheet(workbook: Path, wanted: str) -> str:
    """Resolve one human-readable Zensus sheet by normalized name."""
    wanted_norm = normalize_text(wanted)
    xls = pd.ExcelFile(workbook)
    exact = [s for s in xls.sheet_names if normalize_text(s) == wanted_norm]
    if exact:
        return exact[0]
    partial = [s for s in xls.sheet_names if wanted_norm in normalize_text(s)]
    if len(partial) == 1:
        return partial[0]
    raise RuntimeError(
        f"Could not uniquely resolve Zensus sheet {wanted!r}; sheets={xls.sheet_names}"
    )


def _zensus_total_count(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract the unsplit 'Insgesamt' count for the 400 Kreis rows."""
    key_col, codes, valid_geo, data_start = zensus_kreis_rows(raw)
    candidates: list[tuple[int, int]] = []
    for column in raw.columns:
        col = int(column)
        if col == key_col:
            continue
        context = header_context(raw, col, data_start)
        if "insgesamt" not in context:
            continue
        # Prefer the unsplit total, not male/female subcolumns.
        penalty = 1000 if ("mannlich" in context or "weiblich" in context) else 0
        n = int(numeric_series(raw.loc[valid_geo, column]).notna().sum())
        candidates.append((n - penalty, col))
    if not candidates:
        raise RuntimeError("Could not identify the Zensus 'Insgesamt' count column")
    _, value_col = max(candidates)
    print(f"    Zensus denominator column {value_col}: {header_context(raw, value_col, data_start)[:120]}")
    out = pd.DataFrame({
        "AGS": codes.loc[valid_geo].values,
        "value": numeric_series(raw.loc[valid_geo, value_col]).values,
    })
    return out.dropna(subset=["AGS"]).drop_duplicates("AGS")


def _zensus_category_count(
    raw: pd.DataFrame,
    phrase_groups: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Extract a category's unsplit Kreis count from a formatted Zensus sheet."""
    key_col, codes, valid_geo, data_start = zensus_kreis_rows(raw)
    candidates: list[tuple[int, int]] = []
    for column in raw.columns:
        col = int(column)
        if col == key_col:
            continue
        context = header_context(raw, col, data_start)
        if not bool(target_phrase_mask(pd.Series([context]), phrase_groups).iloc[0]):
            continue
        # In the formatted tables, a category heading starts at the 'zusammen'
        # column of its male/female triplet.  Prefer that column explicitly.
        score = int(numeric_series(raw.loc[valid_geo, column]).notna().sum())
        if "zusammen" in context:
            score += 2000
        if "mannlich" in context or "weiblich" in context:
            score -= 1000
        candidates.append((score, col))
    if not candidates:
        raise RuntimeError(
            "Could not identify Zensus category column for phrases="
            f"{phrase_groups}"
        )
    _, value_col = max(candidates)
    print(f"    Zensus category column {value_col}: {header_context(raw, value_col, data_start)[:120]}")
    out = pd.DataFrame({
        "AGS": codes.loc[valid_geo].values,
        "value": numeric_series(raw.loc[valid_geo, value_col]).values,
    })
    return out.dropna(subset=["AGS"]).drop_duplicates("AGS")


def _share_from_counts(numerator: pd.DataFrame, denominator: pd.DataFrame, output: str) -> pd.DataFrame:
    merged = numerator.rename(columns={"value": "num"}).merge(
        denominator.rename(columns={"value": "den"}),
        on="AGS",
        how="outer",
        validate="one_to_one",
    )
    merged[output] = np.where(
        merged["den"] > 0,
        merged["num"] / merged["den"] * 100.0,
        np.nan,
    )
    return merged[["AGS", output]]


def load_zensus_indicators(
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, str]]:
    demography_path, demography_url = discover_zensus_download(
        "demografie",
        refresh=refresh,
    )
    education_path, education_url = discover_zensus_download(
        "bildung",
        refresh=refresh,
    )

    # Use the human-readable sheets to map the official category labels to the
    # machine values.  The Regionaltabellen provide counts; derive percentages
    # ourselves with a same-sheet 2022 denominator instead of pretending those
    # count columns are already percentages.
    demo_sheet = _formatted_zensus_sheet(demography_path, "Demografie")
    demo_raw = pd.read_excel(demography_path, sheet_name=demo_sheet, header=None, dtype=object)
    demo_total = _zensus_total_count(demo_raw)

    migration_count = _zensus_category_count(
        demo_raw,
        [["mit einwanderungsgeschichte", "einwanderungsgeschichte: mit"]],
    )
    migration = _share_from_counts(
        migration_count,
        demo_total,
        "immigration_history_2022_pct",
    )

    foreign_count = _zensus_category_count(
        demo_raw,
        [
            ["staatsangehorigkeit"],
            ["auslandisch", "ausland", "nicht deutsch", "ausland und sonstige"],
        ],
    )
    foreign = _share_from_counts(
        foreign_count,
        demo_total,
        "foreign_national_2022_pct",
    )

    education_sheet = _formatted_zensus_sheet(
        education_path,
        "Höchster berufl. Abschluss",
    )
    education_raw = pd.read_excel(
        education_path,
        sheet_name=education_sheet,
        header=None,
        dtype=object,
    )
    education_total = _zensus_total_count(education_raw)

    no_voc_count = _zensus_category_count(
        education_raw,
        [[
            "ohne berufl. bildungsabschluss",
            "ohne beruflichen bildungsabschluss",
            "ohne beruflichen abschluss",
            "ohne berufsabschluss",
        ]],
    )
    no_vocational = _share_from_counts(
        no_voc_count,
        education_total,
        "no_vocational_qualification_2022_pct",
    )

    academic_counts: list[pd.DataFrame] = []
    for phrases in (["bachelor"], ["master"], ["diplom"], ["promotion"]):
        academic_counts.append(_zensus_category_count(education_raw, [phrases]))

    academic = academic_counts[0].rename(columns={"value": "academic_0"})
    for i, part in enumerate(academic_counts[1:], start=1):
        academic = academic.merge(
            part.rename(columns={"value": f"academic_{i}"}),
            on="AGS",
            how="outer",
            validate="one_to_one",
        )
    academic_cols = [c for c in academic.columns if c.startswith("academic_")]
    academic["value"] = academic[academic_cols].sum(axis=1, min_count=len(academic_cols))
    university = _share_from_counts(
        academic[["AGS", "value"]],
        education_total,
        "university_degree_2022_pct",
    )

    out = migration
    for part in (foreign, no_vocational, university):
        out = out.merge(part, on="AGS", how="outer", validate="one_to_one")

    print(
        "  Zensus Kreis rows: "
        f"migration={migration['immigration_history_2022_pct'].notna().sum()}, "
        f"foreign={foreign['foreign_national_2022_pct'].notna().sum()}, "
        f"no_vocational={no_vocational['no_vocational_qualification_2022_pct'].notna().sum()}, "
        f"university={university['university_degree_2022_pct'].notna().sum()}"
    )

    return (
        out,
        {"demography": demography_path, "education": education_path},
        {"demography": demography_url, "education": education_url},
    )


# ---------------------------------------------------------------------------
# Validation / coverage
# ---------------------------------------------------------------------------

CORE_COLUMNS = [
    "population_2024",
    "population_under15_2024",
    "population_15_64_2024",
    "population_65plus_2024",
    "share_under15_2024_pct",
    "share_15_64_2024_pct",
    "share_65plus_2024_pct",
    "median_age_2024",
    "population_density_2024_per_km2",
    "unemployment_rate_2024_pct",
    "disposable_income_2023_eur_per_capita",
    "immigration_history_2022_pct",
    "foreign_national_2022_pct",
    "no_vocational_qualification_2022_pct",
    "university_degree_2022_pct",
]


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        if column in {"AGS", "Name", "NUTS"}:
            continue
        n = int(frame[column].notna().sum())
        rows.append(
            {
                "variable": column,
                "n_available": n,
                "n_total": len(frame),
                "coverage_pct": (
                    100.0 * n / len(frame)
                    if len(frame)
                    else float("nan")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("variable")
        .reset_index(drop=True)
    )


def print_coverage(frame: pd.DataFrame) -> None:
    coverage = coverage_table(frame)
    print(f"\nCoverage ({len(frame)} Kreise):")
    for row in coverage.itertuples(index=False):
        print(
            f"  {row.variable:<50} "
            f"{row.n_available:>3}/{row.n_total} "
            f"({row.coverage_pct:5.1f}%)"
        )


def assert_range(
    frame: pd.DataFrame,
    column: str,
    lo: float,
    hi: float,
) -> None:
    values = frame[column].dropna()
    bad = values[(values < lo) | (values > hi)]
    if len(bad):
        raise AssertionError(
            f"{column}: {len(bad)} values outside [{lo}, {hi}]"
        )


def validate_final(frame: pd.DataFrame) -> None:
    if frame["AGS"].duplicated().any():
        raise AssertionError("Final table has duplicate AGS")
    if frame["NUTS"].duplicated().any():
        raise AssertionError("Final table has duplicate NUTS")

    if len(frame) != EXPECTED_2024_KREISE:
        raise AssertionError(
            f"Final 2024 geography has {len(frame)} rows; "
            f"expected {EXPECTED_2024_KREISE}. "
            "Resolve the boundary/NUTS vintage before using the table."
        )

    for column in CORE_COLUMNS:
        if column not in frame.columns:
            raise AssertionError(f"Missing expected column {column}")
        coverage = float(frame[column].notna().mean())
        if coverage < 0.95:
            raise AssertionError(
                f"{column} coverage is only {coverage:.1%}; "
                "do not silently use an incomplete socioeconomic table."
            )

    assert_range(frame, "share_under15_2024_pct", 0, 40)
    assert_range(frame, "share_15_64_2024_pct", 30, 90)
    assert_range(frame, "share_65plus_2024_pct", 0, 60)
    assert_range(frame, "median_age_2024", 20, 70)
    assert_range(frame, "unemployment_rate_2024_pct", 0, 30)
    assert_range(frame, "immigration_history_2022_pct", 0, 100)
    assert_range(frame, "foreign_national_2022_pct", 0, 100)
    assert_range(frame, "no_vocational_qualification_2022_pct", 0, 100)
    assert_range(frame, "university_degree_2022_pct", 0, 100)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(*, refresh: bool) -> pd.DataFrame:
    SOURCE_RECORDS.clear()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading project AGS/NUTS crosswalk...")
    crosswalk = load_geo_crosswalk(refresh=refresh)
    print(f"  boundary rows: {len(crosswalk)}")

    print("\n1/6 Eurostat population and age groups...")
    population, population_cache = load_population_age(
        refresh=refresh
    )

    # Use the 2024 NUTS population data to define the current statistical base.
    valid_2024_nuts = set(population["NUTS"].dropna())
    base = (
        crosswalk[crosswalk["NUTS"].isin(valid_2024_nuts)]
        .sort_values("AGS")
        .drop_duplicates("AGS")
        .reset_index(drop=True)
    )
    print(f"  2024 NUTS3-backed Kreis base: {len(base)} rows")

    merged = safe_merge(
        base,
        population,
        key="NUTS",
        label="Eurostat population",
    )
    record_source(
        merged,
        [
            "population_2024",
            "population_under15_2024",
            "population_15_64_2024",
            "population_65plus_2024",
            "share_under15_2024_pct",
            "share_15_64_2024_pct",
            "share_65plus_2024_pct",
        ],
        source="Eurostat",
        reference_year=YEAR,
        source_id_or_url="demo_r_pjanaggr3",
        cached_file=population_cache,
    )

    print("\n2/6 Eurostat median age...")
    median_age, median_cache = load_median_age(refresh=refresh)
    merged = safe_merge(
        merged,
        median_age,
        key="NUTS",
        label="Eurostat median age",
    )
    record_source(
        merged,
        ["median_age_2024"],
        source="Eurostat",
        reference_year=YEAR,
        source_id_or_url="demo_r_pjanind3",
        cached_file=median_cache,
    )

    print("\n3/6 Eurostat population density...")
    density, density_cache = load_population_density(
        refresh=refresh
    )
    merged = safe_merge(
        merged,
        density,
        key="NUTS",
        label="Eurostat population density",
    )
    record_source(
        merged,
        ["population_density_2024_per_km2"],
        source="Eurostat",
        reference_year=YEAR,
        source_id_or_url="demo_r_d3dens",
        cached_file=density_cache,
        note=(
            "Used primarily as an urbanicity/structural control, not as a "
            "socioeconomic disadvantage measure."
        ),
    )

    print("\n4/6 Bundesagentur fuer Arbeit unemployment...")
    unemployment, unemployment_cache = load_unemployment(
        refresh=refresh
    )
    merged = safe_merge(
        merged,
        unemployment,
        key="AGS",
        label="BA unemployment",
    )
    record_source(
        merged,
        ["unemployment_rate_2024_pct"],
        source="Bundesagentur fuer Arbeit",
        reference_year=YEAR,
        source_id_or_url=BA_UNEMPLOYMENT_URL,
        cached_file=unemployment_cache,
    )

    print("\n5/6 VGRdL disposable household income...")
    income, income_cache = load_disposable_income(
        refresh=refresh
    )
    merged = safe_merge(
        merged,
        income,
        key="AGS",
        label="VGRdL disposable income",
    )
    record_source(
        merged,
        ["disposable_income_2023_eur_per_capita"],
        source="VGRdL",
        reference_year=INCOME_YEAR,
        source_id_or_url=VGRDL_INCOME_URL,
        cached_file=income_cache,
        note=(
            "Disposable household income per inhabitant; 2023 is retained "
            "explicitly because Kreis-level VGRdL income currently lags the "
            "2024 pollution/demographic reference year."
        ),
    )

    print("\n6/6 Destatis Zensus migration and education...")
    zensus, zensus_paths, zensus_urls = load_zensus_indicators(
        refresh=refresh
    )
    merged = safe_merge(
        merged,
        zensus,
        key="AGS",
        label="Zensus migration/education",
    )
    record_source(
        merged,
        [
            "immigration_history_2022_pct",
            "foreign_national_2022_pct",
        ],
        source="Zensus 2022 / Statistisches Bundesamt",
        reference_year=ZENSUS_YEAR,
        source_id_or_url=zensus_urls["demography"],
        cached_file=zensus_paths["demography"],
        note=(
            "Migration variables are demographic characteristics; they must "
            "not be interpreted as socioeconomic disadvantage by themselves."
        ),
    )
    record_source(
        merged,
        [
            "no_vocational_qualification_2022_pct",
            "university_degree_2022_pct",
        ],
        source="Zensus 2022 / Statistisches Bundesamt",
        reference_year=ZENSUS_YEAR,
        source_id_or_url=zensus_urls["education"],
        cached_file=zensus_paths["education"],
    )

    column_order = [
        "AGS",
        "Name",
        "NUTS",
        "population_2024",
        "population_under15_2024",
        "population_15_64_2024",
        "population_65plus_2024",
        "share_under15_2024_pct",
        "share_15_64_2024_pct",
        "share_65plus_2024_pct",
        "median_age_2024",
        "population_density_2024_per_km2",
        "unemployment_rate_2024_pct",
        "disposable_income_2023_eur_per_capita",
        "immigration_history_2022_pct",
        "foreign_national_2022_pct",
        "no_vocational_qualification_2022_pct",
        "university_degree_2022_pct",
    ]
    merged = (
        merged[column_order]
        .sort_values("AGS")
        .reset_index(drop=True)
    )

    print_coverage(merged)
    validate_final(merged)

    merged.to_csv(OUT_CSV, index=False)
    coverage_table(merged).to_csv(OUT_COVERAGE, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output": _relative(OUT_CSV),
        "geography": {
            "reference": "2024 NUTS3-backed Kreis geography",
            "n_rows": len(merged),
            "expected_n_rows": EXPECTED_2024_KREISE,
            "crosswalk": (
                _relative(ADMIN_PATH) if ADMIN_PATH.exists()
                else _relative(BKG_DIR / "vg250-ew_12-31.utm32s.gpkg.ebenen.zip")
            ),
            "note": (
                "AGS/NUTS/name comes from the existing project boundary GeoJSON when "
                "available, otherwise from the official BKG VG250-EW 31.12.2024 Excel "
                "archive. The final base is restricted to NUTS3 codes present in "
                "Eurostat 2024 population data."
            ),
        },
        "analysis_notes": {
            "mortality": "Not included in the primary table by design.",
            "material_deprivation": "Not included by design.",
            "migration": (
                "Immigration history / nationality are demographic "
                "characteristics, not SES-disadvantage proxies."
            ),
            "population_density": (
                "Intended mainly as an urbanicity/structural control."
            ),
            "mixed_reference_years": (
                "The true reference year is encoded in every column name. "
                "Zensus 2022 and VGRdL 2023 indicators are not relabeled 2024."
            ),
        },
        "sources": [asdict(record) for record in SOURCE_RECORDS],
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved -> {_relative(OUT_CSV)}")
    print(f"Saved -> {_relative(OUT_COVERAGE)}")
    print(f"Saved -> {_relative(OUT_MANIFEST)}")
    return merged


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def self_check() -> None:
    assert normalize_ags("08111") == "08111"
    assert normalize_ags("8111.0") == "08111"
    assert normalize_text("Verfügbares Einkommen") == "verfugbares einkommen"

    assert math.isclose(parse_number("8,3"), 8.3)
    assert math.isclose(parse_number("1.234,5"), 1234.5)
    assert math.isclose(parse_number("8.3"), 8.3)
    assert math.isnan(parse_number(":"))

    payload = {
        "class": "dataset",
        "id": ["geo", "time"],
        "size": [2, 1],
        "dimension": {
            "geo": {
                "category": {
                    "index": {"DE111": 0, "DE112": 1},
                    "label": {"DE111": "A", "DE112": "B"},
                }
            },
            "time": {
                "category": {
                    "index": {"2024": 0},
                    "label": {"2024": "2024"},
                }
            },
        },
        "value": [10.0, 20.0],
    }
    frame = jsonstat_to_frame(payload)
    assert frame["geo"].tolist() == ["DE111", "DE112"]
    assert frame["value"].tolist() == [10.0, 20.0]

    collector = LinkCollector()
    collector.feed(
        '<a href="/regional_demografie.xlsx">'
        "Regionaltabelle Demografie</a>"
    )
    assert collector.links == [
        ("/regional_demografie.xlsx", "Regionaltabelle Demografie")
    ]

    phrase = pd.Series(["Mit Einwanderungsgeschichte"])
    assert bool(
        target_phrase_mask(
            phrase,
            [["mit einwanderungsgeschichte"]],
        ).iloc[0]
    )

    # Zensus geography selection must use the labelled ARS/_RS + Regionalebene
    # columns, not whichever numeric column happens to contain many five-digit
    # values.  Include a misleading count column to guard against that bug.
    rows = [
        ["Amtlicher Regionalschlüssel (ARS)", "Name", "Regionalebene", "Insgesamt"],
        [None, None, None, None],
    ]
    for i in range(60):
        rows.append([f"{10000+i:05d}", f"Kreis {i}", "Stadtkreis/kreisfreie Stadt/Landkreis", 50000+i])
    rows.append(["010010000000", "Gemeinde", "Gemeinde", 50001])
    synthetic = pd.DataFrame(rows)
    key_col, kreis_codes, valid_geo, _ = zensus_kreis_rows(synthetic)
    assert key_col == 0 and int(valid_geo.sum()) == 60
    assert kreis_codes.loc[valid_geo].notna().all()

    print("socioeconomic Kreis builder self-check passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically gather and build the Kreis-level socioeconomic "
            "table for the 2024 air-pollution exposure analysis."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download/re-query all public sources instead of using cache",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run lightweight parser checks without internet and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
