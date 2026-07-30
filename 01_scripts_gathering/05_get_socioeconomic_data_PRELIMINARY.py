"""
Downloads district-level (Kreis) socioeconomic indicators from INKAR (BBSR)
for the environmental-justice exposure analysis.

Source: INKAR 2024 full-database bulk export (BBSR):
https://www.bbr-server.de/imagemap/inkar/download/inkar_2024.zip
The zip contains one ~6.8 GB CSV with every indicator, every spatial level,
and every year back to ~1995 -- far more than we need. This script never
extracts that CSV to disk: it streams it directly from the zip (Deflate64,
hence the zipfile_inflate64 import -- stdlib zipfile can't decompress this
codec on its own) and keeps only the handful of rows we actually want,
row by row, discarding the rest as it goes.

Target year is 2021 for all four indicators -- the newest year with Kreis
data for every one of them (income and unemployment stop at 2021; the two
school-leaver ones would go to 2022, but 2021 was chosen for consistency
across all four rather than mixing report years).

Migration background was considered and dropped: it comes from the
Mikrozensus and is not reliably published at Kreis level (small-sample
suppression), unlike everything below.

Two non-obvious things worth knowing before touching this data:
  - INKAR's Kennziffer for Raumbezug=="Kreise" is the 5-digit AGS
    (Amtlicher Gemeindeschluessel, same as VG250's AGS column, see
    04_get_admin_boundaries.py) zero-padded to 8 digits, e.g. "08435000"
    for Bodenseekreis ("08435" + "000"). AGS here is derived by truncating
    to the first 5 characters.
  - Several indicators (e.g. a_schul_oA) appear twice under the same
    Kreis/year -- once under Bereich "Bildung", once under "SDG-Indikatoren
    fuer Kommunen" -- with an identical value both times (it's the same
    indicator cross-listed in two thematic categories, not a data
    inconsistency). Deduplicated by (Kuerzel, Kennziffer) here.

Input:  none (downloads from BBSR)
Output: data/raw/inkar/inkar_2024.zip                        (cached download, ~395 MB)
        data/processed/socioeconomic/inkar_kreis_2021.csv    (400 Kreise x 4 indicators)
"""

import io
import zipfile
from pathlib import Path

import zipfile_inflate64  # noqa: F401 -- patches zipfile to support Deflate64 (method 9)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "inkar"
OUT_PATH = BASE_DIR / "data" / "processed" / "socioeconomic" / "inkar_kreis_2021.csv"

INKAR_URL = "https://www.bbr-server.de/imagemap/inkar/download/inkar_2024.zip"
ZIP_PATH = RAW_DIR / "inkar_2024.zip"
CSV_MEMBER = "inkar_2024.csv"

TARGET_YEAR = "2021"

# Kuerzel -> (output column name, Klartext) -- district-level (Kreise) proxies
# for the socioeconomic variables from the project brief, minus migration
# background (see module docstring).
INDICATORS = {
    "m_ek": "median_einkommen",  # Medianeinkommen (per capita income proxy)
    "q_alo": "arbeitslosenquote",  # Arbeitslosenquote (unemployment rate)
    "a_schul_abi": "schulabgaenger_abitur_anteil",  # school leavers w/ Abitur (%)
    "a_schul_oA": "schulabgaenger_ohne_abschluss_anteil",  # school leavers w/o any degree (%)
    "m_bev_alter": "durchschnittsalter",  # average population age
}


def download_inkar() -> None:
    if ZIP_PATH.exists():
        print(f"Using cached download: {ZIP_PATH}")
        return
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading INKAR 2024 from BBSR ({INKAR_URL})...")
    print("  this is a ~395 MB download; the ~6.8 GB CSV inside is never")
    print("  extracted to disk, only streamed and filtered on the fly.")
    resp = requests.get(INKAR_URL, timeout=300)
    resp.raise_for_status()
    ZIP_PATH.write_bytes(resp.content)
    print(f"  saved {len(resp.content) / 1024**2:.1f} MiB -> {ZIP_PATH}")


def extract_kreis_rows() -> list[dict]:
    rows = []
    seen = set()  # (Kuerzel, Kennziffer) -- dedupes the Bereich cross-listing

    with zipfile.ZipFile(ZIP_PATH) as z, z.open(CSV_MEMBER) as raw:
        f = io.TextIOWrapper(raw, encoding="utf-8")
        header = next(f).rstrip("\n").split(";")
        assert header == [
            "Bereich", "ID", "Kuerzel", "Indikator", "Raumbezug",
            "Kennziffer", "Name", "Zeitbezug", "Wert",
        ], f"unexpected INKAR CSV header: {header}"

        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) != 9:
                continue
            _, _, kuerzel, _, raumbezug, kennziffer, name, jahr, wert = parts

            if (
                raumbezug != "Kreise"
                or kuerzel not in INDICATORS
                or jahr != TARGET_YEAR
            ):
                continue

            key = (kuerzel, kennziffer)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "AGS": kennziffer[:5],
                "Name": name,
                "indicator": INDICATORS[kuerzel],
                "value": float(wert.replace(",", ".")),
            })

    return rows


def main() -> None:
    download_inkar()

    print(f"\nStreaming {CSV_MEMBER} from the zip (never fully extracted)...")
    rows = extract_kreis_rows()
    print(f"  kept {len(rows)} matching rows out of ~56M scanned")

    import pandas as pd

    long_df = pd.DataFrame(rows)
    wide = long_df.pivot_table(
        index=["AGS", "Name"], columns="indicator", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for col in INDICATORS.values():
        n = wide[col].notna().sum() if col in wide.columns else 0
        print(f"  {col}: {n}/400 Kreise")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(
        "\nJoin key: AGS (5-digit Amtlicher Gemeindeschluessel), matches "
        "vg250_kreis.geojson's AGS column from 04_get_admin_boundaries.py."
    )


if __name__ == "__main__":
    main()
