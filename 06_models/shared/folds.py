import pandas as pd

from . import paths

FOLDS = {
    "fold1_iberia": ["PT", "ES", "AD"],
    "fold2_france": ["FR", "NL", "BE", "LU"],
    "fold3_italy": ["IT", "MT"],
    "fold4_alpine": ["DE", "CH", "AT"],
    "fold5_north": ["DK", "SE", "NO", "FI", "IS", "IE", "LT", "LV", "EE"],
    "fold6_balkan_e": ["HU", "SI", "HR", "BA", "RS", "XK", "ME", "RO", "BG"],
    "fold7_balkan_s": ["AL", "GR", "CY", "TR", "MK"],
    "fold8_poland": ["PL", "CZ", "SK"],
}

DE_TEST_LAENDER = {
    "Brandenburg",
    "Mecklenburg-Vorpommern",
    "Sachsen",
    "Sachsen-Anhalt",
    "Thueringen",
    "Berlin",
    "Hamburg",
    "Bremen",
    "Niedersachsen",
    "Schleswig-Holstein",
}

COUNTRY_TO_FOLD = {cc: fold for fold, countries in FOLDS.items() for cc in countries}
FOLD_ORDER = list(FOLDS)
NON_DEV_FOLDS = {"TEST", "UNASSIGNED"}


def load_de_land(path=paths.STATION_LAND):
    if path.exists():
        sl = pd.read_csv(path, dtype={"station_code": str})
        return dict(zip(sl["station_code"], sl["land"]))
    print(f"WARNING: {path} not found -- no German Land split; all DE -> fold4_alpine")
    return {}


def assign_fold(code, de_land):
    cc = code[:2]
    if cc == "DE" and de_land.get(code) in DE_TEST_LAENDER:
        return "TEST"
    return COUNTRY_TO_FOLD.get(cc)


def load_station_folds(path=paths.STATION_FOLD):
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found. Run 06_models/00_assign_folds.py first to "
            "generate the frozen fold assignment and QC map."
        )
    return pd.read_csv(path, dtype={"station_code": str})


def development_fold_names(station_folds=None):
    sf = load_station_folds() if station_folds is None else station_folds
    present = [f for f in sf["fold"].dropna().unique() if f not in NON_DEV_FOLDS]
    known = [f for f in FOLD_ORDER if f in present]
    extra = sorted(f for f in present if f not in FOLD_ORDER)
    return known + extra


def fold_definition_signature(station_folds=None):
    sf = load_station_folds() if station_folds is None else station_folds
    out = {}
    for fold, sub in sf.groupby("fold", dropna=False):
        countries = sorted(c for c in sub["country"].dropna().unique())
        lands = sorted(l for l in sub.get("land", pd.Series(dtype=str)).dropna().unique() if l)
        out[fold] = {"stations": int(len(sub)), "countries": countries, "lands": lands}
    return out

