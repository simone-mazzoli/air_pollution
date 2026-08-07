#!/usr/bin/env python3
"""Check that the model data tree is ready before running 06_models."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "06_models"
sys.path.insert(0, str(MODELS))

from shared import folds, paths  # noqa: E402
from shared.config import (  # noqa: E402
    MAX_PM10,
    MAX_PM25,
    POLLUTANTS,
    S5P_STREAMS,
    USE_AER_WIDE,
    USE_DEM,
)


def main():
    errors = []
    warnings = []

    def require_file(path, why):
        if path.exists():
            print(f"OK   {path}")
            return True
        errors.append(f"Missing {path} ({why})")
        print(f"MISS {path}")
        return False

    def require_dir(path, why):
        if path.is_dir():
            print(f"OK   {path}")
            return True
        errors.append(f"Missing directory {path} ({why})")
        print(f"MISS {path}")
        return False

    print("== Required tabular files ==")
    have_labels = require_file(paths.LABELS, "build with 02_scripts_cleaning/01_build_eea_labels.py")
    have_land = require_file(paths.STATION_LAND, "needed to assign German stations to TEST/fold4")
    have_folds = require_file(paths.STATION_FOLD, "run 06_models/00_assign_folds.py after labels exist")

    labels = None
    if have_labels:
        labels = pd.read_csv(paths.LABELS, dtype={"station_code": str})
        needed = {"station_code", "lat", "lon", "Datum", "PM10", "PM2.5"}
        missing = sorted(needed - set(labels.columns))
        if missing:
            errors.append(f"{paths.LABELS} is missing columns: {missing}")
        else:
            countries = labels["station_code"].str[:2].value_counts().sort_index()
            print(f"labels: {len(labels):,} rows, {labels['station_code'].nunique():,} stations, {len(countries)} countries")
            print("countries:", dict(countries))
            if len(countries) <= 1:
                errors.append("Label file is not Europe-wide; it contains only one country.")

    if have_land:
        land = pd.read_csv(paths.STATION_LAND, dtype={"station_code": str})
        missing = sorted({"station_code", "land"} - set(land.columns))
        if missing:
            errors.append(f"{paths.STATION_LAND} is missing columns: {missing}")

    station_folds = None
    if have_folds:
        station_folds = pd.read_csv(paths.STATION_FOLD, dtype={"station_code": str})
        needed = {"station_code", "country", "fold", "lat", "lon"}
        missing = sorted(needed - set(station_folds.columns))
        if missing:
            errors.append(f"{paths.STATION_FOLD} is missing columns: {missing}")
        else:
            present = set(station_folds["fold"].dropna())
            missing_folds = [f for f in folds.FOLD_ORDER if f not in present]
            if "TEST" not in present:
                missing_folds.append("TEST")
            if missing_folds:
                errors.append(f"station_fold.csv is missing expected folds: {missing_folds}")
            print("fold counts:", station_folds["fold"].value_counts().sort_index().to_dict())

    print("\n== Required satellite patch folders ==")
    patch_dirs = [
        ("high_res_multispec", paths.HIGH, (120, 120, 10)),
        ("low_res_multispec", paths.LOW, (60, 60, 10)),
    ]
    patch_dirs += [(f"{s}_tropomi", paths.SAT / f"{s}_tropomi", (5, 5)) for s in S5P_STREAMS]
    if USE_AER_WIDE:
        patch_dirs.append(("aer_wide_tropomi", paths.AERW, (31, 31)))
    if USE_DEM:
        patch_dirs.append(("dem_glo30", paths.DEMD, (60, 60)))

    existing_patch_dirs = []
    for name, path, expected_shape in patch_dirs:
        if not require_dir(path, f"unpack satellite patches into {paths.SAT}"):
            continue
        files = sorted(path.glob("*.npy"))
        print(f"  {name}: {len(files):,} .npy files")
        if not files:
            errors.append(f"{path} has no .npy files")
            continue
        existing_patch_dirs.append((name, path))
        sample = np.load(files[0])
        shape = sample.shape
        ok = shape == expected_shape or (len(expected_shape) == 2 and shape[:2] == expected_shape[:2])
        if not ok:
            warnings.append(f"{files[0]} has shape {shape}, expected about {expected_shape}")
        print(f"  sample shape: {shape}")

    if (
        labels is not None
        and station_folds is not None
        and {"station_code", "PM10", "PM2.5"} <= set(labels.columns)
        and {"station_code", "fold"} <= set(station_folds.columns)
        and existing_patch_dirs
    ):
        print("\n== Model-ready station counts ==")
        ann = labels.groupby("station_code").agg({"PM10": "mean", "PM2.5": "mean"}).reset_index()
        ann = ann.rename(columns={"PM10": "pm10", "PM2.5": "pm25"})
        ann.loc[(ann["pm10"] <= 0) | (ann["pm10"] > MAX_PM10), "pm10"] = np.nan
        ann.loc[(ann["pm25"] <= 0) | (ann["pm25"] > MAX_PM25), "pm25"] = np.nan
        ann = ann.merge(station_folds[["station_code", "fold"]], on="station_code", how="left")

        def has_patches(code):
            return all((path / f"{code}.npy").exists() for _, path in existing_patch_dirs)

        ann["has_all_patches"] = ann["station_code"].map(has_patches)
        for pollutant in POLLUTANTS:
            valid = ann[pollutant].notna()
            ready = ann[valid & ann["has_all_patches"]]
            counts = ready["fold"].value_counts().sort_index().to_dict()
            print(f"{pollutant} ready stations by fold:", counts)
            for fold in folds.FOLD_ORDER + ["TEST"]:
                if counts.get(fold, 0) == 0:
                    errors.append(f"No {pollutant} stations with all patches in {fold}")

    if warnings:
        print("\n== Warnings ==")
        for msg in warnings:
            print(f"WARN {msg}")
    if errors:
        print("\n== Data check failed ==")
        for msg in errors:
            print(f"FAIL {msg}")
        raise SystemExit(1)

    print("\nData check passed.")


if __name__ == "__main__":
    main()
