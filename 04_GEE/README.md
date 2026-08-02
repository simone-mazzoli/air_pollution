# Earth Engine Satellite Patches

## Purpose

This directory downloads and inspects Sentinel-2 image patches centered on calibrated low-cost sensor locations. The patches are intended as CNN inputs.

## Scripts

| Script | What it does | Main inputs | Main outputs | Role |
| --- | --- | --- | --- | --- |
| `01_download_satellite_patches.py` | Authenticates with Google Earth Engine, builds a 2024 cloud-filtered Sentinel-2 median composite, and downloads high-resolution and low-resolution multispectral patches for calibrated sensor locations. | Earth Engine access; optional `client_id_GEE.txt`; `data/processed/corrected/fold/*/annual/*.csv`. | `data/processed/satellite/high_res_multispec/<location>.npy`; `data/processed/satellite/low_res_multispec/<location>.npy`; `data/processed/satellite/manifest.csv` | Main pipeline step |
| `02_inspect_patches.py` | Builds an RGB preview grid from downloaded high/low-resolution multispectral patches. | `data/processed/satellite/high_res_multispec/*.npy`; `data/processed/satellite/low_res_multispec/*.npy`. | `data/processed/satellite/preview_patches.png` | Validation / diagnostic |

## Pipeline position

Run this after [`../03_scripts_calibration`](../03_scripts_calibration/README.md), because patch locations are read from calibrated annual target files. Run `02_inspect_patches.py` after patches have been downloaded.

The next stage is [`../06_models/02_resnet`](../06_models/02_resnet/README.md), which normalizes patches and trains the current ResNet baseline.

## Data flow

Calibrated annual sensor targets -> unique sensor locations -> Sentinel-2 SR annual median composite -> high-resolution local patches and low-resolution context patches -> manifest of downloaded, failed, or corrupted patches -> visual preview.

## Running the scripts

```bash
python 04_GEE/01_download_satellite_patches.py
python 04_GEE/01_download_satellite_patches.py --streams high_res_multispec
python 04_GEE/01_download_satellite_patches.py --streams high_res_multispec low_res_multispec --limit 25
python 04_GEE/01_download_satellite_patches.py --force

python 04_GEE/02_inspect_patches.py
python 04_GEE/02_inspect_patches.py --n 10
python 04_GEE/02_inspect_patches.py --locations 11789 20067
```

## Important assumptions and caveats

- `client_id_GEE.txt` is ignored by git and may contain a service-account JSON key. If absent or not JSON, the script falls back to interactive Earth Engine OAuth.
- The Earth Engine project id is hard-coded as `air-pollution-501614`.
- The year is hard-coded to `2024`; date range is `2024-01-01` through `2024-12-31`.
- The Sentinel-2 source is `COPERNICUS/S2_SR_HARMONIZED` with `CLOUDY_PIXEL_PERCENTAGE < 20`, followed by a median composite.
- The downloaded band order is `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`. Atmospheric bands `B1`, `B9`, and `B10` are excluded.
- High-resolution patches are `120 x 120` pixels over a `1.2 km` footprint. Low-resolution patches are `60 x 60` pixels over a `12 km` footprint.
- Patch grids are requested in EPSG:3857 after converting WGS84 lon/lat to Web Mercator meters.
- Patches with nodata fraction above `0.10` are marked `corrupted` and not saved.
- `02_inspect_patches.py` has stale docstring names for the old scripts, but the current implementation reads outputs from `01_download_satellite_patches.py`.
