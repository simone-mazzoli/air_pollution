# 04 Google Earth Engine Satellite Data

This folder downloads satellite image patches. A patch is a small image window
centered on a sensor or station. The model uses these patches as inputs.

Run commands from the repository root.

## Setup

These scripts require Google Earth Engine access. If `client_id_GEE.txt` exists
at the repository root and contains a service-account JSON key, the scripts use
it. Otherwise they try interactive Earth Engine authentication.

The Earth Engine project id is currently hard-coded in the scripts as
`air-pollution-501614`.

## Main Sentinel-2 Scripts

| Script | What it does | Main output |
| --- | --- | --- |
| `01_download_satellite_patches.py` | Downloads Sentinel-2 patches for Sensor.Community label locations. | `data/processed/satellite/high_res_multispec/*.npy`, `low_res_multispec/*.npy`, `manifest.csv` |
| `02_inspect_patches.py` | Makes a preview image so we can quickly see whether downloaded patches look reasonable. | `data/processed/satellite/preview_patches.png` |
| `05_download_satellite_patches_uba.py` | Downloads Sentinel-2 patches centered on UBA reference stations. | station-centered arrays under `data/processed/satellite/` |

Sentinel-2 is optical satellite imagery. We use it because local land cover,
roads, vegetation, and urban structure can help predict pollution.

## Usual Commands

```bash
python3 04_GEE/01_download_satellite_patches.py --limit 25
python3 04_GEE/01_download_satellite_patches.py
python3 04_GEE/02_inspect_patches.py --n 10
```

Use `--limit` for a small test before starting a full download. A full run can
take a long time and create many `.npy` files.

## Sentinel-5P Scripts

| Script | Status | Notes |
| --- | --- | --- |
| `03_download_s5p_patches.py` | experimental | Downloads S5P patches for sensor locations. Options include `--product`, `--force`, and `--limit`. |
| `04_download_s5p_nation.py` | experimental | Downloads national S5P rasters and crops per-location windows with `--download` and `--crop`. |
| `06_download_s5p_patches_uba.py` | experimental | Downloads S5P patches for UBA stations. |

Sentinel-5P is much coarser than Sentinel-2 but measures atmospheric products
such as NO2. These streams are not fully integrated into the main trainer yet.

## Current Patch Assumptions

- The year is fixed to 2024.
- Sentinel-2 uses `COPERNICUS/S2_SR_HARMONIZED` with a cloud filter and median
  composite.
- High-resolution patches are `120 x 120` pixels over about `1.2 km`.
- Low-resolution patches are `60 x 60` pixels over about `12 km`.
- Patch arrays are generated locally and should not be committed.

The next folder for the current baseline is
[06_models/02_resnet](../06_models/02_resnet/README.md).
