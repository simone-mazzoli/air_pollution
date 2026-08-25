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

## Main Scripts
| Script | What it does | Main output |
| --- | --- | --- |
| `01_download_eea_patches.py` | Downloads Sentinel-2 (high-res local + low-res wide-context), Sentinel-5P (NO2, aerosol index, CO, plus a wide-context aerosol stream), and a static Copernicus DEM elevation patch, per EEA station across Europe. Streams are independently selectable, resumable, and filtered for cloud cover/nodata. | `data/processed/satellite_eea/<stream_folder>/<station_code>.npy`, `manifest_<stream>.csv` |
| `02_download_dense_grid_patches.py` | Builds a 10 km regular grid over the sealed-test Laender (points kept only where they fall inside the actual Land polygons, from FAO/GAUL boundaries) and downloads the same stream types for continuous pollution-map inference. Reuses `01_download_eea_patches.py`'s fetch logic directly rather than reimplementing it. | `data/processed/satellite_grid/<stream_folder>/<grid_id>.npy`, `data/processed/eea/grid_points.csv` |

## Usual Commands
```bash
python3 04_GEE/01_download_eea_patches.py --stream high_res low_res no2 aer co
python3 04_GEE/02_download_dense_grid_patches.py --stream high_res low_res no2 co aer_wide dem
```
`01_download_eea_patches.py` also takes `--country` (restrict to specific
countries) and `--limit` (first N stations, for testing). Both scripts skip
already-downloaded patches unless `--force`.

## Data Folder
These scripts write into `data/processed/satellite_eea` and `data/processed/satellite_grid` relative to the repository. If your data are on an external drive, make `data/` a symlink or
mount/link that points there. Do not commit the downloaded ZIPs or generated
processed tables.


## Current Patch Assumptions
- The year is fixed to 2024.
- Bounding box is EU-wide (covers all EEA-reporting countries)
- Sentinel-2 uses `COPERNICUS/S2_SR_HARMONIZED` with a cloud filter
  (`CLOUDY_PIXEL_PERCENTAGE < 20`) and annual median composite.
- `high_res`: `120 x 120` px @ 10 m/px (1.2 km). `low_res`: `60 x 60` px @
  200 m/px (12 km).
- `no2`/`aer`/`co`: `5 x 5` px @ 7 km/px (35 km). `aer_wide`: `31 x 31` px @
  7 km/px (217 km).
- `dem`: static Copernicus GLO-30 mosaic, `60 x 60` px @ 200 m/px (12 km), no
  date filter.
- Patch arrays are generated locally and should not be committed.


