# Local Data Directory

Large raw and processed data are intentionally not committed to this repository.

This folder is the expected local mount point for data used by the scripts. It
may be a normal directory on a working machine, or it may be replaced locally by
a symlink to external storage. Do not commit a personal absolute symlink.

Expected high-level layout:

```text
data/
  raw/
  processed/
    daily_avg/
    monthly_avg/
    eea/
    satellite_eea/
    satellite_grid/
    calibration/
    admin_boundaries/
```

Main external inputs:

- EEA and UBA reference-station data
- Sensor.Community PM and humidity downloads
- HYRAS weather files for calibration diagnostics
- Google Earth Engine satellite/context patch arrays
- dense-grid patch arrays
- boundary files needed for map regeneration

The repository keeps saved model and analysis results needed to inspect the
final report, including CV results, TEST predictions, dense-grid predictions,
Kreis-level outputs and report figures.
