# 05 Visualization Checks

This folder has optional maps for checking where sensors and UBA stations are.
They are useful for sanity checks but are not part of the model pipeline.

Run commands from the repository root.

```bash
python3 05_scripts_visual/plot_sensors_map_leaflet.py
python3 05_scripts_visual/plot_sensors_map_static.py
```

Both scripts currently use `MONTH = "2024-01"` inside the file instead of a CLI
argument.

## Outputs

| Script | Output |
| --- | --- |
| `plot_sensors_map_leaflet.py` | `data/processed/plots/sensors_map_leaflet_<MONTH>.html` |
| `plot_sensors_map_static.py` | `data/processed/plots/sensors_map_static_<MONTH>.png` |

## Current Caveat

These scripts still expect the older PM monthly filename:

```text
data/processed/monthly_avg/all_pm_sensors/germany_monthly_avg_ALLPM_<MONTH>.csv
```

The current PM cleaning script writes:

```text
data/processed/monthly_avg/all_pm_sensors/<YYYY-MM>.csv
```

So these map scripts probably need a small path update before they run on the
current cleaned outputs.
