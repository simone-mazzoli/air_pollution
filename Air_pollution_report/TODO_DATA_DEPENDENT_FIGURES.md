# Data-Dependent Figure TODOs

These figures are required by the assignment but cannot be produced reproducibly from the current local checkout. Do not invent substitutes or infer classifications from density thresholds without explicit approval.

## Urban/Rural PM Distribution

Needed:

- Station-level annual PM data.
- A defensible station urban/rural classification.

Preferred figure once data are available:

- Seaborn boxplot or violin plot with lightly jittered station points.
- PM$_{2.5}$ by urban/rural class.
- Shared report style from `report_plot_style.py`.
- Units shown as `µg/m³`.

Do not invent urban/rural labels from station density, land cover, or administrative labels unless that rule is explicitly approved later.

## High/Low PM Patch Examples

Needed:

- Model-ready annual PM$_{2.5}$ station table.
- Corresponding Sentinel-2 patch arrays.

Preferred reproducible selection:

- Select stations from predefined upper/lower PM$_{2.5}$ quantiles.
- Do not cherry-pick by image appearance.

Preferred output:

- Matched low/high PM examples.
- Annual observed PM$_{2.5}$ printed in each panel.
- High-resolution patch and, if space allows, wider-context patch.
- Same RGB rendering pipeline as the existing patch preview.

Do not alter the report to imply these requirements have already been satisfied.
