# Report Layout Plan

Goal: keep the main report plausibly within 10 pages without shrinking fonts or rewriting scientific prose. Structural LaTeX moves are deferred until teammate Overleaf edits are merged.

## Main-Report Target

- Keep: pipeline overview, EEA coverage + fold geography as one combined figure, calibration summary, pooled model comparison, data-size ablation, TEST summary, dense prediction + residuals as one combined figure, compact socioeconomic correlation result.
- Add when data return: high/low PM patch examples and urban/rural PM distribution.
- Move to appendix: detailed model architecture/hyperparameter tables, per-fold CV tables, TEST-by-state table, socioeconomic diagnostics, full socioeconomic map, generic sensor photo.
- Replace later: generic patch preview once the assignment-required high/low PM patch figure is reproducible.

## Figure/Table Classification

| Current no. | Item | Current section | Class | Reason | Approx. space | Assignment-required? | Role |
|---|---|---|---|---|---:|---|---|
| Figure 1 | Pipeline overview | Introduction / Pipeline Overview | MAIN | Useful orientation, but use compact redesigned version. | 0.25-0.35 page | No | Detail/method overview |
| Figure 2 | EEA station coverage map | Data Collection / PM Data | MAIN | Establishes reference-station coverage and sparse regions. Combine with Figure 3. | 0.25 page combined | Yes, supports data description | Central setup |
| Figure 3 | Geographic fold/test-region map | Data Collection / PM Data | MAIN | Explains spatial CV and held-out region. Combine with Figure 2. | 0.25 page combined | Yes, supports validation design | Central method |
| Figure 4 | Generic Sentinel-2 patch preview | Data Collection / Satellite Data | APPENDIX / REPLACE WHEN DATA AVAILABLE | Current generic examples do not satisfy high-vs-low PM requirement. Keep for provenance/appendix only until reproducible replacement exists. | 0.30 page | Yes, but not current form | Detail/placeholder |
| Figure 5 | Socioeconomic indicator distributions | Socio-Economic Data | APPENDIX | Useful data diagnostic, not needed for the 10-page main argument. | 0.45 page | No | Detail/diagnostic |
| Figure 6 | Socioeconomic indicator correlation matrix | Socio-Economic Data | APPENDIX | Supports interpretation but competes with final PM-socioeconomic result. | 0.35 page | No | Detail/diagnostic |
| Unnumbered table | Sensor.Community calibration summary | Calibration of LC PM sensors | MAIN | Justifies using EEA labels instead of low-cost sensors. Should become a compact numbered table. | 0.30 page | Yes, supports sensor-validity task | Central method/result |
| Table 1 | Scratch CNN architecture | Models / CNN created from scratch | APPENDIX | Full architecture detail is too large for main report. Summarize in prose/main comparison. | 0.75-1.0 page | No | Detail |
| Table 2 | Scratch CNN hyperparameters | Models / CNN created from scratch | APPENDIX | Reproducibility detail, not central narrative. | 0.35 page | No | Detail |
| Table 3 | Frozen ResNet architecture | Models / Fine-tuned Model | APPENDIX | Reproducibility detail, too large for main report. | 0.35-0.45 page | No | Detail |
| Table 4 | Frozen ResNet hyperparameters | Models / Fine-tuned Model | APPENDIX | Reproducibility detail, compact but noncentral. | 0.20 page | No | Detail |
| Table 5 | Scratch CNN per-fold CV | Results / Training | APPENDIX | Pooled comparison is the central result; fold table is diagnostic. | 0.45 page | No | Detail/diagnostic |
| Table 6 | ResNet per-fold CV | Results / Training | APPENDIX | Same as Table 5. | 0.45 page | No | Detail/diagnostic |
| Figure 7 | ResNet validation learning curves | Results / Training | APPENDIX | Large diagnostic. Keep out of 10-page main unless prose explicitly depends on it. | 0.65 page | No | Diagnostic |
| Figure 8 | Data-size ablation RMSE + MAE | Results / Training | MAIN | Compact two-panel figure supports data-efficiency comparison. | 0.35-0.45 page | No | Central secondary result |
| Table 7 | Pooled scratch-vs-ResNet CV comparison | Results / Evaluation | MAIN | Primary model-selection evidence. | 0.20 page | No | Central result |
| Table 8 | Final sealed TEST summary | Results / Evaluation | MAIN | Primary held-out evaluation result. | 0.30 page | No | Central result |
| Table 9 | TEST performance by federal state | Results / Evaluation | APPENDIX | Important caveat/detail, but too granular for 10-page main. Summarize in prose. | 0.45 page | No | Detail/diagnostic |
| Figure 9 | Dense PM prediction map | Results / Dense Grid Map | MAIN | Central deployment output. Combine with Figure 10. | 0.45-0.55 page combined | Yes, supports dense prediction task | Central result |
| Figure 10 | TEST residual map | Results / Dense Grid Map | MAIN | Necessary reliability companion to Figure 9. Combine with Figure 9. | included above | No | Central caveat |
| Figure 11 | Full socioeconomic summary maps | Socio-Economic Analysis / Spatial patterns | APPENDIX / REGENERATE WHEN KREIS GEOMETRY AVAILABLE | Eight-panel map is informative but too large; main can use correlation table and concise text. Current raster is preserved because local Kreis geometry is absent, so it still needs regeneration from `08_kreislevel_data/03_map_pollution_inequality.py` once `vg250_kreis.geojson` or the cached BKG GeoPackage is available. | 0.75-1.0 page | No | Detail |
| Table 10 | PM2.5-socioeconomic correlations | Socio-Economic Analysis / Association | MAIN | Most compact socioeconomic result. | 0.30 page | Yes, supports socioeconomic analysis | Central result |
| Appendix figure | SDS011 sensor photo | Appendix / Additional Graphs and Images | REMOVE | Generic image adds little scientific value under page pressure. Keep file in repo if desired. | 0.15 page | No | Decorative/detail |

## Deferred Mechanical Moves

- Done in the active report: Figure 1 uses `Figures/generated/pipeline_overview_compact.pdf`.
- Done in the active report: Figures 2 and 3 use the combined `Figures/generated/coverage_validation_geography.pdf`.
- Done in the active report: the two data-size subfigures use `Figures/generated/data_size_ablation_summary.png`.
- Done in the active report: Figures 9 and 10 use `Figures/generated/dense_prediction_residual_maps.png`.
- Move appendix-classified tables/figures after `\appendix` after teammate Overleaf prose edits are merged.
- Move the current learning-curve block at `Air_pollution_report/main_merged.tex` lines 611-617 to the appendix after the surrounding Results prose is restructured; both canonical objective-loss images are already prepared in `Figures/generated/`.
- Regenerate `Figures/generated/socioeconomic_summary_maps.png` from `08_kreislevel_data/03_map_pollution_inequality.py` once Kreis geometry is available locally.
- Replace Figure 4 with a reproducible high/low PM patch figure when JupyterHub data are available.
