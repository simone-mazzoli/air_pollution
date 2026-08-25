# Figure Sources

Generated report-facing figures live in `Air_pollution_report/Figures/generated/`.
Static/manual assets remain directly under `Air_pollution_report/Figures/`.

## learning_curves_cnn_deep_wide_objective_loss.png

- script: `06_models/plot_learning_curves.py --experiment cnn_deep_wide`
- inputs: `06_models/results/cnn_deep_wide/cv_history.csv`, `06_models/results/cnn_deep_wide/cv_folds.csv`
- source: `06_models/results/cnn_deep_wide/figures/learning_curves_summary_objective_loss.png`
- report output: `Air_pollution_report/Figures/generated/learning_curves_cnn_deep_wide_objective_loss.png`
- status: regenerated from saved CV history; objective-loss values come from the same CSV inputs

## learning_curves_resnet_frozen_objective_loss.png

- script: `06_models/plot_learning_curves.py --experiment resnet_frozen`
- inputs: `06_models/results/resnet_frozen/cv_history.csv`, `06_models/results/resnet_frozen/cv_folds.csv`
- source: `06_models/results/resnet_frozen/figures/learning_curves_summary_objective_loss.png`
- report output: `Air_pollution_report/Figures/generated/learning_curves_resnet_frozen_objective_loss.png`
- status: regenerated from saved CV history; objective-loss values come from the same CSV inputs

## data_size_rmse_learning_curve.png

- script: `06_models/data_size_ablation/plot_results.py --data-size-only`
- inputs: `06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`
- source: `06_models/data_size_ablation/results/figures/data_size_rmse_learning_curve.png`
- report output: `Air_pollution_report/Figures/generated/data_size_rmse_learning_curve.png`
- status: regenerated from saved ablation summary; plotted values unchanged

## data_size_mae_learning_curve.png

- script: `06_models/data_size_ablation/plot_results.py --data-size-only`
- inputs: `06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`
- source: `06_models/data_size_ablation/results/figures/data_size_mae_learning_curve.png`
- report output: `Air_pollution_report/Figures/generated/data_size_mae_learning_curve.png`
- status: regenerated from saved ablation summary; plotted values unchanged

## cnn_deep_wide_grid_map.png

- script: `07_prediction_analysis/02_predict_grid.py`
- inputs: `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv` plus previously generated map output; grid patch arrays/checkpoint are not present here
- source: `07_prediction_analysis/grid_results/cnn_deep_wide_grid_map.png`
- report output: `Air_pollution_report/Figures/generated/cnn_deep_wide_grid_map.png`
- status: copied existing grid map; prediction CSV is present but grid patch arrays needed to regenerate are absent

## test_residual_map.png

- script: `07_prediction_analysis/01_analyze_test_predictions.py --no-boundary-download`
- inputs: `06_models/results/cnn_deep_wide/test_predictions.csv`, cached boundary GeoJSON files
- source: `07_prediction_analysis/outputs/figures/test_residual_map.png`
- report output: `Air_pollution_report/Figures/generated/test_residual_map.png`
- status: regenerated from saved TEST predictions; plotted values unchanged

## socioeconomic_correlation_matrix.png

- script: `08_kreislevel_data/visualize.py`
- inputs: `08_kreislevel_data/socioeconomic_kreis_2024.csv`
- source: `08_kreislevel_data/figures/socioeconomic/01_correlation_matrix.png`
- report output: `Air_pollution_report/Figures/generated/socioeconomic_correlation_matrix.png`
- status: regenerated from saved socioeconomic table; correlations unchanged

## socioeconomic_indicator_boxplots.png

- script: `08_kreislevel_data/visualize.py`
- inputs: `08_kreislevel_data/socioeconomic_kreis_2024.csv`
- source: `08_kreislevel_data/figures/socioeconomic/02_indicator_boxplots.png`
- report output: `Air_pollution_report/Figures/generated/socioeconomic_indicator_boxplots.png`
- status: regenerated from saved socioeconomic table; plotted values unchanged

## socioeconomic_summary_maps.png

- script: `08_kreislevel_data/03_map_pollution_inequality.py`
- inputs: `08_kreislevel_data/kreis_exposure_socioeconomic.csv`, Kreis boundary geometry
- source: `08_kreislevel_data/kreis_exposure_socioeconomic.csv; missing local Kreis boundary geometry`
- report output: `Air_pollution_report/Figures/generated/socioeconomic_summary_maps.png`
- status: not regenerated: FileNotFoundError: No Kreis GeoJSON and no cached BKG GeoPackage found. Run build_socioeconomic_kreis_2024.py first.

## sensors_map_static.png

- script: `05_scripts_visual/plot_sensors_map_static.py`
- inputs: `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`, Europe boundary GeoJSON
- source: `Air_pollution_report/Figures/sensors_map_static.png`
- report output: `Air_pollution_report/Figures/generated/sensors_map_static.png`
- status: copied existing report figure; source data directory is absent in this checkout

## fold_map.png

- script: `06_models/00_assign_folds.py`
- inputs: `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`, `data/processed/uba/station_land.csv`
- source: `Air_pollution_report/Figures/fold_map.png`
- report output: `Air_pollution_report/Figures/generated/fold_map.png`
- status: copied existing report figure; source data directory is absent in this checkout

## preview_patches.png

- script: `04_GEE patch-preview workflow noted in PIPELINE_OVERVIEW.md`
- inputs: Sentinel-2 patch arrays under `data/processed/satellite_eea/`
- source: `Air_pollution_report/Figures/preview_patches.png`
- report output: `Air_pollution_report/Figures/generated/preview_patches.png`
- status: copied existing report figure; patch arrays are absent in this checkout

## pipeline_overview_original_drawio.jpg

- script: `manual draw.io export`
- inputs: manual pipeline stages
- source: `Air_pollution_report/Figures/graphs_and_plots/drawio.jpg`
- report output: `Air_pollution_report/Figures/generated/pipeline_overview_original_drawio.jpg`
- status: copied for provenance before redesign; not intended as the report-facing replacement

## pipeline_overview_compact.pdf

- script: `09_report_figures/build_report_figures.py`
- inputs: same logical stages as `Air_pollution_report/Figures/graphs_and_plots/drawio.jpg`
- source: `Air_pollution_report/Figures/graphs_and_plots/drawio.jpg`
- report output: `Air_pollution_report/Figures/generated/pipeline_overview_compact.pdf`
- status: redesigned with fixed node coordinates and orthogonal routes; PNG preview also exported

## data_size_ablation_summary.png

- script: `09_report_figures/build_report_figures.py`
- inputs: `06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`
- source: `06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`
- report output: `Air_pollution_report/Figures/generated/data_size_ablation_summary.png`
- status: regenerated from saved ablation summary; same RMSE/MAE values as the separate source figures

## coverage_validation_geography.pdf / coverage_validation_geography.png

- script: `09_report_figures/build_report_figures.py::write_coverage_validation_geography`
- inputs: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv`, `06_models/results/cnn_deep_wide/test_predictions.csv`, `07_prediction_analysis/boundaries/ne_50m_admin_0_countries.geojson`
- source: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv (1869 rows), 06_models/results/cnn_deep_wide/test_predictions.csv (164 rows), 07_prediction_analysis/boundaries/ne_50m_admin_0_countries.geojson`
- report output: `Air_pollution_report/Figures/generated/coverage_validation_geography.pdf`
- preview output: `Air_pollution_report/Figures/generated/coverage_validation_geography.png`
- status: regenerated from numerical station data; CRS EPSG:3035; development stations=1869, TEST stations=164; country extent from station-country polygon union; mixed countries=none

## dense_prediction_residual_maps.png

- script: `09_report_figures/build_report_figures.py::write_dense_prediction_residual_maps`
- inputs: `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv`, `06_models/results/cnn_deep_wide/test_predictions.csv`, `07_prediction_analysis/boundaries/geoboundaries_deu_adm1.geojson`
- source: `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv (1728 rows), 06_models/results/cnn_deep_wide/test_predictions.csv (164 rows), 07_prediction_analysis/boundaries/geoboundaries_deu_adm1.geojson`
- report output: `Air_pollution_report/Figures/generated/dense_prediction_residual_maps.png`
- status: regenerated from numerical prediction data; CRS EPSG:25832; identical extent/projection for both panels

## urban_rural_pm_distribution.png

- script: `09_report_figures/build_report_figures.py`
- inputs: no available CSV contains an explicit urban/rural or station-setting field
- source: `no available CSV contains an explicit urban/rural or station-setting field`
- report output: `Air_pollution_report/Figures/generated/urban_rural_pm_distribution.png`
- status: not produced: explicit urban/rural classification not available

## high_low_pm_patch_examples.png

- script: `09_report_figures/build_report_figures.py`
- inputs: missing data/processed/daily_avg/eea/pm_reference_stations_2024.csv, data/processed/satellite_eea/high_res_multispec, data/processed/satellite_eea/low_res_multispec
- source: `missing data/processed/daily_avg/eea/pm_reference_stations_2024.csv, data/processed/satellite_eea/high_res_multispec, data/processed/satellite_eea/low_res_multispec`
- report output: `Air_pollution_report/Figures/generated/high_low_pm_patch_examples.png`
- status: not produced: model-ready label/patch artifacts unavailable
