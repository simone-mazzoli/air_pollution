# Figure Sources

Generated report figures live in `Air_pollution_report/Figures/generated/`.
Static assets used directly by the report remain in `Air_pollution_report/Figures/`.

## pipeline_overview_compact.pdf

- script: `09_report_figures/build_report_figures.py`
- inputs: project pipeline stages documented in `PIPELINE_OVERVIEW.md`
- output: `Air_pollution_report/Figures/generated/pipeline_overview_compact.pdf`
- note: compact overview figure for the report

## coverage_validation_geography.pdf

- script: `09_report_figures/build_report_figures.py`
- inputs: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv`, `06_models/results/cnn_deep_wide/test_predictions.csv`, `07_prediction_analysis/boundaries/ne_50m_admin_0_countries.geojson`
- output: `Air_pollution_report/Figures/generated/coverage_validation_geography.pdf`
- note: station coverage, development folds and sealed TEST region

## urban_rural_pm25_distribution.png

- script: `09_report_figures/build_preliminary_analysis_figures.py`
- inputs: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv`, `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`, `data/processed/eea/airbase_raw/metadata.csv`
- supporting outputs: `analysis_outputs/preliminary_analysis/urban_rural_pm25_summary_collapsed.csv`, `analysis_outputs/preliminary_analysis/urban_rural_pm25_summary_detailed.csv`
- output: `Air_pollution_report/Figures/generated/urban_rural_pm25_distribution.png`
- note: development stations only, with rural subtypes collapsed to Rural

## high_low_pm25_patch_examples.png

- script: `09_report_figures/build_preliminary_analysis_figures.py`
- inputs: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv`, `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`, `data/processed/eea/airbase_raw/metadata.csv`, `data/processed/satellite_eea/high_res_multispec/`
- supporting outputs: `analysis_outputs/preliminary_analysis/high_low_pm25_selected_stations.csv`, `analysis_outputs/preliminary_analysis/high_low_pm25_patch_quality_checks.csv`, `analysis_outputs/preliminary_analysis/high_low_pm25_image_scaling.csv`
- output: `Air_pollution_report/Figures/generated/high_low_pm25_patch_examples.png`
- note: three lowest and three highest German development stations with at least 330 valid PM2.5 days and usable high-resolution Sentinel-2 data

## high_low_pm25_patch_examples_lowres.png

- script: `09_report_figures/build_preliminary_analysis_figures.py`
- inputs: same selected stations with low-resolution Sentinel-2 arrays under `data/processed/satellite_eea/low_res_multispec/`
- output: `Air_pollution_report/Figures/generated/high_low_pm25_patch_examples_lowres.png`
- note: wider-context companion figure for the appendix

## data_size_ablation_summary.png

- script: `09_report_figures/build_report_figures.py`
- inputs: `06_models/data_size_ablation/results/data_size_summary_by_fraction.csv`
- output: `Air_pollution_report/Figures/generated/data_size_ablation_summary.png`
- note: combines the saved RMSE and MAE data-size summaries

## dense_prediction_residual_maps.png

- script: `09_report_figures/build_report_figures.py`
- inputs: `07_prediction_analysis/grid_results/cnn_deep_wide_grid_predictions.csv`, `06_models/results/cnn_deep_wide/test_predictions.csv`, `07_prediction_analysis/boundaries/geoboundaries_deu_adm1.geojson`
- output: `Air_pollution_report/Figures/generated/dense_prediction_residual_maps.png`
- note: dense-grid predictions and TEST residuals shown together

## learning_curves_cnn_deep_wide_objective_loss.png

- script: `06_models/plot_learning_curves.py --experiment cnn_deep_wide`
- inputs: `06_models/results/cnn_deep_wide/cv_history.csv`, `06_models/results/cnn_deep_wide/cv_folds.csv`
- output: `Air_pollution_report/Figures/generated/learning_curves_cnn_deep_wide_objective_loss.png`
- note: training diagnostic from the later reproducibility run

## learning_curves_resnet_frozen_objective_loss.png

- script: `06_models/plot_learning_curves.py --experiment resnet_frozen`
- inputs: `06_models/results/resnet_frozen/cv_history.csv`, `06_models/results/resnet_frozen/cv_folds.csv`
- output: `Air_pollution_report/Figures/generated/learning_curves_resnet_frozen_objective_loss.png`
- note: training diagnostic from the later reproducibility run

## socioeconomic_indicator_boxplots.png

- script: `08_kreislevel_data/visualize.py`
- inputs: `08_kreislevel_data/socioeconomic_kreis_2024.csv`
- output: `Air_pollution_report/Figures/generated/socioeconomic_indicator_boxplots.png`

## socioeconomic_correlation_matrix.png

- script: `08_kreislevel_data/visualize.py`
- inputs: `08_kreislevel_data/socioeconomic_kreis_2024.csv`
- output: `Air_pollution_report/Figures/generated/socioeconomic_correlation_matrix.png`

## socioeconomic_summary_maps.png

- script: `08_kreislevel_data/03_map_pollution_inequality.py`
- inputs: `08_kreislevel_data/kreis_exposure_socioeconomic.csv` and Kreis boundary geometry
- output: `Air_pollution_report/Figures/generated/socioeconomic_summary_maps.png`
- note: current PNG is kept because the local Kreis geometry needed to regenerate it is not included
