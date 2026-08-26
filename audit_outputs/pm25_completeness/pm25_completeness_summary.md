# PM2.5 completeness audit

## Status

The redownloaded `air_pollution_data.tar.gz` is readable. `gzip -t air_pollution_data.tar.gz` completed with no output. The audit could read the daily EEA label file, the EEA station metadata and the satellite arrays from the archive.

## Checks completed

1. Daily EEA file used: `data/processed/daily_avg/eea/pm_reference_stations_2024.csv`.
2. Archive check: the archive contains 42,798 members. The last member scanned was `data/raw/socioeconomic_auto/zensus2022/regionaltabelle_bildung.url.txt`.
3. Development stations: `06_models/results/cnn_deep_wide/eea_cv_predictions.csv` has 1,869 rows and 1,869 unique station codes.
4. TEST stations: `06_models/results/cnn_deep_wide/test_predictions.csv` has 164 rows and 164 unique station codes.
5. Model station union: 2,033 station rows. The development and TEST station codes do not overlap.
6. PM2.5 valid-day distribution: development min 6, p25 354, median 363, mean 352.7, p75 366 and max 366. TEST min 151, p25 364, median 366, mean 356.0, p75 366 and max 366.
7. Counts below the 329-day 90 percent threshold: 108 of 2,033 model stations. This is 96 of 1,869 development stations and 12 of 164 TEST stations.
8. PM10-OR issue count: 108 model stations have fewer than 329 PM2.5 days but at least 329 PM10 days. This means all below-threshold PM2.5 cases entered through the PM10 side of the preprocessing OR rule.
9. Annual means sample match: the recomputed annual PM2.5 means from the daily file match the saved model targets. The largest absolute difference is 0.0000019.
10. Severity classification: moderate. The issue did matter for station inclusion because 5.3 percent of all model stations and 7.3 percent of TEST stations would fail a PM2.5-specific 90 percent day-count rule. It does not look like a target recomputation error.
11. Exact report wording: use this wording if the report text is updated later: "During EEA preprocessing, a station was retained when either PM10 or PM2.5 met the 90 percent completeness threshold. The final PM2.5 model then used stations with a non-missing PM2.5 annual mean and complete satellite/context inputs. In the saved PM2.5 model set, 108 of 2,033 stations had fewer than 329 PM2.5 days, including 12 of 164 held-out TEST stations, because they passed the preprocessing threshold through PM10 completeness. The resulting annual PM2.5 labels match the daily-file means, so this affects completeness strictness rather than target calculation."
12. Urban/rural availability: available in `data/processed/eea/airbase_raw/metadata.csv` as `Air Quality Station Area`. All 2,033 model stations matched to an area value.
13. Urban/rural PM distribution: urban stations have higher mean PM2.5 than rural groups in both splits. Development urban mean is 12.09 and rural mean is 8.58. TEST urban mean is 9.18 and rural mean is 7.82.
14. High-vs-low PM2.5 patch example inputs: available. The 10 lowest and 10 highest saved PM2.5 stations all have arrays for 7 satellite/context inputs: `aer_tropomi`, `aer_wide_tropomi`, `co_tropomi`, `dem_glo30`, `high_res_multispec`, `low_res_multispec` and `no2_tropomi`.
15. Newly available data: daily EEA labels, EEA station metadata, satellite arrays and socioeconomic raw files are now readable from the archive.

## Files created or updated

- `audit_outputs/pm25_completeness/pm25_completeness_summary.md`
- `audit_outputs/pm25_completeness/pm25_completeness_by_station.csv`
- `audit_outputs/pm25_completeness/pm25_completeness_split_summary.csv`
- `audit_outputs/pm25_completeness/pm25_valid_day_bins.csv`
- `audit_outputs/pm25_completeness/pm10_or_only_model_stations.csv`
- `audit_outputs/pm25_completeness/urban_rural_pm25_summary.csv`
- `audit_outputs/pm25_completeness/high_low_pm25_patch_availability.csv`

## Git status note

The archive `air_pollution_data.tar.gz` and `audit_outputs/` are untracked. There are also existing modified and untracked repository files from earlier cleanup work. This audit did not retrain models, rerun CV, run TEST inference, modify preprocessing, modify model code, modify the report, regenerate patches, commit or push.
