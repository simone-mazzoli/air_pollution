from pathlib import Path

MODELS = Path(__file__).resolve().parents[1]
REPO = MODELS.parent
PROC = REPO / "data" / "processed"
SAT = PROC / "satellite_eea"

LABELS = PROC / "daily_avg" / "eea" / "pm_reference_stations_2024.csv"
STATION_LAND = PROC / "uba" / "station_land.csv"
STATION_FOLD = PROC / "eea" / "station_fold.csv"
FOLD_MAP = PROC / "eea" / "fold_map.png"

HIGH = SAT / "high_res_multispec"
LOW = SAT / "low_res_multispec"
AERW = SAT / "aer_wide_tropomi"
DEMD = SAT / "dem_glo30"

RESULTS = MODELS / "results"
RESULTS_RESNET = RESULTS / "resnet"
RESULTS_CNN = RESULTS / "cnn"

