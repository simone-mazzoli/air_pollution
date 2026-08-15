from .config import MODEL

FINAL_MAIN_EXPERIMENTS = ("cnn_deep_wide", "resnet_frozen")
FINAL_MAIN_RUNS = (("cnn_deep", True), ("resnet_frozen", False))
SUPPLEMENTARY_FULL_CV_EXPERIMENTS = ("cnn",)
SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS = ("cnn_deep", "cnn_large", "resnet_full")
REPORT_EXPERIMENTS = (*FINAL_MAIN_EXPERIMENTS, *SUPPLEMENTARY_FULL_CV_EXPERIMENTS, *SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS)
DIAGNOSTIC_EXPERIMENTS = (*SUPPLEMENTARY_FULL_CV_EXPERIMENTS, *SUPPLEMENTARY_DIAGNOSTIC_EXPERIMENTS)
HISTORICAL_EXPERIMENTS = ("resnet", "resnet_layer4")
DEFAULT_CV_EXPERIMENTS = FINAL_MAIN_EXPERIMENTS
SUPPORTED_EXPERIMENTS = ("cnn", "cnn_deep", "cnn_large", "cnn_deep_wide", "resnet_frozen", "resnet_full")
CV_EXPERIMENT_CHOICES = (*SUPPORTED_EXPERIMENTS, "all")
CNN_EXPERIMENTS = {"cnn", "cnn_deep", "cnn_large", "cnn_deep_wide"}
SUMMARY_EXPERIMENTS = FINAL_MAIN_EXPERIMENTS
ALL_RESULT_EXPERIMENTS = (
    "cnn", "cnn_deep", "cnn_deep_wide", "cnn_large",
    "resnet", "resnet_frozen", "resnet_full", "resnet_layer4",
)


def validate_experiment(name, wide=False):
    if name in ("cnn_deep_wide", "cnn_large"):
        if wide:
            raise SystemExit(f"ERROR: {name} already includes --wide")
        return name
    if name not in SUPPORTED_EXPERIMENTS:
        raise SystemExit(f"ERROR: unknown experiment {name}")
    if wide and name not in CNN_EXPERIMENTS:
        raise SystemExit("ERROR: --wide is only valid with --experiment cnn or cnn_deep")
    return name


def expand_cv_experiments(name, wide=False):
    if name == "all":
        if wide:
            raise SystemExit("ERROR: --wide cannot be combined with --experiment all")
        return list(FINAL_MAIN_EXPERIMENTS)
    return [validate_experiment(name, wide)]


def cv_run_plan(name, selected_folds, wide=False):
    if name == "all":
        if wide:
            raise SystemExit("ERROR: --wide cannot be combined with --experiment all")
        return [(experiment, selected_folds, run_wide) for experiment, run_wide in FINAL_MAIN_RUNS]
    if name == "cnn_deep_wide":
        validate_experiment(name, wide)
        return [("cnn_deep", selected_folds, True)]
    if name == "cnn_large":
        validate_experiment(name, wide)
        return [("cnn_large", selected_folds, False)]
    return [(experiment, selected_folds, wide) for experiment in expand_cv_experiments(name, wide)]


def require_single_experiment(name, stage, wide=False):
    if name == "all":
        choices = ", ".join(SUPPORTED_EXPERIMENTS)
        raise SystemExit(f"ERROR: {stage} requires one explicitly selected experiment. Choose one of: {choices}.")
    return validate_experiment(name, wide)


def _resnet_config(mode):
    from resnet.config import RESNET_CONFIG

    cfg = dict(RESNET_CONFIG)
    cfg["experiment"] = f"resnet_{mode}"
    cfg["backbone_mode"] = mode
    return cfg


def _cnn_config(name, wide):
    from cnn.config import BASE_HIGH_CHANNELS, CNN_CONFIG, CNN_DEEP_CONFIG, WIDE_HIGH_CHANNELS

    if name == "cnn_deep_wide":
        name, wide = "cnn_deep", True
    if name == "cnn_large":
        name, wide = "cnn", True
    cfg = dict(CNN_DEEP_CONFIG if name == "cnn_deep" else CNN_CONFIG)
    cfg["wide"] = wide
    cfg["high_channels"] = WIDE_HIGH_CHANNELS if wide else BASE_HIGH_CHANNELS
    if wide:
        cfg["experiment"] = "cnn_deep_wide" if name == "cnn_deep" else "cnn_large"
    return cfg


def selected_model(name=MODEL, wide=False):
    if name == "resnet":
        name = "resnet_frozen"
    if name == "cnn_deep_wide":
        name, wide = "cnn_deep", True
    if name == "cnn_large":
        name, wide = "cnn_large", False
    if name in ("resnet_frozen", "resnet_full"):
        validate_experiment(name, wide)
        from resnet.model import build_model

        return build_model, _resnet_config(name.removeprefix("resnet_"))
    if name in CNN_EXPERIMENTS:
        from cnn.model import build_model

        return build_model, _cnn_config(name, wide)
    raise SystemExit(f"ERROR: unknown model {name}")
