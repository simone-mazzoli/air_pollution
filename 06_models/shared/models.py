from .config import MODEL

DEFAULT_CV_EXPERIMENTS = ("cnn", "resnet_frozen", "resnet_full")
SUPPORTED_EXPERIMENTS = (*DEFAULT_CV_EXPERIMENTS, "cnn_deep")
CV_EXPERIMENT_CHOICES = (*SUPPORTED_EXPERIMENTS, "all")
CNN_EXPERIMENTS = {"cnn", "cnn_deep"}
SUMMARY_EXPERIMENTS = ("cnn", "cnn_wide", "cnn_deep", "cnn_deep_wide", "resnet_frozen", "resnet_full")


def validate_experiment(name, wide=False):
    if name not in SUPPORTED_EXPERIMENTS:
        raise SystemExit(f"ERROR: unknown experiment {name}")
    if wide and name not in CNN_EXPERIMENTS:
        raise SystemExit("ERROR: --wide is only valid with --experiment cnn or cnn_deep")
    return name


def expand_cv_experiments(name, wide=False):
    if name == "all":
        if wide:
            raise SystemExit("ERROR: --wide cannot be combined with --experiment all")
        return list(DEFAULT_CV_EXPERIMENTS)
    return [validate_experiment(name, wide)]


def cv_run_plan(name, selected_folds, wide=False):
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

    cfg = dict(CNN_DEEP_CONFIG if name == "cnn_deep" else CNN_CONFIG)
    cfg["wide"] = wide
    cfg["high_channels"] = WIDE_HIGH_CHANNELS if wide else BASE_HIGH_CHANNELS
    if wide:
        cfg["experiment"] += "_wide"
    return cfg


def selected_model(name=MODEL, wide=False):
    if name == "resnet":
        name = "resnet_frozen"
    if name in ("resnet_frozen", "resnet_full"):
        validate_experiment(name, wide)
        from resnet.model import build_model

        return build_model, _resnet_config(name.removeprefix("resnet_"))
    if name in CNN_EXPERIMENTS:
        from cnn.model import build_model

        return build_model, _cnn_config(name, wide)
    raise SystemExit(f"ERROR: unknown model {name}")
