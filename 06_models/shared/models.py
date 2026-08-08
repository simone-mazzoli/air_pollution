from .config import MODEL

SUPPORTED_EXPERIMENTS = ("cnn", "resnet_frozen", "resnet_full")
CV_EXPERIMENT_CHOICES = (*SUPPORTED_EXPERIMENTS, "all")


def expand_cv_experiments(name):
    if name == "all":
        return list(SUPPORTED_EXPERIMENTS)
    if name in SUPPORTED_EXPERIMENTS:
        return [name]
    raise SystemExit(f"ERROR: unknown experiment {name}")


def cv_run_plan(name, selected_folds):
    return [(experiment, selected_folds) for experiment in expand_cv_experiments(name)]


def require_single_experiment(name, stage):
    if name == "all":
        choices = ", ".join(SUPPORTED_EXPERIMENTS)
        raise SystemExit(f"ERROR: {stage} requires one explicitly selected experiment. Choose one of: {choices}.")
    if name not in SUPPORTED_EXPERIMENTS:
        raise SystemExit(f"ERROR: unknown experiment {name}")
    return name


def _resnet_config(mode):
    from resnet.config import RESNET_CONFIG

    cfg = dict(RESNET_CONFIG)
    cfg["experiment"] = f"resnet_{mode}"
    cfg["backbone_mode"] = mode
    return cfg


def selected_model(name=MODEL):
    if name == "resnet":
        name = "resnet_frozen"
    if name in ("resnet_frozen", "resnet_full"):
        from resnet.model import build_model

        return build_model, _resnet_config(name.removeprefix("resnet_"))
    if name == "cnn":
        from cnn.config import CNN_CONFIG
        from cnn.model import build_model

        return build_model, CNN_CONFIG
    raise SystemExit(f"ERROR: unknown model {name}")
