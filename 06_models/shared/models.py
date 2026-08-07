from .config import MODEL

SUPPORTED_EXPERIMENTS = ("resnet_frozen", "resnet_layer4", "cnn")


def _resnet_config(mode):
    from resnet.config import RESNET_CONFIG

    cfg = dict(RESNET_CONFIG)
    cfg["experiment"] = f"resnet_{mode}"
    cfg["backbone_mode"] = mode
    return cfg


def selected_model(name=MODEL):
    if name == "resnet":
        name = "resnet_frozen"
    if name in ("resnet_frozen", "resnet_layer4"):
        from resnet.model import build_model

        return build_model, _resnet_config(name.removeprefix("resnet_"))
    if name == "cnn":
        from cnn.config import CNN_CONFIG
        from cnn.model import build_model

        return build_model, CNN_CONFIG
    raise SystemExit(f"ERROR: unknown model {name}")
