from .config import MODEL


def selected_model(name=MODEL):
    if name == "resnet":
        from resnet.config import RESNET_CONFIG
        from resnet.model import build_model

        return build_model, RESNET_CONFIG
    if name == "cnn":
        from cnn.model import build_model

        return build_model, {"model": "cnn", "experiment": "cnn"}
    raise SystemExit(f"ERROR: unknown model {name}")
