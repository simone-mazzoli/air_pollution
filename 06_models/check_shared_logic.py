import pandas as pd
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from torch.utils.data import TensorDataset

from shared import data, evaluation, folds, training
from shared.config import NUM_WORKERS, result_paths
from resnet.config import RESNET_CONFIG
from resnet.model import build_model


def resnet_cfg(mode):
    cfg = dict(RESNET_CONFIG)
    cfg.update({
        "experiment": f"resnet_{mode}",
        "backbone_mode": mode,
        "pretrained": False,
        "use_aer_wide": True,
        "use_dem": True,
        "s5p_streams": ["no2", "co"],
        "pollutants": ["pm25"],
    })
    return cfg


def backbone_bn_modules(model):
    return [m for m in model.backbone.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def dummy_forward(model):
    xh = torch.zeros(2, 10, 120, 120)
    xl = torch.zeros(2, 10, 60, 60)
    xs_patch = torch.zeros(2, 2, 5, 5)
    xw = torch.zeros(2, 1, 31, 31)
    xd = torch.zeros(2, 1, 60, 60)
    xs_mean = torch.zeros(2, 4)
    assert tuple(model(xh, xl, xs_patch, xw, xd, xs_mean).shape) == (2, 1)


def main():
    sf = pd.DataFrame({"fold": ["TEST", "fold2_france", "UNASSIGNED", "fold1_iberia"]})
    assert folds.development_fold_names(sf) == ["fold1_iberia", "fold2_france"]

    train = pd.DataFrame({
        "station_code": ["near", "far"],
        "lat": [0.0, 2.0],
        "lon": [0.1, 0.0],
    })
    val = pd.DataFrame({"station_code": ["heldout"], "lat": [0.0], "lon": [0.0]})
    kept = data.buffer_exclude(train, val, 100.0)
    assert kept["station_code"].tolist() == ["far"]

    cfg = {"batch": 4, "num_workers": NUM_WORKERS}
    for n, batches, drop_last in [(3, 1, False), (8, 2, False), (9, 2, True), (10, 3, False)]:
        loader, info = training.train_loader(TensorDataset(torch.arange(n)), cfg)
        assert len(loader) == batches
        assert loader.num_workers == NUM_WORKERS
        assert info == {"n_train_samples": n, "n_train_batches": batches, "effective_drop_last": drop_last}
    try:
        training.train_loader(TensorDataset(torch.arange(1)), cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("singleton training set should fail")

    base_cfg = {"pollutants": ["pm25"]}
    train_df = pd.DataFrame({"pm25": [1.0, 9.0]})
    eval_df = pd.DataFrame({"pm25": [5.0]})
    baseline = evaluation.constant_baseline(train_df, eval_df, base_cfg)
    assert baseline["pm25"]["mean"] == 5.0
    assert baseline["pm25"]["rmse"] == 0.0

    preds = evaluation.prediction_table(
        pd.DataFrame({
            "station_code": ["a", "b"],
            "country": ["DE", "FR"],
            "land": ["Berlin", ""],
            "lat": [1.0, 2.0],
            "lon": [3.0, 4.0],
            "fold": ["TEST", "fold2_france"],
            "pm25": [7.0, float("nan")],
        }),
        {"pm25": ([6.5], [7.0])},
        base_cfg,
    )[0]
    assert preds.loc[0, "station_code"] == "a"
    assert preds.loc[0, "land"] == "Berlin"

    assert result_paths("resnet_frozen")["cv_results"] != result_paths("resnet_layer4")["cv_results"]
    assert result_paths("resnet_frozen")["cv_results"] != result_paths("cnn")["cv_results"]
    with TemporaryDirectory() as td:
        cv_path = Path(td) / "cv.json"
        cv_path.write_text('{"fold1_iberia": {"best_epoch": 3}, "fold2_france": {"best_epoch": 4}}')
        epochs, best_epochs, rule = training.final_epochs_from_cv(cv_path, ["fold1_iberia", "fold2_france"])
        assert epochs == 4
        assert best_epochs == [3, 4]
        assert rule == "median_cv_best_epoch_ceil"

    model_cfg = resnet_cfg("frozen")
    model = build_model(2, model_cfg, 1)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.train()
    assert not model.backbone.training
    assert all(not bn.training for bn in backbone_bn_modules(model))
    assert any(p.requires_grad for n, p in model.named_parameters() if not n.startswith("backbone."))
    trainable = training.trainable_parameters(model)
    expected_trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(expected_trainable)
    assert all(a is b for a, b in zip(trainable, expected_trainable))
    counts = training.parameter_counts(model)
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts == {"total": 23766049, "trainable": 236065, "frozen": 23529984}
    frozen_groups = training.optimizer_parameter_groups(model, model_cfg)
    assert len(frozen_groups) == 1
    assert frozen_groups[0]["lr"] == model_cfg["lr_head"]
    assert training.model_metadata(model, model_cfg)["experiment"] == "resnet_frozen"
    dummy_forward(model)

    layer4_cfg = resnet_cfg("layer4")
    layer4_model = build_model(2, layer4_cfg, 1)
    for name, p in layer4_model.backbone.named_parameters():
        if not name.startswith("layer4."):
            assert not p.requires_grad
    assert any(p.requires_grad for p in layer4_model.backbone.layer4.parameters())
    assert all(not p.requires_grad for bn in backbone_bn_modules(layer4_model) for p in bn.parameters())
    layer4_model.train()
    assert all(not bn.training for bn in backbone_bn_modules(layer4_model))
    assert any(p.requires_grad for n, p in layer4_model.named_parameters() if not n.startswith("backbone."))
    groups = training.optimizer_parameter_groups(layer4_model, layer4_cfg)
    assert [g["lr"] for g in groups] == [layer4_cfg["lr_head"], layer4_cfg["lr_layer4"]]
    grouped = [p for g in groups for p in g["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in layer4_model.parameters() if p.requires_grad}
    layer4_ids = {id(p) for p in layer4_model.layer4_trainable_parameters()}
    assert layer4_ids == {id(p) for p in groups[1]["params"]}
    assert not any(id(p) in layer4_ids for p in groups[0]["params"])
    layer4_meta = training.model_metadata(layer4_model, layer4_cfg)
    assert layer4_meta["experiment"] == "resnet_layer4"
    assert layer4_meta["backbone_mode"] == "layer4"
    assert layer4_meta["trainable_layer4_parameters"] == sum(p.numel() for p in groups[1]["params"])
    dummy_forward(layer4_model)


if __name__ == "__main__":
    main()
