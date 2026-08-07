import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from torch.utils.data import TensorDataset

from shared import data, evaluation, experiment, folds, paths, runtime, summary, training
from shared.config import CPU_INTEROP_THREADS, CPU_THREADS, NUM_WORKERS, result_paths
from shared.models import SUPPORTED_EXPERIMENTS, selected_model
from cnn.config import CNN_CONFIG
from cnn.model import ScratchHighEncoder, build_model as build_cnn_model
from resnet.config import RESNET_CONFIG
from resnet.model import build_model as build_resnet_model


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


def cnn_cfg():
    cfg = dict(CNN_CONFIG)
    cfg.update({
        "use_aer_wide": True,
        "use_dem": True,
        "s5p_streams": ["no2", "co"],
        "pollutants": ["pm25"],
    })
    return cfg


def check_scratch_cnn():
    encoder = ScratchHighEncoder()
    assert tuple(encoder(torch.zeros(2, 10, 120, 120)).shape) == (2, 256)
    assert all(p.requires_grad for p in encoder.parameters())

    cfg = cnn_cfg()
    model = build_cnn_model(2, cfg, 1)
    dummy_forward(model)
    assert all(p.requires_grad for p in model.parameters())
    counts = training.parameter_counts(model)
    assert counts["total"] == counts["trainable"]
    assert counts["frozen"] == 0
    meta = training.model_metadata(model, cfg)
    assert meta["model"] == "cnn"
    assert meta["experiment"] == "cnn"
    assert meta["high_encoder_feature_dim"] == 256
    assert meta["trainable_high_encoder_parameters"] == sum(p.numel() for p in model.backbone.parameters())


def check_experiment_artifacts():
    result_dirs = [result_paths(name)["dir"] for name in SUPPORTED_EXPERIMENTS]
    assert len(set(result_dirs)) == len(SUPPORTED_EXPERIMENTS)
    for name in SUPPORTED_EXPERIMENTS:
        _, cfg = selected_model(name)
        assert cfg["experiment"] == name

    with TemporaryDirectory() as td:
        root = Path(td)
        row = experiment.epoch_history_row(
            {"experiment": "cnn", "pollutants": ["pm25"]},
            "fold1_iberia",
            1,
            0.5,
            {"pm25": {"rmse": 1.0, "mae": 0.8, "r2": 0.1, "n": 2}},
            {"pm25": {"rmse": 1.2, "mae": 0.9, "r2": -0.1, "n": 2}},
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-5),
            {"train_seconds": 1.0, "train_eval_seconds": 0.2, "val_seconds": 0.3, "total_seconds": 1.5},
            True,
            0,
        )
        history_path = root / "cv_history.csv"
        experiment.append_csv_row(history_path, row)
        history = pd.read_csv(history_path)
        assert history.loc[0, "experiment"] == "cnn"
        assert history.loc[0, "val_pm25_rmse"] == 1.2

        fold_result = {
            "n_train": 10,
            "n_val": 2,
            "best_epoch": 1,
            "buffer_removed_train_stations": 3,
            "n_train_before_buffer": 13,
            "parameter_counts": {"total": 7, "trainable": 7, "frozen": 0},
            "pm25": {"rmse": 1.2, "mae": 0.9, "r2": -0.1, "n": 2, "baseline": 1.5},
        }
        fold_path = root / "cv_folds.csv"
        for fold_row in experiment.fold_summary_rows({"experiment": "cnn", "pollutants": ["pm25"]},
                                                     "fold1_iberia", fold_result):
            experiment.append_csv_row(fold_path, fold_row)
        folds_csv = pd.read_csv(fold_path)
        assert folds_csv.loc[0, "n_buffer_dropped"] == 3

        meta = experiment.run_metadata(
            {"experiment": "cnn", "model": "cnn", "pollutants": ["pm25"], "buffer_km": 100.0},
            parameter_counts={"total": 7, "trainable": 7, "frozen": 0},
            model_metadata={"high_encoder": "scratch", "trainable_high_encoder_parameters": 4},
            started_at="start",
        )
        assert meta["experiment"] == "cnn"
        assert meta["parameter_counts"]["total"] == 7
        assert meta["parameter_counts"]["trainable_high_encoder_parameters"] == 4
        assert meta["model_metadata"]["high_encoder"] == "scratch"


def write_fake_result(name, pollutant="pm25"):
    result = result_paths(name)
    result["dir"].mkdir(parents=True, exist_ok=True)
    result["cv_results"].write_text(
        '{"pooled_out_of_fold": {"pm25": {"rmse": 1.0, "mae": 0.8, "r2": 0.2, "n": 5}}}'
    )
    pd.DataFrame([{
        "experiment": name,
        "fold": "fold1_iberia",
        "pollutant": pollutant,
        "n_train": 10,
        "n_val": 5,
        "n_buffer_dropped": 1,
        "best_epoch": 2,
        "rmse": 1.1,
        "mae": 0.9,
        "r2": 0.1,
        "n": 5,
        "baseline_rmse": 1.4,
        "total_parameters": 100,
        "trainable_parameters": 80,
        "frozen_parameters": 20,
    }]).to_csv(result["cv_folds"], index=False)


def check_summary_script_logic():
    old_results = paths.RESULTS
    with TemporaryDirectory() as td:
        paths.RESULTS = Path(td)
        try:
            write_fake_result("cnn")
            partial = summary.summarize_results()
            assert partial["available"] == ["cnn"]
            assert "resnet_frozen" in partial["missing"]
            assert len(partial["comparison"]) == 1

            write_fake_result("resnet_frozen")
            write_fake_result("resnet_layer4")
            complete = summary.summarize_results()
            assert sorted(set(complete["available"])) == sorted(SUPPORTED_EXPERIMENTS)
            assert complete["missing"] == []
            assert (paths.RESULTS / "summary" / "experiment_comparison.csv").exists()
            assert (paths.RESULTS / "summary" / "fold_comparison.csv").exists()
        finally:
            paths.RESULTS = old_results


def check_patch_cache():
    with TemporaryDirectory() as td:
        patch_dir = Path(td) / "high"
        patch_dir.mkdir()
        path = patch_dir / "station_a.npy"
        raw = np.arange(40, dtype="float32").reshape(2, 2, 10) + 1
        np.save(path, raw)

        data.clear_patch_cache()
        uncached = data.load_s2_station(patch_dir, "high", "station_a", cache_patches=False)
        first = data.load_s2_station(patch_dir, "high", "station_a", cache_patches=True)
        first[...] = 999.0
        original_resolve = Path.resolve
        original_load = np.load
        try:
            Path.resolve = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache hit must not resolve paths")
            )
            np.load = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache hit must not load files")
            )
            second = data.load_s2_station(patch_dir, "high", "station_a", cache_patches=True)
        finally:
            Path.resolve = original_resolve
            np.load = original_load
        assert np.allclose(second, uncached)
        assert data.patch_cache_stats()["misses"] == 1
        assert data.patch_cache_stats()["hits"] == 1

        raw_dir = Path(td) / "raw"
        raw_dir.mkdir()
        raw_path = raw_dir / "station_a.npy"
        raw_patch = np.arange(9, dtype="float32").reshape(3, 3, 1)
        np.save(raw_path, raw_patch)
        uncached_raw = data.load_patch_raw_station(raw_dir, "raw", "station_a", cache_patches=False)
        cached_raw = data.load_patch_raw_station(raw_dir, "raw", "station_a", cache_patches=True)
        cached_raw[...] = -1.0
        assert np.allclose(
            data.load_patch_raw_station(raw_dir, "raw", "station_a", cache_patches=True),
            uncached_raw,
        )

    data.clear_patch_cache()


def check_configured_pollutant_filtering():
    frame = pd.DataFrame({
        "station_code": ["pm10_only", "pm25_only", "both", "neither"],
        "pm10": [12.0, np.nan, 13.0, np.nan],
        "pm25": [np.nan, 7.0, 8.0, np.nan],
    })
    cfg = {"max_pm10": 120.0, "max_pm25": 80.0}

    pm25 = data._apply_label_filters(frame.copy(), {**cfg, "pollutants": ["pm25"]})
    assert pm25["station_code"].tolist() == ["pm25_only", "both"]

    pm10 = data._apply_label_filters(frame.copy(), {**cfg, "pollutants": ["pm10"]})
    assert pm10["station_code"].tolist() == ["pm10_only", "both"]

    joint = data._apply_label_filters(frame.copy(), {**cfg, "pollutants": ["pm10", "pm25"]})
    assert joint["station_code"].tolist() == ["pm10_only", "pm25_only", "both"]


def check_joint_target_masks():
    with TemporaryDirectory() as td:
        root = Path(td)
        for name in ("high", "low", "no2_tropomi"):
            (root / name).mkdir()
        for code in ("pm10_only", "pm25_only"):
            np.save(root / "high" / f"{code}.npy", np.ones((2, 2, 10), dtype="float32"))
            np.save(root / "low" / f"{code}.npy", np.ones((2, 2, 10), dtype="float32"))
            np.save(root / "no2_tropomi" / f"{code}.npy", np.ones((2, 2), dtype="float32"))

        old_high, old_low, old_sat = data.paths.HIGH, data.paths.LOW, data.paths.SAT
        try:
            data.paths.HIGH = root / "high"
            data.paths.LOW = root / "low"
            data.paths.SAT = root
            ds = data.EEA(
                pd.DataFrame({
                    "station_code": ["pm10_only", "pm25_only"],
                    "pm10": [12.0, np.nan],
                    "pm25": [np.nan, 7.0],
                }),
                ["no2_tropomi"],
                np.array([2.0, 2.0]),
                np.array([1.0, 1.0]),
                {"no2_tropomi": (1.0, 1.0)},
                {
                    "pollutants": ["pm10", "pm25"],
                    "use_aer_wide": False,
                    "use_dem": False,
                    "cache_patches": False,
                },
            )
            assert ds[0][-1].tolist() == [1.0, 0.0]
            assert ds[1][-1].tolist() == [0.0, 1.0]
        finally:
            data.paths.HIGH = old_high
            data.paths.LOW = old_low
            data.paths.SAT = old_sat


def main():
    runtime.apply_runtime_config()
    assert torch.get_num_threads() == CPU_THREADS
    assert torch.get_num_interop_threads() == CPU_INTEROP_THREADS
    sf = pd.DataFrame({"fold": ["TEST", "fold2_france", "UNASSIGNED", "fold1_iberia"]})
    assert folds.development_fold_names(sf) == ["fold1_iberia", "fold2_france"]
    check_experiment_artifacts()
    check_summary_script_logic()
    check_patch_cache()
    check_configured_pollutant_filtering()
    check_joint_target_masks()

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
    model = build_resnet_model(2, model_cfg, 1)
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
    layer4_model = build_resnet_model(2, layer4_cfg, 1)
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
    check_scratch_cnn()


if __name__ == "__main__":
    main()
