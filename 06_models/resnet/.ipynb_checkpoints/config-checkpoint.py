BACKBONE_MODE = "frozen"
LR_LAYER4 = 1e-6  # tuning value: select with development-fold CV before using TEST

RESNET_CONFIG = {
    "model": "resnet",
    "experiment": f"resnet_{BACKBONE_MODE}",
    "backbone_mode": BACKBONE_MODE,
    "lr_head": 1e-5,
    "lr_layer4": LR_LAYER4,
    "wd_head": 1e-7,
    "proj_dim": 64,
    "dropout": 0.8,
    "low_cnn_ch1": 32,
    "low_cnn_ch2": 128,
    "s5p_cnn_hidden": 32,
    "head_hidden": 64,
    "wide_feat": 32,
    "dem_feat": 32,
    "pretrained": True,
}
