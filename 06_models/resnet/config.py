BACKBONE_MODE = "frozen"  # "frozen" trains only new branches/head; "layer4" also fine-tunes ResNet layer4.
LR_LAYER4 = 1e-6  # tuning value: select with development-fold CV before using TEST

RESNET_CONFIG = {
    "model": "resnet",
    "experiment": f"resnet_{BACKBONE_MODE}",
    "backbone_mode": BACKBONE_MODE,
    "lr_head": 1e-5,        # Learning rate for new pollution-specific branches and regression head.
    "lr_layer4": LR_LAYER4, # Smaller learning rate for trainable pretrained ResNet layer4 parameters.
    "wd_head": 1e-7,        # AdamW weight decay used by each active ResNet optimizer group.
    "proj_dim": 64,         # Size of the high-resolution image vector after projection before fusion.
    "dropout": 0.8,         # Drop 80% of activations at shared dropout layers during training.
    "low_cnn_ch1": 32,      # First channel width in the low-resolution Sentinel-2 branch.
    "low_cnn_ch2": 128,     # Second channel width and output width before low-branch projection.
    "s5p_cnn_hidden": 32,   # Channel width used inside the Sentinel-5P patch branch.
    "head_hidden": 64,      # Hidden width of the final regression head after feature concatenation.
    "wide_feat": 32,        # Number of learned features from the wide aerosol-context patch.
    "dem_feat": 32,         # Number of learned features from the DEM elevation patch.
    "pretrained": True,     # Load BigEarthNet ResNet50 weights before applying freeze/fine-tune rules.
}
