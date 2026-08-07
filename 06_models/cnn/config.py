CNN_CONFIG = {
    "model": "cnn",
    "experiment": "cnn",
    "lr": 3e-4,             # Learning rate for the single AdamW group containing all CNN parameters.
    "weight_decay": 1e-7,   # AdamW weight decay for that same single CNN parameter group.
    "proj_dim": 64,         # Size of the high-resolution image vector after projection before fusion.
    "dropout": 0.5,         # Drop 50% of activations at shared dropout layers during training.
    "low_cnn_ch1": 32,      # First channel width in the low-resolution Sentinel-2 branch.
    "low_cnn_ch2": 128,     # Second channel width and output width before low-branch projection.
    "s5p_cnn_hidden": 32,   # Channel width used inside the Sentinel-5P patch branch.
    "head_hidden": 64,      # Hidden width of the final regression head after feature concatenation.
    "wide_feat": 32,        # Number of learned features from the wide aerosol-context patch.
    "dem_feat": 32,         # Number of learned features from the DEM elevation patch.
}
