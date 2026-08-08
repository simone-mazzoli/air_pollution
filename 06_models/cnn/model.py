import torch.nn as nn

from shared import multimodal


class ScratchHighEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.channels = tuple(channels)
        self.feature_dim = c4
        self.net = nn.Sequential(
            nn.Conv2d(10, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(True),
            nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out):
        super().__init__()
        self.backbone = ScratchHighEncoder(cfg["high_channels"])
        self.lr = cfg["lr"]
        self.experiment = cfg["experiment"]
        multimodal.init_common_branches(self, n_s5p, cfg, n_out, self.backbone.feature_dim)

    def parameter_metadata(self):
        high_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model": "cnn",
            "experiment": self.experiment,
            "high_encoder": "scratch_cnn_" + "_".join(str(c) for c in self.backbone.channels),
            "high_encoder_channels": self.backbone.channels,
            "high_encoder_feature_dim": self.backbone.feature_dim,
            "lr": self.lr,
            "trainable_high_encoder_parameters": high_trainable,
            "trainable_non_high_encoder_parameters": total_trainable - high_trainable,
        }

    def forward(self, xh, xl, xs_patch, xw, xd, xs_mean):
        return multimodal.forward_common(self, xh, xl, xs_patch, xw, xd, xs_mean)


def build_model(n_s5p, cfg, n_out):
    return Net(n_s5p, cfg, n_out)
