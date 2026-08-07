import torch.nn as nn

from shared import multimodal


class ScratchHighEncoder(nn.Module):
    feature_dim = 256

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out):
        super().__init__()
        self.backbone = ScratchHighEncoder()
        self.lr = cfg["lr"]
        multimodal.init_common_branches(self, n_s5p, cfg, n_out, self.backbone.feature_dim)

    def parameter_metadata(self):
        high_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model": "cnn",
            "experiment": "cnn",
            "high_encoder": "scratch_cnn_32_64_128_256",
            "high_encoder_feature_dim": self.backbone.feature_dim,
            "lr": self.lr,
            "trainable_high_encoder_parameters": high_trainable,
            "trainable_non_high_encoder_parameters": total_trainable - high_trainable,
        }

    def forward(self, xh, xl, xs_patch, xw, xd, xs_mean):
        return multimodal.forward_common(self, xh, xl, xs_patch, xw, xd, xs_mean)


def build_model(n_s5p, cfg, n_out):
    return Net(n_s5p, cfg, n_out)
