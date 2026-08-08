import torch.nn as nn

from shared import multimodal


class ScratchHighEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), convs_per_block=2):
        super().__init__()
        self.channels = tuple(channels)
        self.convs_per_block = convs_per_block
        self.feature_dim = self.channels[-1]
        layers = []
        in_ch = 10
        for out_ch in self.channels:
            for _ in range(convs_per_block):
                layers.extend([nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)])
                in_ch = out_ch
            layers.append(nn.MaxPool2d(2))
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out):
        super().__init__()
        self.backbone = ScratchHighEncoder(cfg["high_channels"], cfg["convs_per_block"])
        self.lr = cfg["lr"]
        self.experiment = cfg["experiment"]
        self.wide = cfg["wide"]
        multimodal.init_common_branches(self, n_s5p, cfg, n_out, self.backbone.feature_dim)

    def parameter_metadata(self):
        high_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model": "cnn",
            "experiment": self.experiment,
            "high_encoder": "scratch_cnn_" + "_".join(str(c) for c in self.backbone.channels),
            "wide": self.wide,
            "high_encoder_channels": self.backbone.channels,
            "convs_per_block": self.backbone.convs_per_block,
            "high_encoder_feature_dim": self.backbone.feature_dim,
            "lr": self.lr,
            "total_model_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_high_encoder_parameters": high_trainable,
            "trainable_non_high_encoder_parameters": total_trainable - high_trainable,
        }

    def forward(self, xh, xl, xs_patch, xw, xd, xs_mean):
        return multimodal.forward_common(self, xh, xl, xs_patch, xw, xd, xs_mean)


def build_model(n_s5p, cfg, n_out):
    return Net(n_s5p, cfg, n_out)
