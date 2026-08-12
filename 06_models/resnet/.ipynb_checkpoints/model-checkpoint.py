import torch
import torch.nn as nn

SUPPORTED_BACKBONE_MODES = {"frozen", "layer4"}


class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out, pretrained=True):
        super().__init__()
        import timm

        proj, dropout = cfg["proj_dim"], cfg["dropout"]
        c1, c2 = cfg["low_cnn_ch1"], cfg["low_cnn_ch2"]
        s5p_hidden, head_hidden = cfg["s5p_cnn_hidden"], cfg["head_hidden"]
        wf, df = cfg["wide_feat"], cfg["dem_feat"]
        self.use_wide, self.use_dem = cfg["use_aer_wide"], cfg["use_dem"]
        self.backbone_mode = cfg.get("backbone_mode", "frozen")
        self.lr_head = cfg["lr_head"]
        self.lr_layer4 = cfg.get("lr_layer4")
        if self.backbone_mode not in SUPPORTED_BACKBONE_MODES:
            raise ValueError(f"unsupported ResNet backbone_mode: {self.backbone_mode}")
        self.backbone = timm.create_model("resnet50", pretrained=False, in_chans=10, num_classes=0)
        feat = self.backbone.num_features
        if pretrained:
            self._load_pretrained()
        self._configure_backbone_trainability()
        self.proj_h = nn.Linear(feat, proj)
        self.low_cnn = nn.Sequential(
            nn.Conv2d(10, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, 128), nn.ReLU(True))
        self.s5p_cnn = nn.Sequential(
            nn.Conv2d(n_s5p, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
            nn.Conv2d(s5p_hidden, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.norm_h = nn.BatchNorm1d(proj, affine=False)
        self.norm_l = nn.BatchNorm1d(128, affine=False)
        self.norm_s = nn.BatchNorm1d(s5p_hidden, affine=False)
        n_scalars = n_s5p + int(self.use_wide) + int(self.use_dem)
        head_in = proj + 128 + s5p_hidden + n_scalars
        if self.use_wide:
            self.wide_cnn = nn.Sequential(
                nn.Conv2d(1, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
                nn.Conv2d(wf, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.norm_w = nn.BatchNorm1d(wf, affine=False)
            head_in += wf
        if self.use_dem:
            self.dem_cnn = nn.Sequential(
                nn.Conv2d(1, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(df, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.norm_d = nn.BatchNorm1d(df, affine=False)
            head_in += df
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, head_hidden), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(head_hidden, n_out))

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_mode == "frozen":
            self.backbone.eval()
        self._set_backbone_batchnorm_eval()
        return self

    def _configure_backbone_trainability(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.backbone_mode == "layer4":
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        self._set_backbone_batchnorm_eval()

    def _set_backbone_batchnorm_eval(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def layer4_trainable_parameters(self):
        if self.backbone_mode != "layer4":
            return []
        return [p for p in self.backbone.layer4.parameters() if p.requires_grad]

    def optimizer_parameter_groups(self, cfg):
        layer4_params = self.layer4_trainable_parameters()
        layer4_ids = {id(p) for p in layer4_params}
        new_params = [p for p in self.parameters() if p.requires_grad and id(p) not in layer4_ids]
        groups = [{"params": new_params, "lr": cfg["lr_head"], "weight_decay": cfg["wd_head"]}]
        if layer4_params:
            groups.append({"params": layer4_params, "lr": cfg["lr_layer4"], "weight_decay": cfg["wd_head"]})
        return groups

    def parameter_metadata(self):
        layer4_trainable = sum(p.numel() for p in self.layer4_trainable_parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model": "resnet",
            "experiment": f"resnet_{self.backbone_mode}",
            "backbone_mode": self.backbone_mode,
            "lr_head": self.lr_head,
            "lr_layer4": self.lr_layer4,
            "trainable_layer4_parameters": layer4_trainable,
            "trainable_non_layer4_parameters": total_trainable - layer4_trainable,
        }

    def _load_pretrained(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0", "model.safetensors")
        ckpt = load_file(path)
        sd = self.backbone.state_dict()
        remap, n = {}, 0
        for k, v in ckpt.items():
            b = k
            for p in ("model.vision_encoder.", "vision_encoder.", "model.", "backbone.",
                      "network.", "encoder.", "resnet.", "timm_model."):
                if b.startswith(p):
                    b = b[len(p):]
            if b in sd and sd[b].shape == v.shape:
                remap[b] = v
                n += 1
        self.backbone.load_state_dict(remap, strict=False)
        print(f"  loaded {n}/{len(sd)} pretrained tensors")

    def forward(self, xh, xl, xs_patch, xw, xd, xs_mean):
        parts = [self.norm_h(self.proj_h(self.backbone(xh))),
                 self.norm_l(self.low_cnn(xl)),
                 self.norm_s(self.s5p_cnn(xs_patch))]
        if self.use_wide:
            parts.append(self.norm_w(self.wide_cnn(xw)))
        if self.use_dem:
            parts.append(self.norm_d(self.dem_cnn(xd)))
        parts.append(xs_mean)
        return self.head(torch.cat(parts, dim=1))


def build_model(n_s5p, cfg, n_out):
    return Net(n_s5p, cfg, n_out, pretrained=cfg.get("pretrained", True))
