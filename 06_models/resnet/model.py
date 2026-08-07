import torch.nn as nn

from shared import multimodal

SUPPORTED_BACKBONE_MODES = {"frozen", "layer4"}


class Net(nn.Module):
    def __init__(self, n_s5p, cfg, n_out, pretrained=True):
        super().__init__()
        import timm

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
        multimodal.init_common_branches(self, n_s5p, cfg, n_out, feat)

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
        return multimodal.forward_common(self, xh, xl, xs_patch, xw, xd, xs_mean)


def build_model(n_s5p, cfg, n_out):
    return Net(n_s5p, cfg, n_out, pretrained=cfg.get("pretrained", True))
