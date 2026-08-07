import torch
import torch.nn as nn


def init_common_branches(model, n_s5p, cfg, n_out, high_feat):
    proj, dropout = cfg["proj_dim"], cfg["dropout"]
    c1, c2 = cfg["low_cnn_ch1"], cfg["low_cnn_ch2"]
    s5p_hidden, head_hidden = cfg["s5p_cnn_hidden"], cfg["head_hidden"]
    wf, df = cfg["wide_feat"], cfg["dem_feat"]
    model.use_wide, model.use_dem = cfg["use_aer_wide"], cfg["use_dem"]
    model.proj_h = nn.Linear(high_feat, proj)
    model.low_cnn = nn.Sequential(
        nn.Conv2d(10, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, 128), nn.ReLU(True))
    model.s5p_cnn = nn.Sequential(
        nn.Conv2d(n_s5p, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
        nn.Conv2d(s5p_hidden, s5p_hidden, 3, padding=1), nn.BatchNorm2d(s5p_hidden), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model.norm_h = nn.BatchNorm1d(proj, affine=False)
    model.norm_l = nn.BatchNorm1d(128, affine=False)
    model.norm_s = nn.BatchNorm1d(s5p_hidden, affine=False)
    n_scalars = n_s5p + int(model.use_wide) + int(model.use_dem)
    head_in = proj + 128 + s5p_hidden + n_scalars
    if model.use_wide:
        model.wide_cnn = nn.Sequential(
            nn.Conv2d(1, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
            nn.Conv2d(wf, wf, 3, stride=2, padding=1), nn.BatchNorm2d(wf), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model.norm_w = nn.BatchNorm1d(wf, affine=False)
        head_in += wf
    if model.use_dem:
        model.dem_cnn = nn.Sequential(
            nn.Conv2d(1, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(df, df, 3, padding=1), nn.BatchNorm2d(df), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model.norm_d = nn.BatchNorm1d(df, affine=False)
        head_in += df
    model.head = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(head_in, head_hidden), nn.ReLU(True),
        nn.Dropout(dropout), nn.Linear(head_hidden, n_out))


def forward_common(model, xh, xl, xs_patch, xw, xd, xs_mean):
    parts = [model.norm_h(model.proj_h(model.backbone(xh))),
             model.norm_l(model.low_cnn(xl)),
             model.norm_s(model.s5p_cnn(xs_patch))]
    if model.use_wide:
        parts.append(model.norm_w(model.wide_cnn(xw)))
    if model.use_dem:
        parts.append(model.norm_d(model.dem_cnn(xd)))
    parts.append(xs_mean)
    return model.head(torch.cat(parts, dim=1))
