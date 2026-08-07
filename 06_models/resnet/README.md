# ResNet Models

The ResNet model uses a BigEarthNet-pretrained ResNet50 for the high-resolution
Sentinel-2 patch. The rest of the pollution model adds trainable branches for
the lower-resolution Sentinel-2 patch, Sentinel-5P NO2/CO, wider aerosol input,
DEM/elevation, scalar inputs, and the regression head.

ResNet-specific settings live in `resnet/config.py`.

## Backbone Modes

Select the mode with:

```python
BACKBONE_MODE = "frozen"
BACKBONE_MODE = "layer4"
```

`"frozen"` is transfer learning with a frozen feature extractor. It is not
fine-tuning of the pretrained backbone.

In `"frozen"` mode:

- ResNet stem and layers 1-4 are frozen.
- The projection, auxiliary branches, and regression head are trainable.
- Pretrained ResNet BatchNorm running statistics stay fixed.

`"layer4"` is partial fine-tuning.

In `"layer4"` mode:

- ResNet stem and layers 1-3 are frozen.
- Non-BatchNorm parameters in ResNet layer4 are trainable.
- BatchNorm modules in the pretrained ResNet stay in eval mode.
- BatchNorm affine parameters stay frozen for now.
- The projection, auxiliary branches, and regression head are trainable.

The pollution-model parameters use `lr_head`. Trainable pretrained layer4
parameters use `lr_layer4`, which is intentionally smaller. That value is a
tuning setting and should be selected using development-fold CV before any
sealed-test evaluation.

`"layer4"` is not automatically the preferred model. It is only available as a
controlled experiment.

