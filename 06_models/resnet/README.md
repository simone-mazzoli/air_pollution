# ResNet Models

This folder contains the BigEarthNet-pretrained ResNet50 model used for the
current transfer-learning experiments.

BigEarthNet is a remote-sensing dataset built from Sentinel imagery. Pretraining
on it gives the ResNet image features that are useful for satellite patches
before the model is trained on the smaller pollution dataset.

The ResNet is used only for the high-resolution Sentinel-2 branch. The lower
resolution Sentinel-2 patch, Sentinel-5P inputs, aerosol context, DEM, scalar
values, and regression head are pollution-specific branches defined around that
backbone.

## Sentinel-2 Input Channels

The high-resolution Sentinel-2 patch has 10 channels. A normal ImageNet ResNet
expects 3 RGB channels, so this model creates the ResNet with `in_chans=10`.
That lets the first convolution accept the multispectral Sentinel-2 tensor used
by this project.

The BigEarthNet checkpoint is loaded only for tensors whose names and shapes
match the local 10-channel ResNet.

## Configuration

ResNet-specific settings live in `06_models/resnet/config.py`.

Change the ResNet mode with:

```python
BACKBONE_MODE = "frozen"
BACKBONE_MODE = "layer4"
```

`RESNET_CONFIG["experiment"]` is built from this mode:

```text
frozen -> resnet_frozen
layer4 -> resnet_layer4
```

That experiment name decides which result folder is used.

## `resnet_frozen`

In frozen mode:

- the pretrained ResNet stem and layers 1-4 are frozen;
- the projection after the ResNet is trainable;
- the lower-resolution Sentinel-2 branch is trainable;
- the Sentinel-5P, aerosol, DEM, scalar, and regression head parameters are
  trainable;
- pretrained ResNet BatchNorm layers stay in evaluation mode.

This mode treats the pretrained ResNet as a fixed feature extractor.

Checked parameter counts:

```text
total      23,766,049
trainable     236,065
frozen     23,529,984
```

## `resnet_layer4`

In layer4 mode:

- the ResNet stem, layer1, layer2, and layer3 are frozen;
- non-BatchNorm parameters in ResNet layer4 are trainable;
- pretrained BatchNorm affine parameters stay frozen;
- pretrained BatchNorm running statistics stay in evaluation mode;
- the pollution-specific branches and regression head remain trainable.

This is partial fine-tuning because only the last stage of the pretrained
backbone can adapt to the PM2.5 task.

Checked parameter counts:

```text
total                       23,766,049
trainable                   15,178,273
trainable layer4            14,942,208
trainable non-layer4           236,065
```

The large increase in trainable parameters is why this mode may adapt better but
also has more overfitting risk.

## BatchNorm Behavior

BatchNorm layers in the pretrained ResNet stay frozen and in evaluation mode in
both ResNet modes. The model overrides `train()` so that calling `model.train()`
for the full pollution model does not accidentally switch pretrained BatchNorm
back to training mode.

This matters because small station batches are not a good basis for updating
BatchNorm statistics in a large pretrained image backbone.

## Optimizer Groups

Frozen mode uses one optimizer group for all trainable pollution-specific
parameters:

```text
lr_head
```

Layer4 mode uses two optimizer groups:

```text
lr_head    -> new pollution-specific parameters
lr_layer4  -> trainable pretrained layer4 parameters
```

`lr_layer4` is intentionally smaller because pretrained weights are usually
updated more cautiously than newly initialized pollution heads. It should be
selected with development-fold CV, not with the sealed TEST set.
