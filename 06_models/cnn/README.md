# Custom CNN

This folder contains the custom CNN trained from scratch. It is not a
BigEarthNet-pretrained model, and it does not load any pretrained weights.

The scratch CNN is needed for the main scientific comparison:

```text
Does BigEarthNet pretraining help, or can a CNN trained only on this pollution
dataset learn comparable satellite features from scratch?
```

## What Should Stay The Same

To make the comparison fair, the CNN should use the same shared setup as the
ResNet experiments:

- the frozen `station_fold.csv` station assignments;
- the same sealed northern/eastern Germany TEST region;
- the same 100 km training-station buffer;
- the same label preparation and target transformation;
- the same required satellite and DEM inputs unless a change is explicitly part
  of the experiment;
- the same augmentation and TTA settings unless they are being tested on purpose;
- the same metrics and result metadata.

The CNN must not create its own fold assignment or its own TEST split.

## Current High-Resolution Encoder

The first implemented CNN component is the high-resolution Sentinel-2 encoder.
It replaces only the pretrained ResNet50 branch used for `xh`.

Input:

```text
10 x 120 x 120 Sentinel-2 patch
```

`cnn` architecture:

```text
Block 1: 10 -> 32 channels, two 3x3 convolutions, BatchNorm, ReLU, MaxPool
Block 2: 32 -> 64 channels, two 3x3 convolutions, BatchNorm, ReLU, MaxPool
Block 3: 64 -> 128 channels, two 3x3 convolutions, BatchNorm, ReLU, MaxPool
Block 4: 128 -> 256 channels, two 3x3 convolutions, BatchNorm, ReLU, MaxPool
AdaptiveAvgPool2d(1)
Flatten
```

`cnn --wide` uses the exact same four-block structure, but with wider channels:

```text
Block 1: 10 -> 48
Block 2: 48 -> 96
Block 3: 96 -> 192
Block 4: 192 -> 384
```

`cnn_deep` keeps the original widths but uses three Conv-BN-ReLU layers in each
block, for 12 high-resolution convolution layers total. `cnn_deep --wide`
combines that deeper block structure with the wider channels above. All variants
keep four MaxPool operations and the final `AdaptiveAvgPool2d(1)`.

Output:

```text
256-dimensional high-resolution feature vector for `cnn`
384-dimensional high-resolution feature vector for `cnn --wide`
```

That vector is passed through the same high-resolution projection used by the
ResNet setup so the shared fusion stage still receives a 64-dimensional
high-resolution representation.

The rest of the multimodal model is shared with the ResNet experiments:

- lower-resolution Sentinel-2 branch;
- Sentinel-5P branch;
- aerosol branch;
- DEM branch;
- scalar/context values;
- fusion and regression head.

## Current Optimization Settings

All trainable CNN parameters are currently optimized together in one AdamW
parameter group for `cnn`, `cnn --wide`, `cnn_deep`, and `cnn_deep --wide`:

```text
lr            3e-4
weight_decay  1e-7
dropout       0.5
```

`dropout=0.5` means the shared dropout layers drop 50% of activations during
training. There is not yet a separate learning rate for the scratch
high-resolution encoder versus the shared branches and regression head.

## Expected Model Interface

The shared dataset currently gives each model these tensors:

```text
xh       high-resolution Sentinel-2 patch
xl       lower-resolution Sentinel-2 context patch
xs_patch Sentinel-5P local patches
xw       wide aerosol patch
xd       DEM/elevation patch
xs_mean  scalar/context values
```

The model should return one tensor:

```text
(batch, number_of_pollutants)
```

## Still Open

This is only the first scratch-CNN integration step. The high-resolution encoder
is now implemented, but development-fold CV still has to decide whether this
capacity and the current training settings are appropriate.

Open choices include:

- whether this compact encoder is the right capacity;
- whether the current CNN-specific learning rate, dropout, and weight decay are
  best;
- whether learning rate and weight decay should be split by parameter group.

Those choices should be made with development-fold CV only, then compared with
the ResNet runs using the same frozen station assignments and buffer.
