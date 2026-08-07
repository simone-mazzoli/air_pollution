# Custom CNN

This folder is reserved for the required CNN trained from scratch. It is not a
BigEarthNet-pretrained model.

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

The CNN architecture is intentionally not decided yet. Open design choices
include:

- how deep the high-resolution CNN should be;
- how to combine high-resolution and lower-resolution image branches;
- whether the scratch CNN should reuse the current auxiliary branches exactly or
  only match their input information;
- dropout and regularization choices;
- learning rate and weight decay.

Those choices should be made with development-fold CV only, then compared with
the ResNet runs using the same frozen station assignments and buffer.
