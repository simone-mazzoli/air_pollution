# Custom CNN

The custom CNN is not implemented yet. It will be a CNN trained from scratch,
not a BigEarthNet-pretrained ResNet.

When it is added, it should use the shared EEA dataset from `06_models/shared/`.
That means it receives the same batch fields as the ResNet variants:

```text
xh, xl, xs_patch, xw, xd, xs_mean
```

It must return one tensor with shape:

```text
(batch, number_of_pollutants)
```

The CNN must not create its own fold assignment or its own geographic buffer.
It should use the same folds, 100 km buffer, preprocessing, metrics, and sealed
test setup as the ResNet experiments.

The architecture has not been finalized yet.

