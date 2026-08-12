def build_model(*args, **kwargs):
    raise NotImplementedError(
        "CNN model has not been implemented yet. It must accept the same EEA batch inputs as ResNet "
        "and return a (batch, n_pollutants) prediction tensor."
    )
