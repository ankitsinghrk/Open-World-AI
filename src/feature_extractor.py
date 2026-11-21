import math


class FeatureExtractor:

    def __init__(self, encoder):
        if not callable(encoder):
            raise ValueError("Encoder must be callable")

        self.encoder = encoder

    def extract(self, input_data):

        if input_data is None:
            raise ValueError("Input data cannot be empty")

        features = self.encoder(input_data)

        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError("Encoder must return a non-empty feature vector")

        for value in features:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("Features must contain finite numbers")

        return list(features)