import math


class NoveltyEstimator:

    def __init__(self, k=3, calibration_distance=1.0):

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")

        if (not isinstance(calibration_distance, (int, float))
                or isinstance(calibration_distance, bool)
                or not math.isfinite(calibration_distance)
                or calibration_distance <= 0):
            raise ValueError("Calibration distance must be positive")

        self.k = k
        self.calibration_distance = calibration_distance

    def calculate(self, features, reference_bank):

        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("Features cannot be empty")

        if not isinstance(reference_bank, (list, tuple)):
            raise ValueError("Reference bank must be a list or tuple")

        if len(reference_bank) < self.k:
            raise ValueError("Not enough reference features")

        if any(not isinstance(x, (int, float))
               or isinstance(x, bool) or not math.isfinite(x)
               for x in features):
            raise ValueError("Invalid feature value")

        distances = []

        for reference in reference_bank:

            if not isinstance(reference, (list, tuple)):
                raise ValueError("Invalid reference feature")

            if len(reference) != len(features):
                raise ValueError("Feature dimensions must match")

            if any(not isinstance(x, (int, float))
                   or isinstance(x, bool) or not math.isfinite(x)
                   for x in reference):
                raise ValueError("Invalid reference value")

            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(features, reference))
            )

            distances.append(distance)

        distances.sort()

        average_distance = sum(distances[:self.k]) / self.k

        novelty = min(
            average_distance / self.calibration_distance,
            1.0
        )

        return novelty