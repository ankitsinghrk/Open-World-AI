import math


class UncertaintyEstimator:

    def calculate(self, probabilities):

        if not isinstance(probabilities, (list, tuple)):
            raise ValueError("Probabilities must be a list or tuple")

        if len(probabilities) == 0:
            raise ValueError("Probabilities cannot be empty")

        if any(not isinstance(p, (int, float)) or isinstance(p, bool)
               or not math.isfinite(p) or p < 0 or p > 1
               for p in probabilities):
            raise ValueError("Invalid probability value")

        total = sum(probabilities)

        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Probabilities must sum to 1")

        if len(probabilities) == 1:
            return 0.0

        entropy = 0.0

        for probability in probabilities:
            if probability > 0:
                entropy -= probability * math.log(probability)

        maximum_entropy = math.log(len(probabilities))

        uncertainty = entropy / maximum_entropy

        return uncertainty
