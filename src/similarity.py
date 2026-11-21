"""Shared feature-similarity utilities for memory novelty and redundancy.

Cosine similarity is naturally in [-1, 1]. The utility framework uses
normalized factors in [0, 1], so this module maps cosine similarity with
(normalized_cosine) = (cosine + 1) / 2.
"""

import math


class FeatureSimilarity:
    """Calculate normalized cosine similarity between feature vectors."""

    @staticmethod
    def _validate_vector(vector, name):
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"{name} must be a non-empty list or tuple")

        checked = []
        for value in vector:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must contain only finite numbers")
            checked.append(float(value))
        return checked

    def cosine(self, first, second):
        """Return ordinary cosine similarity in [-1, 1]."""
        a = self._validate_vector(first, "First vector")
        b = self._validate_vector(second, "Second vector")

        if len(a) != len(b):
            raise ValueError("Feature dimensions must match")

        norm_a = math.sqrt(sum(value * value for value in a))
        norm_b = math.sqrt(sum(value * value for value in b))
        if norm_a == 0.0 or norm_b == 0.0:
            raise ValueError("Cosine similarity is undefined for a zero vector")

        result = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
        return max(-1.0, min(1.0, result))

    def normalized(self, first, second):
        """Map cosine similarity from [-1, 1] into [0, 1]."""
        return (self.cosine(first, second) + 1.0) / 2.0
