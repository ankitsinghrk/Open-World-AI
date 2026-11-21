"""Calculate normalized memory utility from supplied factor values.

This module scores factors; it does not estimate the factors themselves.
"""

import math

from src.memory_factor_calculator import MemoryFactorCalculator


class MemoryUtility:
    FACTOR_NAMES = (
        "learning",
        "novelty",
        "future_relevance",
        "uncertainty_reduction",
        "temporal_relevance",
        "redundancy",
    )

    def __init__(self, weights):
        if not isinstance(weights, dict):
            raise ValueError("Weights must be provided as a dictionary")

        if set(weights) != set(self.FACTOR_NAMES):
            raise ValueError("Weights must contain exactly the six utility factors")

        checked_weights = {}
        for name in self.FACTOR_NAMES:
            value = weights[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Weights must be finite, non-negative numbers")
            checked_weights[name] = float(value)

        if not math.isclose(sum(checked_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Utility weights must sum to 1")

        self.weights = checked_weights

    def score(self, factors):
        """Return the V1.1 utility score for six normalized factor values.

        Every factor must be a finite number in the inclusive range [0, 1].
        Redundancy is inverted in the formula: lower redundancy adds more utility.
        """
        if not isinstance(factors, dict):
            raise ValueError("Factors must be provided as a dictionary")

        if set(factors) != set(self.FACTOR_NAMES):
            raise ValueError("Factors must contain exactly the six utility factors")

        checked_factors = {}
        for name in self.FACTOR_NAMES:
            value = factors[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError("Every utility factor must be a finite number from 0 to 1")
            checked_factors[name] = float(value)

        score = (
            self.weights["learning"] * checked_factors["learning"]
            + self.weights["novelty"] * checked_factors["novelty"]
            + self.weights["future_relevance"] * checked_factors["future_relevance"]
            + self.weights["uncertainty_reduction"] * checked_factors["uncertainty_reduction"]
            + self.weights["temporal_relevance"] * checked_factors["temporal_relevance"]
            + self.weights["redundancy"] * (1.0 - checked_factors["redundancy"])
        )

        return score

    def score_candidate(self, candidate_features, memory_records, other_factors):
        """Calculate novelty/redundancy, then score the incoming candidate.

        other_factors must contain exactly learning, future_relevance,
        uncertainty_reduction, and temporal_relevance.
        """
        if not isinstance(other_factors, dict):
            raise ValueError("Other factors must be provided as a dictionary")

        required = {
            "learning",
            "future_relevance",
            "uncertainty_reduction",
            "temporal_relevance",
        }
        if set(other_factors) != required:
            raise ValueError("Other factors must contain exactly four remaining factors")

        calculated = MemoryFactorCalculator().calculate(
            candidate_features, memory_records
        )
        factors = dict(other_factors)
        factors.update(calculated)
        return self.score(factors)
