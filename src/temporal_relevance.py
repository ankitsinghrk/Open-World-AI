"""Estimate the probability that a memory item remains valid over time."""

import math


class TemporalRelevanceEstimator:

    def __init__(self, validity_function):
        if not callable(validity_function):
            raise ValueError("Validity function must be callable")

        self.validity_function = validity_function

    @staticmethod
    def validate_record(record):
        if not isinstance(record, dict):
            raise ValueError("Memory record must be a dictionary")

        if set(record) != {"id", "item"}:
            raise ValueError("Memory record must contain id and item")

        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError("Memory ID must be a non-empty string")

        if record["item"] is None:
            raise ValueError("Memory item cannot be empty")

    @staticmethod
    def validate_probability(probability):
        if isinstance(probability, bool):
            raise ValueError("Probability must be numeric")

        if not isinstance(probability, (int, float)):
            raise ValueError("Probability must be numeric")

        if not math.isfinite(probability):
            raise ValueError("Probability must be finite")

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")

        return float(probability)

    def estimate(self, memory_record, current_time):

        self.validate_record(memory_record)

        probability = self.validity_function(
            memory_record["item"],
            current_time
        )

        return self.validate_probability(probability)