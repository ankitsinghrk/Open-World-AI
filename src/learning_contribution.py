"""Estimate a memory item's learning contribution by leave-one-out evaluation.

The evaluator must return a scalar validation-performance score where larger
values mean better performance. This module measures the score drop when one
memory item is removed from the complete memory set.
"""

import copy
import math


class LearningContributionEstimator:
    def __init__(self, evaluator, reference_gain):
        if not callable(evaluator):
            raise ValueError("Evaluator must be callable")

        if (
            isinstance(reference_gain, bool)
            or not isinstance(reference_gain, (int, float))
            or not math.isfinite(reference_gain)
            or reference_gain <= 0
        ):
            raise ValueError("Reference gain must be a finite positive number")

        self.evaluator = evaluator
        self.reference_gain = float(reference_gain)

    @staticmethod
    def _validate_records(memory_records):
        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        seen_ids = set()
        for record in memory_records:
            if not isinstance(record, dict) or set(record) != {"id", "item"}:
                raise ValueError("Each memory record must contain exactly id and item")
            memory_id = record["id"]
            if not isinstance(memory_id, str) or not memory_id:
                raise ValueError("Memory IDs must be non-empty strings")
            if memory_id in seen_ids:
                raise ValueError("Memory IDs must be unique")
            seen_ids.add(memory_id)

        return seen_ids

    @staticmethod
    def _validate_score(score):
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("Evaluator must return a finite numeric score")
        return float(score)

    def estimate(self, memory_id, memory_records, validation_data):
        """Return normalized contribution for one stored memory ID in [0, 1].

        contribution = max(0, score(all memories) - score(without target))
        normalized contribution is clipped by reference_gain.
        """
        seen_ids = self._validate_records(memory_records)

        if memory_id not in seen_ids:
            raise KeyError(f"Unknown memory ID: {memory_id}")

        full_records = copy.deepcopy(list(memory_records))
        ablated_records = [
            copy.deepcopy(record)
            for record in memory_records
            if record["id"] != memory_id
        ]

        full_items = [record["item"] for record in full_records]
        ablated_items = [record["item"] for record in ablated_records]
        safe_validation_data = copy.deepcopy(validation_data)

        full_score = self._validate_score(
            self.evaluator(full_items, copy.deepcopy(safe_validation_data))
        )
        ablated_score = self._validate_score(
            self.evaluator(ablated_items, copy.deepcopy(safe_validation_data))
        )

        performance_gain = max(0.0, full_score - ablated_score)
        normalized = performance_gain / self.reference_gain
        return min(1.0, normalized)
