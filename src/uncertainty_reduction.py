"""Measure predictive entropy reduction from including a memory item."""

import copy
from src.uncertainty import UncertaintyEstimator


class UncertaintyReductionEstimator:

    def __init__(self, prediction_function, uncertainty_estimator=None):
        if not callable(prediction_function):
            raise ValueError("Prediction function must be callable")

        self.prediction_function = prediction_function
        self.uncertainty_estimator = (
            uncertainty_estimator or UncertaintyEstimator()
        )

    @staticmethod
    def validate_record(record, name):
        if not isinstance(record, dict) or set(record) != {"id", "item"}:
            raise ValueError(f"{name} must contain exactly id and item")

        memory_id = record["id"]

        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError(f"{name} ID must be a non-empty string")

        if record["item"] is None:
            raise ValueError(f"{name} item cannot be empty")

        return memory_id

    def predict_uncertainty(self, memory_items, evaluation_sample):
        probabilities = self.prediction_function(
            copy.deepcopy(memory_items),
            copy.deepcopy(evaluation_sample)
        )

        return self.uncertainty_estimator.calculate(probabilities)

    def estimate(self, candidate_record, memory_records, evaluation_data):
        candidate_id = self.validate_record(
            candidate_record, "Candidate record"
        )

        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        seen_ids = set()
        existing_items = []

        for record in memory_records:
            memory_id = self.validate_record(record, "Memory record")

            if memory_id in seen_ids:
                raise ValueError("Memory IDs must be unique")

            seen_ids.add(memory_id)
            existing_items.append(copy.deepcopy(record["item"]))

        if candidate_id in seen_ids:
            raise ValueError("Candidate ID already exists in memory")

        if not isinstance(evaluation_data, (list, tuple)) or not evaluation_data:
            raise ValueError("Evaluation data must be non-empty")

        augmented_items = existing_items + [
            copy.deepcopy(candidate_record["item"])
        ]

        reductions = []

        for sample in evaluation_data:
            uncertainty_without = self.predict_uncertainty(
                existing_items, sample
            )

            uncertainty_with = self.predict_uncertainty(
                augmented_items, sample
            )

            reduction = max(
                0.0,
                uncertainty_without - uncertainty_with
            )

            reductions.append(reduction)

        return sum(reductions) / len(reductions)