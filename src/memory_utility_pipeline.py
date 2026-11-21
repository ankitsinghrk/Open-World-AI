"""Integrate all six memory utility factors into one scoring pipeline."""

import copy

from src.memory_factor_calculator import MemoryFactorCalculator


class MemoryUtilityPipeline:

    def __init__(
        self,
        memory_utility,
        learning_estimator,
        future_relevance_estimator,
        uncertainty_reduction_estimator,
        temporal_relevance_estimator,
        factor_calculator=None
    ):

        components = {
            "Memory utility": (memory_utility, "score"),
            "Learning estimator": (learning_estimator, "estimate"),
            "Future relevance estimator": (
                future_relevance_estimator, "estimate"
            ),
            "Uncertainty reduction estimator": (
                uncertainty_reduction_estimator, "estimate"
            ),
            "Temporal relevance estimator": (
                temporal_relevance_estimator, "estimate"
            )
        }

        for name, (component, method) in components.items():
            if not callable(getattr(component, method, None)):
                raise ValueError(
                    f"{name} must provide a callable {method} method"
                )

        if factor_calculator is None:
            factor_calculator = MemoryFactorCalculator()

        if not callable(getattr(factor_calculator, "calculate", None)):
            raise ValueError("Factor calculator must provide calculate")

        self.memory_utility = memory_utility
        self.learning_estimator = learning_estimator
        self.future_relevance_estimator = future_relevance_estimator
        self.uncertainty_reduction_estimator = (
            uncertainty_reduction_estimator
        )
        self.temporal_relevance_estimator = temporal_relevance_estimator
        self.factor_calculator = factor_calculator

    @staticmethod
    def _validate_record(record, name):

        if not isinstance(record, dict):
            raise ValueError(f"{name} must be a dictionary")

        if set(record) != {"id", "item"}:
            raise ValueError(f"{name} must contain exactly id and item")

        if not isinstance(record["id"], str) or not record["id"]:
            raise ValueError(f"{name} ID must be a non-empty string")

        if record["item"] is None:
            raise ValueError(f"{name} item cannot be empty")

        return record["id"]

    @staticmethod
    def _validate_memory_records(memory_records):

        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        seen_ids = set()

        for record in memory_records:

            memory_id = MemoryUtilityPipeline._validate_record(
                record, "Memory record"
            )

            if memory_id in seen_ids:
                raise ValueError("Memory IDs must be unique")

            seen_ids.add(memory_id)

        return seen_ids

    def score_candidate(
        self,
        candidate_record,
        memory_records,
        validation_data,
        evaluation_data,
        future_tasks,
        current_time
    ):

        candidate_id = self._validate_record(
            candidate_record, "Candidate record"
        )

        existing_ids = self._validate_memory_records(memory_records)

        if candidate_id in existing_ids:
            raise ValueError("Candidate ID already exists in memory")

        candidate = copy.deepcopy(candidate_record)
        records = copy.deepcopy(list(memory_records))

        candidate_item = candidate["item"]

        if not isinstance(candidate_item, dict):
            raise ValueError("Candidate item must contain features")

        if "features" not in candidate_item:
            raise ValueError("Candidate item must contain features")

        # 1. Learning Contribution
        learning_records = records + [candidate]

        learning = self.learning_estimator.estimate(
            candidate_id,
            learning_records,
            validation_data
        )

        # 2. Future Relevance
        future_relevance = self.future_relevance_estimator.estimate(
            candidate_item,
            future_tasks
        )

        # 3. Uncertainty Reduction
        uncertainty_reduction = (
            self.uncertainty_reduction_estimator.estimate(
                candidate,
                records,
                evaluation_data
            )
        )

        # 4. Temporal Relevance
        temporal_relevance = self.temporal_relevance_estimator.estimate(
            candidate,
            current_time
        )

        # 5. Novelty and 6. Redundancy
        calculated_factors = self.factor_calculator.calculate(
            candidate_item["features"],
            records
        )

        # Combine all six factors
        factors = {
            "learning": learning,
            "novelty": calculated_factors["novelty"],
            "future_relevance": future_relevance,
            "uncertainty_reduction": uncertainty_reduction,
            "temporal_relevance": temporal_relevance,
            "redundancy": calculated_factors["redundancy"]
        }

        # Calculate final utility score
        utility_score = self.memory_utility.score(factors)

        return {
            "candidate_id": candidate_id,
            "utility_score": utility_score,
            "factors": factors
        }