import unittest

from src.memory_utility import MemoryUtility
from src.learning_contribution import LearningContributionEstimator
from src.future_relevance import FutureRelevanceEstimator
from src.uncertainty_reduction import UncertaintyReductionEstimator
from src.temporal_relevance import TemporalRelevanceEstimator
from src.memory_utility_pipeline import MemoryUtilityPipeline


class TestMemoryUtilityPipeline(unittest.TestCase):

    def setUp(self):

        weights = {
            "learning": 1 / 6,
            "novelty": 1 / 6,
            "future_relevance": 1 / 6,
            "uncertainty_reduction": 1 / 6,
            "temporal_relevance": 1 / 6,
            "redundancy": 1 / 6
        }

        self.memory_utility = MemoryUtility(weights)

        self.learning_estimator = LearningContributionEstimator(
            evaluator=lambda items, data: len(items) * 0.2,
            reference_gain=0.2
        )

        self.future_estimator = FutureRelevanceEstimator(
            relevance_function=lambda item, task_id: 0.8
        )

        self.uncertainty_estimator = UncertaintyReductionEstimator(
            prediction_function=lambda items, sample: (
                [0.8, 0.2] if len(items) >= 2 else [0.5, 0.5]
            )
        )

        self.temporal_estimator = TemporalRelevanceEstimator(
            validity_function=lambda item, current_time: 0.9
        )

        self.pipeline = MemoryUtilityPipeline(
            memory_utility=self.memory_utility,
            learning_estimator=self.learning_estimator,
            future_relevance_estimator=self.future_estimator,
            uncertainty_reduction_estimator=self.uncertainty_estimator,
            temporal_relevance_estimator=self.temporal_estimator
        )

        self.candidate = {
            "id": "mem_candidate",
            "item": {
                "features": [1.0, 0.0],
                "label": 1
            }
        }

        self.memory = [
            {
                "id": "mem_old",
                "item": {
                    "features": [0.0, 1.0],
                    "label": 0
                }
            }
        ]

        self.validation_data = ["validation_sample"]
        self.evaluation_data = ["evaluation_sample"]

        self.future_tasks = [
            {
                "task_id": "task_1",
                "probability": 1.0,
                "horizon_step": 1
            }
        ]

    def test_calculates_all_six_factors(self):

        result = self.pipeline.score_candidate(
            self.candidate,
            self.memory,
            self.validation_data,
            self.evaluation_data,
            self.future_tasks,
            current_time=10
        )

        factors = result["factors"]

        self.assertEqual(set(factors), {
            "learning",
            "novelty",
            "future_relevance",
            "uncertainty_reduction",
            "temporal_relevance",
            "redundancy"
        })

        for value in factors.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_returns_candidate_id(self):

        result = self.pipeline.score_candidate(
            self.candidate,
            self.memory,
            self.validation_data,
            self.evaluation_data,
            self.future_tasks,
            current_time=10
        )

        self.assertEqual(
            result["candidate_id"],
            "mem_candidate"
        )

    def test_utility_score_is_normalized(self):

        result = self.pipeline.score_candidate(
            self.candidate,
            self.memory,
            self.validation_data,
            self.evaluation_data,
            self.future_tasks,
            current_time=10
        )

        self.assertGreaterEqual(result["utility_score"], 0.0)
        self.assertLessEqual(result["utility_score"], 1.0)

    def test_rejects_duplicate_candidate_id(self):

        duplicate = {
            "id": "mem_old",
            "item": {"features": [1.0, 0.0], "label": 1}
        }

        with self.assertRaises(ValueError):
            self.pipeline.score_candidate(
                duplicate,
                self.memory,
                self.validation_data,
                self.evaluation_data,
                self.future_tasks,
                current_time=10
            )

    def test_rejects_invalid_candidate(self):

        with self.assertRaises(ValueError):
            self.pipeline.score_candidate(
                {},
                self.memory,
                self.validation_data,
                self.evaluation_data,
                self.future_tasks,
                current_time=10
            )

    def test_rejects_duplicate_memory_ids(self):

        duplicate_memory = self.memory + self.memory

        with self.assertRaises(ValueError):
            self.pipeline.score_candidate(
                self.candidate,
                duplicate_memory,
                self.validation_data,
                self.evaluation_data,
                self.future_tasks,
                current_time=10
            )

    def test_does_not_mutate_original_records(self):

        import copy

        candidate_before = copy.deepcopy(self.candidate)
        memory_before = copy.deepcopy(self.memory)

        self.pipeline.score_candidate(
            self.candidate,
            self.memory,
            self.validation_data,
            self.evaluation_data,
            self.future_tasks,
            current_time=10
        )

        self.assertEqual(self.candidate, candidate_before)
        self.assertEqual(self.memory, memory_before)


if __name__ == "__main__":
    unittest.main()