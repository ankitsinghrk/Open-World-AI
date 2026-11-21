import unittest
from unittest.mock import Mock

from src.environment.environment import Environment


class TestEnvironmentUtilityPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = Mock()

        self.environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=Mock(),
            action_manager=Mock(),
            memory_utility_pipeline=self.pipeline
        )

        self.candidate = {
            "id": "mem_candidate",
            "item": {
                "features": [1.0, 0.0],
                "label": 1
            }
        }

        self.memory_records = []
        self.validation_data = ["validation_sample"]
        self.evaluation_data = ["evaluation_sample"]

        self.future_tasks = [
            {
                "task_id": "task_1",
                "probability": 1.0,
                "horizon_step": 1
            }
        ]

    def test_delegates_candidate_scoring(self):

        expected_result = {
            "candidate_id": "mem_candidate",
            "utility_score": 0.75,
            "factors": {}
        }

        self.pipeline.score_candidate.return_value = expected_result

        result = self.environment.evaluate_memory_candidate(
            self.candidate,
            self.memory_records,
            self.validation_data,
            self.evaluation_data,
            self.future_tasks,
            current_time=10
        )

        self.assertEqual(result, expected_result)

        self.pipeline.score_candidate.assert_called_once_with(
            candidate_record=self.candidate,
            memory_records=self.memory_records,
            validation_data=self.validation_data,
            evaluation_data=self.evaluation_data,
            future_tasks=self.future_tasks,
            current_time=10
        )

    def test_rejects_missing_pipeline(self):

        environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=Mock(),
            action_manager=Mock()
        )

        with self.assertRaises(ValueError):
            environment.evaluate_memory_candidate(
                self.candidate,
                self.memory_records,
                self.validation_data,
                self.evaluation_data,
                self.future_tasks,
                current_time=10
            )


if __name__ == "__main__":
    unittest.main()