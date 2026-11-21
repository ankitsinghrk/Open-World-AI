import unittest

from src.temporal_relevance import TemporalRelevanceEstimator


class TestTemporalRelevanceEstimator(unittest.TestCase):

    def setUp(self):
        self.record = {
            "id": "mem_001",
            "item": {"features": [1.0, 0.0], "label": 1}
        }

    def test_valid_memory_probability(self):
        estimator = TemporalRelevanceEstimator(
            lambda item, time: 0.8
        )

        self.assertEqual(
            estimator.estimate(self.record, 10),
            0.8
        )

    def test_zero_probability(self):
        estimator = TemporalRelevanceEstimator(
            lambda item, time: 0.0
        )

        self.assertEqual(
            estimator.estimate(self.record, 10),
            0.0
        )

    def test_full_probability(self):
        estimator = TemporalRelevanceEstimator(
            lambda item, time: 1.0
        )

        self.assertEqual(
            estimator.estimate(self.record, 10),
            1.0
        )

    def test_different_times(self):
        estimator = TemporalRelevanceEstimator(
            lambda item, time: 0.9 if time == 1 else 0.4
        )

        self.assertEqual(
            estimator.estimate(self.record, 1),
            0.9
        )

        self.assertEqual(
            estimator.estimate(self.record, 5),
            0.4
        )

    def test_invalid_function(self):
        with self.assertRaises(ValueError):
            TemporalRelevanceEstimator(None)

    def test_invalid_memory_record(self):
        estimator = TemporalRelevanceEstimator(
            lambda item, time: 0.5
        )

        with self.assertRaises(ValueError):
            estimator.estimate({}, 10)

    def test_invalid_probability(self):
        invalid_values = [
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            True,
            "invalid"
        ]

        for value in invalid_values:
            estimator = TemporalRelevanceEstimator(
                lambda item, time, result=value: result
            )

            with self.assertRaises(ValueError):
                estimator.estimate(self.record, 10)


if __name__ == "__main__":
    unittest.main()