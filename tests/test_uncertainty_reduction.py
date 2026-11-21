import unittest
import math

from src.uncertainty_reduction import UncertaintyReductionEstimator


class TestUncertaintyReductionEstimator(unittest.TestCase):

    def setUp(self):
        self.candidate = {
            "id": "mem_candidate",
            "item": {"features": [1.0, 0.0], "label": 1}
        }

        self.memory = [
            {
                "id": "mem_old",
                "item": {"features": [0.0, 1.0], "label": 0}
            }
        ]

        self.evaluation = ["sample-a", "sample-b"]

    def test_entropy_reduction(self):

        def predictor(items, sample):
            return [0.9, 0.1] if len(items) == 2 else [0.5, 0.5]

        estimator = UncertaintyReductionEstimator(predictor)

        score = estimator.estimate(
            self.candidate, self.memory, self.evaluation
        )

        expected_entropy = -(
            0.9 * math.log(0.9) +
            0.1 * math.log(0.1)
        ) / math.log(2)

        self.assertAlmostEqual(score, 1.0 - expected_entropy)

    def test_average_across_samples(self):

        def predictor(items, sample):

            if len(items) == 1 and sample == "easy":
                return [0.9, 0.1]

            return [0.5, 0.5]

        estimator = UncertaintyReductionEstimator(predictor)

        score = estimator.estimate(
            self.candidate, [], ["easy", "unchanged"]
        )

        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_zero_when_uncertainty_unchanged(self):

        estimator = UncertaintyReductionEstimator(
            lambda items, sample: [0.5, 0.5]
        )

        self.assertEqual(
            estimator.estimate(
                self.candidate, self.memory, self.evaluation
            ),
            0.0
        )

    def test_zero_when_uncertainty_increases(self):

        def predictor(items, sample):
            return [0.9, 0.1] if len(items) == 0 else [0.5, 0.5]

        estimator = UncertaintyReductionEstimator(predictor)

        self.assertEqual(
            estimator.estimate(self.candidate, [], ["sample"]),
            0.0
        )

    def test_empty_memory_bank(self):

        def predictor(items, sample):
            return [0.8, 0.2] if items else [0.5, 0.5]

        estimator = UncertaintyReductionEstimator(predictor)

        score = estimator.estimate(
            self.candidate, [], ["sample"]
        )

        self.assertGreater(score, 0.0)

    def test_invalid_prediction_function(self):

        with self.assertRaises(ValueError):
            UncertaintyReductionEstimator(None)

    def test_empty_evaluation_data(self):

        estimator = UncertaintyReductionEstimator(
            lambda items, sample: [0.5, 0.5]
        )

        with self.assertRaises(ValueError):
            estimator.estimate(self.candidate, self.memory, [])

    def test_invalid_probability_output(self):

        invalid_outputs = [
            [],
            [0.5, 0.6],
            [-0.1, 1.1],
            [float("nan"), 0.0],
            [True, False],
            "invalid"
        ]

        for probabilities in invalid_outputs:

            estimator = UncertaintyReductionEstimator(
                lambda items, sample, value=probabilities: value
            )

            with self.assertRaises(ValueError):
                estimator.estimate(
                    self.candidate, [], ["sample"]
                )


if __name__ == "__main__":
    unittest.main()