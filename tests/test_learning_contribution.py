import unittest

from src.learning_contribution import LearningContributionEstimator


class TestLearningContributionEstimator(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"id": "mem_000001", "item": {"value": 2}},
            {"id": "mem_000002", "item": {"value": 3}},
        ]
        self.validation_data = ["sample-a", "sample-b"]

    def test_normalizes_leave_one_out_performance_gain(self):
        def evaluator(items, validation):
            return sum(item["value"] for item in items) / 10

        estimator = LearningContributionEstimator(evaluator, reference_gain=0.5)
        # Removing item 2 reduces score by 0.2; 0.2 / 0.5 = 0.4.
        self.assertAlmostEqual(
            estimator.estimate("mem_000001", self.records, self.validation_data),
            0.4,
        )

    def test_zero_when_removal_does_not_reduce_performance(self):
        estimator = LearningContributionEstimator(
            lambda items, data: 0.8, reference_gain=0.2
        )
        self.assertEqual(
            estimator.estimate("mem_000001", self.records, self.validation_data),
            0.0,
        )

    def test_zero_when_removal_improves_performance(self):
        def evaluator(items, data):
            return 0.6 if len(items) == 2 else 0.8

        estimator = LearningContributionEstimator(evaluator, reference_gain=0.2)
        self.assertEqual(
            estimator.estimate("mem_000001", self.records, self.validation_data),
            0.0,
        )

    def test_caps_normalized_score_at_one(self):
        def evaluator(items, data):
            return 1.0 if len(items) == 2 else 0.0

        estimator = LearningContributionEstimator(evaluator, reference_gain=0.1)
        self.assertEqual(
            estimator.estimate("mem_000001", self.records, self.validation_data),
            1.0,
        )

    def test_uses_only_target_removal_for_ablation(self):
        observed_lengths = []

        def evaluator(items, data):
            observed_lengths.append(len(items))
            return len(items) / 2

        estimator = LearningContributionEstimator(evaluator, reference_gain=1.0)
        estimator.estimate("mem_000002", self.records, self.validation_data)
        self.assertEqual(observed_lengths, [2, 1])

    def test_rejects_unknown_memory_id(self):
        estimator = LearningContributionEstimator(
            lambda items, data: 0.5, reference_gain=0.2
        )
        with self.assertRaises(KeyError):
            estimator.estimate("mem_999999", self.records, self.validation_data)

    def test_rejects_duplicate_memory_ids(self):
        records = [
            {"id": "same", "item": 1},
            {"id": "same", "item": 2},
        ]
        estimator = LearningContributionEstimator(
            lambda items, data: 0.5, reference_gain=0.2
        )
        with self.assertRaises(ValueError):
            estimator.estimate("same", records, self.validation_data)

    def test_rejects_invalid_evaluator_scores(self):
        for bad_score in (None, float("nan"), float("inf"), True, "0.5"):
            estimator = LearningContributionEstimator(
                lambda items, data, value=bad_score: value,
                reference_gain=0.2,
            )
            with self.subTest(score=bad_score):
                with self.assertRaises(ValueError):
                    estimator.estimate(
                        "mem_000001", self.records, self.validation_data
                    )

    def test_rejects_invalid_reference_gain(self):
        for value in (0, -1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LearningContributionEstimator(
                        lambda items, data: 0.5, reference_gain=value
                    )

    def test_rejects_non_callable_evaluator(self):
        with self.assertRaises(ValueError):
            LearningContributionEstimator(evaluator=None, reference_gain=0.2)

    def test_does_not_mutate_original_records_or_validation_data(self):
        original_records = [
            {"id": record["id"], "item": dict(record["item"])}
            for record in self.records
        ]
        original_validation = list(self.validation_data)

        def evaluator(items, data):
            if items:
                items[0]["value"] = 999
            data.append("changed")
            return 0.7

        estimator = LearningContributionEstimator(evaluator, reference_gain=0.2)
        estimator.estimate("mem_000001", self.records, self.validation_data)

        self.assertEqual(self.records, original_records)
        self.assertEqual(self.validation_data, original_validation)

    def test_accepts_empty_validation_data(self):
        estimator = LearningContributionEstimator(
            lambda items, data: 0.6, reference_gain=0.2
        )
        self.assertEqual(
            estimator.estimate("mem_000001", self.records, []),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
