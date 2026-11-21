import unittest
from src.future_relevance import FutureRelevanceEstimator


class TestFutureRelevanceEstimator(unittest.TestCase):
    def setUp(self):
        self.tasks = [
            {"task_id": "classification", "probability": 0.7, "horizon_step": 1},
            {"task_id": "detection", "probability": 0.3, "horizon_step": 2},
        ]

    def test_probability_weighted_relevance(self):
        scores = {"classification": 1.0, "detection": 0.0}
        estimator = FutureRelevanceEstimator(lambda item, task: scores[task])
        self.assertAlmostEqual(estimator.estimate({"x": 1}, self.tasks), 0.7)

    def test_later_tasks_are_discounted(self):
        scores = {"classification": 1.0, "detection": 0.0}
        estimator = FutureRelevanceEstimator(
            lambda item, task: scores[task], horizon_decay=0.5
        )
        expected = 0.7 / (0.7 + 0.3 * 0.5)
        self.assertAlmostEqual(estimator.estimate({}, self.tasks), expected)

    def test_equal_decay_preserves_probability_weighting(self):
        scores = {"classification": 0.2, "detection": 0.8}
        estimator = FutureRelevanceEstimator(lambda item, task: scores[task])
        self.assertAlmostEqual(estimator.estimate({}, self.tasks), 0.38)

    def test_zero_probability_tasks_are_ignored(self):
        tasks = [
            {"task_id": "unused", "probability": 0.0, "horizon_step": 1},
            {"task_id": "used", "probability": 1.0, "horizon_step": 2},
        ]
        estimator = FutureRelevanceEstimator(
            lambda item, task: 0.6 if task == "used" else 1.0
        )
        self.assertAlmostEqual(estimator.estimate({}, tasks), 0.6)

    def test_rejects_empty_memory_item(self):
        estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
        with self.assertRaises(ValueError):
            estimator.estimate(None, self.tasks)

    def test_rejects_empty_tasks(self):
        estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
        with self.assertRaises(ValueError):
            estimator.estimate({}, [])

    def test_rejects_probabilities_not_summing_to_one(self):
        tasks = [{"task_id": "a", "probability": 0.4, "horizon_step": 1}]
        estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
        with self.assertRaises(ValueError):
            estimator.estimate({}, tasks)

    def test_rejects_invalid_probability(self):
        for probability in (-0.1, 1.1, float("nan"), True, "0.5"):
            tasks = [{"task_id": "a", "probability": probability, "horizon_step": 1}]
            estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    estimator.estimate({}, tasks)

    def test_rejects_invalid_horizon(self):
        for step in (0, -1, 1.5, True):
            tasks = [{"task_id": "a", "probability": 1.0, "horizon_step": step}]
            estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    estimator.estimate({}, tasks)

    def test_rejects_invalid_relevance_output(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            estimator = FutureRelevanceEstimator(
                lambda item, task, result=value: result
            )
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    estimator.estimate({}, self.tasks)

    def test_rejects_invalid_decay(self):
        for decay in (0, -0.1, 1.1, float("nan"), True, "0.5"):
            with self.subTest(decay=decay):
                with self.assertRaises(ValueError):
                    FutureRelevanceEstimator(lambda item, task: 0.5, decay)

    def test_rejects_non_callable_scorer(self):
        with self.assertRaises(ValueError):
            FutureRelevanceEstimator(None)

    def test_rejects_duplicate_task_ids(self):
        tasks = [
            {"task_id": "a", "probability": 0.5, "horizon_step": 1},
            {"task_id": "a", "probability": 0.5, "horizon_step": 2},
        ]
        estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
        with self.assertRaises(ValueError):
            estimator.estimate({}, tasks)

    def test_does_not_mutate_memory_item(self):
        item = {"features": [1, 2]}
        estimator = FutureRelevanceEstimator(
            lambda memory, task: memory["features"].append(99) or 0.5
        )
        estimator.estimate(item, self.tasks)
        self.assertEqual(item, {"features": [1, 2]})

    def test_rejects_malformed_task(self):
        estimator = FutureRelevanceEstimator(lambda item, task: 0.5)
        with self.assertRaises(ValueError):
            estimator.estimate({}, [{"task_id": "a", "probability": 1.0}])
