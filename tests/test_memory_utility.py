import unittest

from src.memory_utility import MemoryUtility


class TestMemoryUtility(unittest.TestCase):

    def setUp(self):
        # Equal weights are a test fixture, not research defaults.
        self.weights = {
            "learning": 1 / 6,
            "novelty": 1 / 6,
            "future_relevance": 1 / 6,
            "uncertainty_reduction": 1 / 6,
            "temporal_relevance": 1 / 6,
            "redundancy": 1 / 6,
        }
        self.utility = MemoryUtility(self.weights)

    def test_calculates_utility_using_v1_1_formula(self):
        factors = {
            "learning": 1.0,
            "novelty": 0.8,
            "future_relevance": 0.6,
            "uncertainty_reduction": 0.4,
            "temporal_relevance": 0.2,
            "redundancy": 0.5,
        }
        expected = (1.0 + 0.8 + 0.6 + 0.4 + 0.2 + (1 - 0.5)) / 6
        self.assertAlmostEqual(self.utility.score(factors), expected)

    def test_score_is_between_zero_and_one(self):
        all_zero = {name: 0.0 for name in self.weights}
        all_one = {name: 1.0 for name in self.weights}
        self.assertAlmostEqual(self.utility.score(all_zero), 1 / 6)
        self.assertAlmostEqual(self.utility.score(all_one), 5 / 6)

    def test_zero_redundancy_contributes_full_redundancy_weight(self):
        factors = {name: 0.0 for name in self.weights}
        self.assertAlmostEqual(self.utility.score(factors), self.weights["redundancy"])

    def test_full_redundancy_contributes_zero_redundancy_weight(self):
        factors = {name: 0.0 for name in self.weights}
        factors["redundancy"] = 1.0
        self.assertAlmostEqual(self.utility.score(factors), 0.0)

    def test_rejects_missing_factor(self):
        factors = {name: 0.5 for name in self.weights}
        factors.pop("novelty")
        with self.assertRaises(ValueError):
            self.utility.score(factors)

    def test_rejects_unknown_factor(self):
        factors = {name: 0.5 for name in self.weights}
        factors["unknown"] = 0.5
        with self.assertRaises(ValueError):
            self.utility.score(factors)

    def test_rejects_factor_outside_zero_to_one(self):
        for invalid_value in (-0.01, 1.01):
            factors = {name: 0.5 for name in self.weights}
            factors["learning"] = invalid_value
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    self.utility.score(factors)

    def test_rejects_non_numeric_or_boolean_factor(self):
        for invalid_value in ("high", True, None):
            factors = {name: 0.5 for name in self.weights}
            factors["learning"] = invalid_value
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    self.utility.score(factors)

    def test_rejects_invalid_weights(self):
        invalid_weights = [
            {**self.weights, "learning": -0.1},
            {**self.weights, "learning": float("nan")},
            {**self.weights, "learning": float("inf")},
            {**self.weights, "learning": 0.5},
        ]
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    MemoryUtility(weights)

    def test_rejects_missing_or_extra_weight_names(self):
        missing = dict(self.weights)
        missing.pop("learning")
        extra = {**self.weights, "other": 0.0}
        with self.assertRaises(ValueError):
            MemoryUtility(missing)
        with self.assertRaises(ValueError):
            MemoryUtility(extra)


if __name__ == "__main__":
    unittest.main()
