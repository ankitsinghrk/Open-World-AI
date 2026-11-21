import unittest

from src.memory_factor_calculator import MemoryFactorCalculator
from src.memory_utility import MemoryUtility


class TestMemoryFactorCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = MemoryFactorCalculator()

    def record(self, memory_id, features):
        return {"id": memory_id, "item": {"features": features, "label": 1}}

    def test_calculates_novelty_and_mean_redundancy(self):
        bank = [
            self.record("mem_1", [1, 0]),
            self.record("mem_2", [0, 1]),
        ]
        result = self.calculator.calculate([1, 0], bank)
        self.assertAlmostEqual(result["novelty"], 0.0)
        self.assertAlmostEqual(result["redundancy"], 0.75)

    def test_empty_bank_defaults(self):
        result = self.calculator.calculate([1, 0], [])
        self.assertEqual(result, {"novelty": 1.0, "redundancy": 0.0})

    def test_rejects_bad_record(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate([1, 0], [{"id": "mem_1"}])

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                [1, 0],
                [self.record("mem_1", [1, 0, 1])]
            )


class TestMemoryUtilityIntegration(unittest.TestCase):

    def setUp(self):
        self.weights = {
            "learning": 1 / 6,
            "novelty": 1 / 6,
            "future_relevance": 1 / 6,
            "uncertainty_reduction": 1 / 6,
            "temporal_relevance": 1 / 6,
            "redundancy": 1 / 6,
        }
        self.utility = MemoryUtility(self.weights)

    def test_score_candidate_integrates_calculated_factors(self):
        records = [
            {"id": "mem_1", "item": {"features": [1, 0], "label": 1}},
            {"id": "mem_2", "item": {"features": [0, 1], "label": 0}},
        ]
        other = {
            "learning": 0.8,
            "future_relevance": 0.7,
            "uncertainty_reduction": 0.6,
            "temporal_relevance": 0.5,
        }
        # Candidate [1,0]: normalized similarities are 1.0 and 0.5.
        # Thus novelty=0, redundancy=0.75.
        expected = (
            0.8 + 0.0 + 0.7 + 0.6 + 0.5 + (1.0 - 0.75)
        ) / 6
        self.assertAlmostEqual(
            self.utility.score_candidate([1, 0], records, other),
            expected
        )

    def test_score_candidate_with_empty_bank(self):
        other = {
            "learning": 0.5,
            "future_relevance": 0.5,
            "uncertainty_reduction": 0.5,
            "temporal_relevance": 0.5,
        }
        score = self.utility.score_candidate([1, 0], [], other)
        expected = (0.5 + 1.0 + 0.5 + 0.5 + 0.5 + 1.0) / 6
        self.assertAlmostEqual(score, expected)

    def test_rejects_missing_other_factor(self):
        with self.assertRaises(ValueError):
            self.utility.score_candidate([1, 0], [], {"learning": 0.5})


if __name__ == "__main__":
    unittest.main()
