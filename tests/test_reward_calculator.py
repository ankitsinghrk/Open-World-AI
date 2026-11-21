
import unittest

from src.reward_calculator import RewardCalculator


class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = RewardCalculator(
            forgetting_weight=1.0,
            label_cost_weight=1.0,
            compute_cost_weight=1.0
        )

    def test_reward_with_learning_improvement(self):
        result = self.calculator.calculate(
            performance_before=0.60,
            performance_after=0.80,
            labels_used=1,
            compute_used=100,
            initial_label_budget=100,
            initial_compute_budget=1000
        )

        self.assertAlmostEqual(
            result["reward"], 0.20 - 0.01 - 0.10
        )

    def test_forgetting_penalty(self):
        result = self.calculator.calculate(
            performance_before=0.60,
            performance_after=0.80,
            previous_task_score_before=0.90,
            previous_task_score_after=0.70,
            labels_used=0,
            compute_used=0,
            initial_label_budget=100,
            initial_compute_budget=1000
        )

        self.assertAlmostEqual(result["forgetting"], 0.20)
        self.assertAlmostEqual(result["reward"], 0.00)

    def test_no_forgetting_when_old_performance_improves(self):
        result = self.calculator.calculate(
            performance_before=0.50,
            performance_after=0.60,
            previous_task_score_before=0.70,
            previous_task_score_after=0.80,
            labels_used=0,
            compute_used=0,
            initial_label_budget=100,
            initial_compute_budget=1000
        )

        self.assertEqual(result["forgetting"], 0.0)

    def test_negative_learning_gain(self):
        result = self.calculator.calculate(
            performance_before=0.80,
            performance_after=0.70,
            labels_used=0,
            compute_used=0,
            initial_label_budget=100,
            initial_compute_budget=1000
        )

        self.assertAlmostEqual(result["reward"], -0.10)

    def test_invalid_performance(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                performance_before=1.5,
                performance_after=0.8,
                labels_used=0,
                compute_used=0,
                initial_label_budget=100,
                initial_compute_budget=1000
            )

    def test_invalid_label_usage(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                performance_before=0.5,
                performance_after=0.6,
                labels_used=101,
                compute_used=0,
                initial_label_budget=100,
                initial_compute_budget=1000
            )

    def test_v1_reward_is_utility_change_minus_operation_cost(self):
        calculator = RewardCalculator(
            forgetting_weight=1.0,
            label_cost_weight=0.2,
            compute_cost_weight=0.01,
            memory_cost_weight=0.5,
        )

        result = calculator.calculate_v1(
            utility_before=0.40,
            utility_after=0.55,
            label_action=1,
            memory_operation_cost=1,
            compute_cost=250,
            label_cost=1,
        )

        expected_cost = 0.2 * 1 * 1 + 0.5 * 1 + 0.01 * 250
        self.assertAlmostEqual(result["operation_cost"], expected_cost)
        self.assertAlmostEqual(result["reward"], 0.15 - expected_cost)

    def test_previous_task_scores_must_be_paired(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate(
                performance_before=0.5,
                performance_after=0.6,
                previous_task_score_before=0.8,
                labels_used=0,
                compute_used=0,
                initial_label_budget=100,
                initial_compute_budget=1000
            )


if __name__ == "__main__":
    unittest.main()
