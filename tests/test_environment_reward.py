import unittest

from src.environment.environment import Environment
from src.reward_calculator import RewardCalculator


class TestEnvironmentReward(unittest.TestCase):

    def setUp(self):
        self.reward_calculator = RewardCalculator(
            forgetting_weight=1.0,
            label_cost_weight=1.0,
            compute_cost_weight=1.0
        )

        self.environment = Environment(
            feature_extractor=None,
            state_builder=None,
            state_encoder=None,
            memory_manager=None,
            action_manager=None,
            reward_calculator=self.reward_calculator
        )

    def test_environment_calculates_reward(self):

        result = self.environment.calculate_transition_reward(
            performance_before=0.60,
            performance_after=0.80,
            labels_used=1,
            compute_used=100,
            initial_label_budget=100,
            initial_compute_budget=1000
        )

        self.assertAlmostEqual(result["reward"], 0.09)

    def test_reward_calculator_not_connected(self):

        environment = Environment(
            None, None, None, None, None
        )

        with self.assertRaises(ValueError):
            environment.calculate_transition_reward(
                0.5, 0.6, 0, 0, 100, 1000
            )


if __name__ == "__main__":
    unittest.main()