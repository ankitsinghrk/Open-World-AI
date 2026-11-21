
import unittest

from src.environment.environment import Environment
from src.replay_buffer import ReplayBuffer


class TestEnvironmentExperience(unittest.TestCase):

    def setUp(self):
        self.environment = Environment(
            None, None, None, None, None
        )

        self.buffer = ReplayBuffer(capacity=100)

    def test_experience_is_stored(self):

        reward_result = {"reward": 0.25}

        result = self.environment.store_dqn_experience(
            replay_buffer=self.buffer,
            state=[0.1, 0.2],
            action_index=3,
            reward_result=reward_result,
            next_state=[0.3, 0.4],
            done=False,
            next_feasible_mask=[True] + [False] * 23
        )

        self.assertEqual(result, 1)
        self.assertEqual(len(self.buffer), 1)

    def test_invalid_action_index(self):

        with self.assertRaises(ValueError):
            self.environment.store_dqn_experience(
                self.buffer,
                [0.1, 0.2],
                24,
                {"reward": 0.1},
                [0.3, 0.4],
                False,
                [True] * 24
            )

    def test_terminal_experience(self):

        self.environment.store_dqn_experience(
            self.buffer,
            [0.1, 0.2],
            2,
            {"reward": 0.5},
            [0.3, 0.4],
            True,
            [False] * 24
        )

        self.assertEqual(len(self.buffer), 1)

    def test_non_terminal_state_requires_feasible_action(self):

        with self.assertRaises(ValueError):
            self.environment.store_dqn_experience(
                self.buffer,
                [0.1, 0.2],
                2,
                {"reward": 0.5},
                [0.3, 0.4],
                False,
                [False] * 24
            )


if __name__ == "__main__":
    unittest.main()
