import unittest
from unittest.mock import Mock

from src.dqn_training_loop import DQNTrainingLoop
from src.replay_buffer import ReplayBuffer


class TestDQNTrainingIntegration(unittest.TestCase):

    def setUp(self):
        self.agent = Mock()
        self.environment = Mock()
        self.replay_buffer = ReplayBuffer(capacity=10)
        self.trainer = Mock()

        self.agent.select_action.return_value = 2

        self.environment.decode_agent_action.return_value = {
             "id": 3,
    "label": 0,
    "memory": "STORE",
    "compute": 0
         }


        self.environment.execute_agent_action.return_value = {
            "success": True
        }

        self.environment.calculate_transition_reward.return_value = {
            "reward": 0.5
        }

        self.environment.store_dqn_experience.side_effect = (
            self.store_experience
        )

        self.trainer.agent = self.agent
        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.training_steps = 1
        self.trainer.train.return_value = 0.25

        self.loop = DQNTrainingLoop(
            self.agent,
            self.environment,
            self.replay_buffer,
            self.trainer
        )

    def store_experience(self, **kwargs):
        kwargs["replay_buffer"].add(
            kwargs["state"],
            kwargs["action_index"],
            kwargs["reward_result"]["reward"],
            kwargs["next_state"],
            kwargs["done"],
            kwargs["next_feasible_mask"]
        )

    def test_complete_action_flow(self):
        state = [0.1, 0.2]
        mask = [True] * 24

        selected = self.loop.select_action(state, mask)

        result = self.loop.execute_action(
            selected,
            item={"features": [0.3, 0.4]}
        )

        self.assertEqual(selected["action_index"], 2)
        self.assertTrue(result["success"])

    def test_complete_learning_flow(self):
        result = self.loop.record_transition(
            state=[0.1, 0.2],
            action_index=2,
            reward_inputs={"performance_before": 0.5},
            next_state=[0.2, 0.3],
            done=False,
            next_feasible_mask=[True] * 24
        )

        self.assertEqual(len(self.replay_buffer), 1)
        self.assertEqual(result["reward"], 0.5)
        self.assertEqual(result["loss"], 0.25)
        self.trainer.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()