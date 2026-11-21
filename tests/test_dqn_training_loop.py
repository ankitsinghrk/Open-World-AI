
import unittest
from unittest.mock import Mock

from src.dqn_training_loop import DQNTrainingLoop


class TestDQNTrainingLoop(unittest.TestCase):

    def setUp(self):

        self.agent = Mock()
        self.agent.select_action.return_value = 2

        self.environment = Mock()

        self.environment.decode_agent_action.return_value = {
            "id": 3,
            "label": 0,
            "memory": "STORE",
            "compute": 0
        }

        self.environment.calculate_transition_reward.return_value = {
            "reward": 0.25,
            "performance_gain": 0.3,
            "forgetting": 0.0,
            "label_cost": 0.01,
            "compute_cost": 0.04
        }

        self.buffer = Mock()
        self.buffer.__len__ = Mock(return_value=5)

        self.trainer = Mock()
        self.trainer.agent = self.agent
        self.trainer.replay_buffer = self.buffer
        self.trainer.train.return_value = 0.5
        self.trainer.training_steps = 1

        self.loop = DQNTrainingLoop(
            self.agent,
            self.environment,
            self.buffer,
            self.trainer
        )

    def test_action_selection(self):

        mask = [True, True, True] + [False] * 21

        result = self.loop.select_action(
            [0.1, 0.2],
            mask,
            epsilon=0.1
        )

        self.assertEqual(result["action_index"], 2)
        self.environment.decode_agent_action.assert_called_once()

    def test_action_execution(self):

        selected = {
            "action_index": 2,
            "action": {"id": 3}
        }

        self.loop.execute_action(
            selected,
            item={"features": [0.1, 0.2]}
        )

        self.environment.execute_agent_action.assert_called_once()

    def test_transition_recording(self):

        result = self.loop.record_transition(
            state=[0.1, 0.2],
            action_index=2,
            reward_inputs={"performance_before": 0.5},
            next_state=[0.2, 0.3],
            done=False,
            next_feasible_mask=[True] + [False] * 23
        )

        self.environment.store_dqn_experience.assert_called_once()
        self.trainer.train.assert_called_once()

        self.assertEqual(result["reward"], 0.25)
        self.assertEqual(result["loss"], 0.5)

    def test_invalid_selected_action(self):

        with self.assertRaises(ValueError):
            self.loop.execute_action(
                {"action_index": 2},
                item={}
            )


    def test_causal_observation_order(self):
        events = []

        self.environment.build_agent_state_with_prediction.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        mask = [False, False, True] + [False] * 21
        self.environment.build_action_mask.side_effect = [mask, mask]

        self.environment.execute_agent_action.return_value = {
            "action_id": 3,
            "memory_operation": "STORE",
            "memory_result": {},
            "acquired_label": None,
            "remaining_compute": 1000,
        }

        self.environment.calculate_v1_reward.return_value = {
            "reward": 0.2
        }

        def observe():
            events.append("observe")
            return {
                "observation": [1.0, 2.0],
                "novelty": 0.1,
                "resources": {},
                "history": [],
            }

        utilities = iter([0.4, 0.6])

        def validation_utility(context):
            events.append("validation")
            return next(utilities)

        def apply_action(action, execution, context):
            events.append("apply_action")

        def execute_and_record(*args, **kwargs):
            events.append("execute")
            return {
                "action_id": 3,
                "memory_operation": "STORE",
                "memory_result": {},
                "acquired_label": None,
                "remaining_compute": 1000,
            }

        self.environment.execute_agent_action.side_effect = execute_and_record

        self.environment.build_agent_state_with_prediction.side_effect = (
            lambda **kwargs: (
                events.append("build_state"),
                [0.1, 0.2] if events.count("build_state") == 1
                else [0.3, 0.4]
            )[1]
        )

        result = self.loop.step_with_observations(
            observe_fn=observe,
            validation_utility_fn=validation_utility,
            apply_action_fn=apply_action,
            epsilon=0.0,
        )

        self.assertEqual(result["utility_before"], 0.4)
        self.assertEqual(result["utility_after"], 0.6)
        self.assertEqual(result["reward"], 0.2)

        self.assertLess(events.index("execute"), events.index("apply_action"))
        self.assertLess(
            events.index("apply_action"),
            len(events) - 1 - events[::-1].index("observe")
        )

        self.assertEqual(events.count("observe"), 2)
        self.assertEqual(events.count("validation"), 2)
        self.environment.store_dqn_experience.assert_called_once()
        self.trainer.train.assert_called_once()



if __name__ == "__main__":
    unittest.main()
