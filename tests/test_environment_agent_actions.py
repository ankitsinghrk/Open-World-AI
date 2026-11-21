import unittest
from unittest.mock import Mock

from src.environment.environment import Environment
from src.action_manager import ActionManager
from src.memory_manager import MemoryManager
from src.resource_manager import ResourceManager


class TestEnvironmentAgentActions(unittest.TestCase):

    def setUp(self):
        self.memory_manager = MemoryManager(capacity=3)
        self.resource_manager = ResourceManager(
            label_budget=10,
            compute_budget=1000
        )

        self.action_manager = ActionManager()

        self.environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=self.memory_manager,
            action_manager=self.action_manager,
            resource_manager=self.resource_manager
        )

        self.item = {
            "features": [1.0, 0.0],
            "label": 1
        }

    def test_store_action(self):

        action = self.action_manager.get_action(5)

        result = self.environment.execute_agent_action(
            action=action,
            item=self.item
        )

        self.assertEqual(result["memory_operation"], "STORE")
        self.assertEqual(self.memory_manager.get_memory_status()["used"], 1)
        self.assertEqual(result["remaining_compute"], 1000)

    def test_discard_action(self):

        self.memory_manager.store(self.item)

        action = self.action_manager.get_action(1)

        result = self.environment.execute_agent_action(
            action=action,
            item=self.item,
        )

        self.assertEqual(result["memory_operation"], "DISCARD")
        self.assertIsNone(result["memory_result"])
        self.assertEqual(self.memory_manager.get_memory_status()["used"], 1)

    def test_replace_action(self):

        self.memory_manager.store(self.item)

        new_item = {
            "features": [0.0, 1.0],
            "label": 0
        }

        action = self.action_manager.get_action(9)

        result = self.environment.execute_agent_action(
            action=action,
            item=new_item,
            index=0
        )

        self.assertEqual(result["memory_operation"], "REPLACE")
        self.assertEqual(self.memory_manager.get_memory_status()["used"], 1)

    def test_rejects_invalid_action(self):

        with self.assertRaises(ValueError):
            self.environment.execute_agent_action(
                action={"id": 100}
            )

    def test_rejects_store_without_item(self):

        action = self.action_manager.get_action(5)

        with self.assertRaises(ValueError):
            self.environment.execute_agent_action(
                action=action
            )


    def test_rejects_label_action_when_label_budget_is_zero(self):

        resource_manager = ResourceManager(
            label_budget=0,
            compute_budget=1000
        )

        environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=self.memory_manager,
            action_manager=self.action_manager,
            resource_manager=resource_manager
        )

        action = self.action_manager.get_action(17)

        with self.assertRaises(ValueError):
            environment.execute_agent_action(
                action=action,
                item={"features": [1.0, 0.0]},
                sample_id="sample_001",
                label_provider=lambda sample_id: 1
            )

        self.assertEqual(
            resource_manager.get_resources(),
            {"labels": 0, "compute": 1000}
        )

        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            0
        )

    def test_rejects_action_when_compute_budget_is_insufficient(self):

        resource_manager = ResourceManager(
            label_budget=10,
            compute_budget=99
        )

        environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=self.memory_manager,
            action_manager=self.action_manager,
            resource_manager=resource_manager
        )

        action = self.action_manager.get_action(8)

        with self.assertRaises(ValueError):
            environment.execute_agent_action(
                action=action,
                item={"features": [1.0, 0.0]}
            )

        self.assertEqual(
            resource_manager.get_resources(),
            {"labels": 10, "compute": 99}
        )

        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            0
        )


    def test_action_succeeds_when_compute_budget_exactly_matches_cost(self):

        resource_manager = ResourceManager(
            label_budget=10,
            compute_budget=250
        )

        environment = Environment(
            feature_extractor=Mock(),
            state_builder=Mock(),
            state_encoder=Mock(),
            memory_manager=self.memory_manager,
            action_manager=self.action_manager,
            resource_manager=resource_manager
        )

        action = self.action_manager.get_action(6)

        result = environment.execute_agent_action(
            action=action,
            item=self.item
        )

        self.assertEqual(result["action_id"], 6)
        self.assertEqual(
            resource_manager.get_resources(),
            {"labels": 10, "compute": 0}
        )
        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            1
        )




    def test_memory_failure_does_not_consume_label_budget(self):

        label_provider = Mock()
        label_provider.get_label.return_value = 1

        self.memory_manager.store = Mock(
            side_effect=RuntimeError("Simulated memory failure")
        )

        action = self.action_manager.get_action(17)

        with self.assertRaises(RuntimeError):
            self.environment.execute_agent_action(
                action=action,
                item={"features": [1.0, 0.0]},
                sample_id="sample_001",
                label_provider=label_provider
            )

        self.assertEqual(
            self.resource_manager.get_resources(),
            {"labels": 10, "compute": 1000}
        )

        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            0
        )


    def test_compute_failure_does_not_change_memory(self):

        self.resource_manager.consume_compute = Mock(
            side_effect=RuntimeError("Simulated compute failure")
        )

        action = self.action_manager.get_action(5)

        with self.assertRaises(RuntimeError):
            self.environment.execute_agent_action(
                action=action,
                item=self.item
            )

        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            0
        )

        self.assertEqual(
            self.resource_manager.get_resources(),
            {"labels": 10, "compute": 1000}
        )





if __name__ == "__main__":
    unittest.main()