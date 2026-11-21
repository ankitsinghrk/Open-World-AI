import unittest

from src.environment.environment import Environment
from src.memory_manager import MemoryManager


class TestEnvironmentMemoryOperations(unittest.TestCase):

    def setUp(self):
        self.memory_manager = MemoryManager(capacity=3)
        self.environment = Environment(
            feature_extractor=None,
            state_builder=None,
            state_encoder=None,
            memory_manager=self.memory_manager,
            action_manager=None
        )

    def test_environment_store_assigns_memory_id(self):
        item = {"features": [0.25, 0.64, 0.81], "label": 1}

        result = self.environment.execute_memory_operation(
            operation="STORE",
            item=item
        )

        self.assertTrue(result)
        self.assertEqual(self.memory_manager.get_memory_id(0), "mem_000001")
        self.assertEqual(self.memory_manager.memory, [item])

    def test_environment_discard_keeps_remaining_item_id(self):
        self.memory_manager.store("first")
        self.memory_manager.store("second")
        second_id = self.memory_manager.get_memory_id(1)

        removed = self.environment.execute_memory_operation(
            operation="DISCARD",
            index=0
        )

        self.assertEqual(removed, "first")
        self.assertEqual(self.memory_manager.get_memory_id(0), second_id)

    def test_environment_replace_assigns_new_memory_id(self):
        self.memory_manager.store("old")
        old_id = self.memory_manager.get_memory_id(0)

        returned = self.environment.execute_memory_operation(
            operation="REPLACE",
            item="new",
            index=0
        )

        self.assertEqual(returned, "old")
        self.assertNotEqual(self.memory_manager.get_memory_id(0), old_id)
        self.assertEqual(self.memory_manager.memory, ["new"])


if __name__ == "__main__":
    unittest.main()
