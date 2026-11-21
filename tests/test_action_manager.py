import unittest

from src.action_manager import ActionManager


class TestActionManager(unittest.TestCase):

    def setUp(self):
        self.manager = ActionManager()

    def test_generates_24_actions(self):
        self.assertEqual(len(self.manager.generate_actions()), 24)

    def test_network_index_mapping(self):
        for action_id in range(1, 25):
            self.assertEqual(
                self.manager.get_network_index(action_id),
                action_id - 1
            )

    def test_action_id_mapping(self):
        for index in range(24):
            self.assertEqual(
                self.manager.get_action_id(index),
                index + 1
            )

    def test_get_action(self):
        action = self.manager.get_action(1)
        self.assertEqual(action["id"], 1)

    def _costs(self):
        return {0: 0, 1: 250, 2: 500, 3: 1000}

    def test_label_and_discard_is_feasible(self):
        actions = self.manager.get_feasible_actions(
            label_budget=10,
            compute_budget=1000,
            memory_used=5,
            memory_capacity=10,
            compute_costs=self._costs(),
        )

        labelled_discard = [
            action for action in actions
            if action["label"] == 1 and action["memory"] == "DISCARD"
        ]

        self.assertEqual(len(labelled_discard), 4)

    def test_discard_is_feasible_when_memory_is_empty(self):
        actions = self.manager.get_feasible_actions(
            label_budget=0,
            compute_budget=0,
            memory_used=0,
            memory_capacity=10,
            compute_costs=self._costs(),
        )

        keys = [
            (action["label"], action["memory"], action["compute"])
            for action in actions
        ]
        self.assertIn((0, "DISCARD", 0), keys)
        self.assertIn((0, "STORE", 0), keys)
        self.assertTrue(all(label == 0 and compute == 0 for label, _, compute in keys))
        self.assertFalse(any(memory == "REPLACE" for _, memory, _ in keys))

    def test_store_is_blocked_when_memory_is_full(self):
        actions = self.manager.get_feasible_actions(
            label_budget=10,
            compute_budget=1000,
            memory_used=10,
            memory_capacity=10,
            compute_costs=self._costs(),
        )

        self.assertFalse(any(action["memory"] == "STORE" for action in actions))
        self.assertTrue(any(action["memory"] == "REPLACE" for action in actions))

    def test_replace_is_blocked_when_memory_is_empty(self):
        actions = self.manager.get_feasible_actions(
            label_budget=10,
            compute_budget=1000,
            memory_used=0,
            memory_capacity=10,
            compute_costs=self._costs(),
        )

        self.assertFalse(any(action["memory"] == "REPLACE" for action in actions))
        self.assertTrue(any(action["memory"] == "STORE" for action in actions))

    def test_tier_zero_cost_must_be_zero(self):
        with self.assertRaises(ValueError):
            self.manager.get_feasible_actions(
                label_budget=10,
                compute_budget=1000,
                memory_used=0,
                memory_capacity=10,
                compute_costs={0: 100, 1: 250, 2: 500, 3: 1000},
            )


if __name__ == "__main__":
    unittest.main()