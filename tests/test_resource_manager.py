
import unittest

from src.resource_manager import ResourceManager


class TestResourceManager(unittest.TestCase):

    def test_initializes_with_custom_budgets(self):
        manager = ResourceManager(
            label_budget=10,
            compute_budget=500
        )

        self.assertEqual(
            manager.get_resources(),
            {"labels": 10, "compute": 500}
        )

    def test_rejects_invalid_initial_budgets(self):

        invalid_values = [-1, 1.5, True]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ResourceManager(label_budget=value)

                with self.assertRaises(ValueError):
                    ResourceManager(compute_budget=value)

    def test_consumes_one_label(self):
        manager = ResourceManager(10, 500)

        manager.consume_label()

        self.assertEqual(manager.get_resources()["labels"], 9)

    def test_rejects_label_consumption_when_exhausted(self):
        manager = ResourceManager(1, 500)

        manager.consume_label()

        with self.assertRaises(ValueError):
            manager.consume_label()

        self.assertEqual(manager.get_resources()["labels"], 0)

    def test_consumes_compute(self):
        manager = ResourceManager(10, 500)

        manager.consume_compute(100)

        self.assertEqual(manager.get_resources()["compute"], 400)

    def test_rejects_compute_consumption_beyond_budget(self):
        manager = ResourceManager(10, 100)

        with self.assertRaises(ValueError):
            manager.consume_compute(101)

        self.assertEqual(manager.get_resources()["compute"], 100)

    def test_compute_budget_can_reach_zero(self):
        manager = ResourceManager(10, 100)

        manager.consume_compute(100)

        self.assertEqual(manager.get_resources()["compute"], 0)

    def test_rejects_invalid_compute_amounts(self):
        manager = ResourceManager(10, 500)

        for amount in (-1, 1.5, True):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    manager.consume_compute(amount)

    def test_zero_compute_consumption(self):
        manager = ResourceManager(10, 500)

        manager.consume_compute(0)

        self.assertEqual(manager.get_resources()["compute"], 500)


    def test_restores_one_consumed_label(self):
        manager = ResourceManager(10, 500)

        manager.consume_label()
        manager.restore_label()

        self.assertEqual(
            manager.get_resources()["labels"],
            10
        )

    def test_restores_consumed_compute(self):
        manager = ResourceManager(10, 500)

        manager.consume_compute(100)
        manager.restore_compute(100)

        self.assertEqual(
            manager.get_resources()["compute"],
            500
        )

    def test_rejects_label_restore_beyond_initial_budget(self):
        manager = ResourceManager(10, 500)

        with self.assertRaises(ValueError):
            manager.restore_label()

        self.assertEqual(
            manager.get_resources()["labels"],
            10
        )

    def test_rejects_compute_restore_beyond_initial_budget(self):
        manager = ResourceManager(10, 500)

        with self.assertRaises(ValueError):
            manager.restore_compute(100)

        self.assertEqual(
            manager.get_resources()["compute"],
            500
        )



if __name__ == "__main__":
    unittest.main()
