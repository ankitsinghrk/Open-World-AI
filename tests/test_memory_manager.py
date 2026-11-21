import unittest

from src.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):

    def test_initializes_with_given_capacity(self):
        manager = MemoryManager(capacity=5)
        self.assertEqual(manager.capacity, 5)
        self.assertEqual(manager.memory, [])

    def test_rejects_zero_or_negative_capacity(self):
        for capacity in (0, -1):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    MemoryManager(capacity=capacity)

    def test_store_adds_item_and_returns_true(self):
        manager = MemoryManager(capacity=2)
        item = {"features": [0.25, 0.64, 0.81], "label": 1}
        result = manager.store(item)
        self.assertTrue(result)
        self.assertEqual(manager.memory, [item])

    def test_store_rejects_none(self):
        manager = MemoryManager(capacity=2)
        with self.assertRaises(ValueError):
            manager.store(None)
        self.assertEqual(manager.memory, [])

    def test_store_rejects_item_when_memory_is_full(self):
        manager = MemoryManager(capacity=1)
        manager.store("first")
        with self.assertRaises(ValueError):
            manager.store("second")
        self.assertEqual(manager.memory, ["first"])

    def test_discard_removes_and_returns_item(self):
        manager = MemoryManager(capacity=3)
        manager.store("first")
        manager.store("second")
        removed = manager.discard(0)
        self.assertEqual(removed, "first")
        self.assertEqual(manager.memory, ["second"])

    def test_discard_rejects_invalid_indices(self):
        manager = MemoryManager(capacity=2)
        manager.store("first")
        for index in (-1, 1):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    manager.discard(index)
        self.assertEqual(manager.memory, ["first"])

    def test_replace_returns_old_item_and_stores_new_item(self):
        manager = MemoryManager(capacity=2)
        manager.store("old")
        old_item = manager.replace(0, "new")
        self.assertEqual(old_item, "old")
        self.assertEqual(manager.memory, ["new"])

    def test_replace_rejects_invalid_index(self):
        manager = MemoryManager(capacity=2)
        manager.store("old")
        with self.assertRaises(IndexError):
            manager.replace(1, "new")
        self.assertEqual(manager.memory, ["old"])

    def test_replace_rejects_none(self):
        manager = MemoryManager(capacity=2)
        manager.store("old")
        with self.assertRaises(ValueError):
            manager.replace(0, None)
        self.assertEqual(manager.memory, ["old"])

    def test_memory_status_reports_used_and_capacity(self):
        manager = MemoryManager(capacity=3)
        manager.store("first")
        manager.store("second")
        self.assertEqual(
            manager.get_memory_status(),
            {"used": 2, "capacity": 3}
        )


    def test_store_assigns_a_stable_memory_id(self):
        manager = MemoryManager(capacity=2)
        manager.store({"features": [0.25, 0.64, 0.81], "label": 1})

        memory_id = manager.get_memory_id(0)

        self.assertEqual(memory_id, "mem_000001")
        self.assertEqual(manager.get_index_by_id(memory_id), 0)

    def test_ids_remain_stable_when_another_item_is_discarded(self):
        manager = MemoryManager(capacity=3)
        manager.store("first")
        manager.store("second")
        second_id = manager.get_memory_id(1)

        manager.discard(0)

        self.assertEqual(manager.get_memory_id(0), second_id)
        self.assertEqual(manager.get_index_by_id(second_id), 0)

    def test_new_item_gets_new_id_after_discard(self):
        manager = MemoryManager(capacity=2)
        manager.store("first")
        first_id = manager.get_memory_id(0)
        manager.discard(0)
        manager.store("second")
        second_id = manager.get_memory_id(0)

        self.assertNotEqual(first_id, second_id)
        self.assertEqual(second_id, "mem_000002")

    def test_replace_assigns_a_new_id_to_replacement_item(self):
        manager = MemoryManager(capacity=2)
        manager.store("old")
        old_id = manager.get_memory_id(0)

        manager.replace(0, "new")
        new_id = manager.get_memory_id(0)

        self.assertNotEqual(old_id, new_id)
        self.assertEqual(manager.get_index_by_id(new_id), 0)
        with self.assertRaises(KeyError):
            manager.get_index_by_id(old_id)

    def test_unknown_memory_id_is_rejected(self):
        manager = MemoryManager(capacity=2)
        with self.assertRaises(KeyError):
            manager.get_index_by_id("mem_999999")

    def test_get_memory_records_includes_id_and_original_item(self):
        manager = MemoryManager(capacity=2)
        item = {"features": [0.25, 0.64, 0.81], "label": 1}
        manager.store(item)

        records = manager.get_memory_records()

        self.assertEqual(records, [{"id": "mem_000001", "item": item}])

    def test_memory_property_returns_a_separate_list(self):
        manager = MemoryManager(capacity=2)
        manager.store("first")

        visible_memory = manager.memory
        visible_memory.append("outside change")

        self.assertEqual(manager.memory, ["first"])


    def test_rejects_boolean_capacity(self):
        with self.assertRaises(ValueError):
            MemoryManager(capacity=True)

    def test_rejects_float_capacity(self):
        with self.assertRaises(ValueError):
            MemoryManager(capacity=1.5)

    def test_store_protects_item_from_external_changes(self):
        manager = MemoryManager(capacity=2)

        item = {"features": [1.0, 2.0]}
        manager.store(item)

        item["features"][0] = 99.0

        self.assertEqual(
            manager.memory[0]["features"],
            [1.0, 2.0]
        )

    def test_memory_property_protects_nested_data(self):
        manager = MemoryManager(capacity=2)
        manager.store({"features": [1.0, 2.0]})

        visible_memory = manager.memory
        visible_memory[0]["features"][0] = 99.0

        self.assertEqual(
            manager.memory[0]["features"],
            [1.0, 2.0]
        )

    def test_get_memory_records_protects_internal_data(self):
        manager = MemoryManager(capacity=2)
        manager.store({"features": [1.0, 2.0]})

        records = manager.get_memory_records()
        records[0]["item"]["features"][0] = 99.0

        self.assertEqual(
            manager.get_memory_records()[0]["item"]["features"],
            [1.0, 2.0]
        )



    def test_failed_store_does_not_advance_memory_id(self):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise RuntimeError("Simulated copy failure")

        manager = MemoryManager(capacity=2)

        with self.assertRaises(RuntimeError):
            manager.store(Uncopyable())

        manager.store("valid item")

        self.assertEqual(
            manager.get_memory_id(0),
            "mem_000001"
        )

    def test_failed_replace_does_not_change_memory_or_id(self):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise RuntimeError("Simulated copy failure")

        manager = MemoryManager(capacity=2)
        manager.store("original")

        original_id = manager.get_memory_id(0)

        with self.assertRaises(RuntimeError):
            manager.replace(0, Uncopyable())

        self.assertEqual(manager.memory, ["original"])
        self.assertEqual(
            manager.get_memory_id(0),
            original_id
        )

        manager.store("next item")

        self.assertEqual(
            manager.get_memory_id(1),
            "mem_000002"
        )




if __name__ == "__main__":
    unittest.main()
