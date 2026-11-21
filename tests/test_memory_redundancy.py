import unittest

from src.memory_redundancy import MemoryRedundancy


class TestMemoryRedundancy(unittest.TestCase):
    def setUp(self):
        self.redundancy = MemoryRedundancy()

    def record(self, memory_id, features):
        return {"id": memory_id, "item": {"features": features, "label": 1}}

    def test_single_item_has_zero_redundancy(self):
        records = [self.record("mem_1", [1, 0])]
        self.assertEqual(self.redundancy.calculate("mem_1", records), 0.0)

    def test_identical_other_item_has_full_redundancy(self):
        records = [
            self.record("mem_1", [1, 2]),
            self.record("mem_2", [1, 2]),
        ]
        self.assertAlmostEqual(self.redundancy.calculate("mem_1", records), 1.0)

    def test_orthogonal_other_item_has_half_redundancy(self):
        records = [
            self.record("mem_1", [1, 0]),
            self.record("mem_2", [0, 1]),
        ]
        self.assertAlmostEqual(self.redundancy.calculate("mem_1", records), 0.5)

    def test_target_is_excluded_from_its_own_average(self):
        records = [
            self.record("mem_1", [1, 0]),
            self.record("mem_2", [1, 0]),
            self.record("mem_3", [0, 1]),
        ]
        self.assertAlmostEqual(self.redundancy.calculate("mem_1", records), 0.75)

    def test_calculates_mean_similarity_across_other_items(self):
        records = [
            self.record("mem_1", [1, 0]),
            self.record("mem_2", [1, 0]),
            self.record("mem_3", [0, 1]),
            self.record("mem_4", [-1, 0]),
        ]
        # Similarities: 1.0, 0.5, 0.0; mean = 0.5.
        self.assertAlmostEqual(self.redundancy.calculate("mem_1", records), 0.5)

    def test_result_stays_between_zero_and_one(self):
        records = [
            self.record("mem_1", [1, 0]),
            self.record("mem_2", [-1, 0]),
        ]
        result = self.redundancy.calculate("mem_1", records)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_unknown_id_is_rejected(self):
        with self.assertRaises(KeyError):
            self.redundancy.calculate("missing", [self.record("mem_1", [1, 0])])

    def test_duplicate_target_id_is_rejected(self):
        records = [self.record("mem_1", [1, 0]), self.record("mem_1", [0, 1])]
        with self.assertRaises(KeyError):
            self.redundancy.calculate("mem_1", records)

    def test_rejects_invalid_record_shape(self):
        with self.assertRaises(ValueError):
            self.redundancy.calculate("mem_1", [{"id": "mem_1"}])

    def test_rejects_missing_features_on_other_item(self):
        records = [self.record("mem_1", [1, 0]), {"id": "mem_2", "item": {"label": 0}}]
        with self.assertRaises(ValueError):
            self.redundancy.calculate("mem_1", records)

    def test_rejects_mismatched_feature_dimensions(self):
        records = [self.record("mem_1", [1, 0]), self.record("mem_2", [1, 0, 1])]
        with self.assertRaises(ValueError):
            self.redundancy.calculate("mem_1", records)


if __name__ == "__main__":
    unittest.main()
