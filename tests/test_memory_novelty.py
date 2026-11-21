import unittest

from src.memory_novelty import MemoryNovelty


class TestMemoryNovelty(unittest.TestCase):

    def setUp(self):
        self.novelty = MemoryNovelty()

    def record(self, memory_id, features):
        return {"id": memory_id, "item": {"features": features, "label": 1}}

    def test_identical_candidate_has_zero_novelty(self):
        bank = [self.record("mem_1", [1, 0])]
        self.assertAlmostEqual(self.novelty.calculate([1, 0], bank), 0.0)

    def test_orthogonal_candidate_has_half_novelty(self):
        bank = [self.record("mem_1", [1, 0])]
        self.assertAlmostEqual(self.novelty.calculate([0, 1], bank), 0.5)

    def test_opposite_candidate_has_full_novelty(self):
        bank = [self.record("mem_1", [1, 0])]
        self.assertAlmostEqual(self.novelty.calculate([-1, 0], bank), 1.0)

    def test_uses_maximum_similarity_from_reference_bank(self):
        bank = [self.record("mem_1", [0, 1]), self.record("mem_2", [1, 0]), self.record("mem_3", [-1, 0])]
        self.assertAlmostEqual(self.novelty.calculate([1, 0], bank), 0.0)

    def test_empty_reference_bank_gives_full_novelty(self):
        self.assertEqual(self.novelty.calculate([0.3, 0.7], []), 1.0)

    def test_result_stays_between_zero_and_one(self):
        bank = [self.record("mem_1", [0.2, 0.8]), self.record("mem_2", [-0.3, 0.4])]
        result = self.novelty.calculate([0.5, 0.5], bank)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_rejects_non_list_memory_bank(self):
        with self.assertRaises(ValueError):
            self.novelty.calculate([1, 0], "not a memory bank")

    def test_rejects_invalid_record(self):
        with self.assertRaises(ValueError):
            self.novelty.calculate([1, 0], [{"id": "mem_1"}])

    def test_rejects_missing_features(self):
        with self.assertRaises(ValueError):
            self.novelty.calculate([1, 0], [{"id": "mem_1", "item": {"label": 1}}])

    def test_rejects_dimension_mismatch(self):
        bank = [self.record("mem_1", [1, 0, 1])]
        with self.assertRaises(ValueError):
            self.novelty.calculate([1, 0], bank)

    def test_rejects_zero_vector(self):
        bank = [self.record("mem_1", [1, 0])]
        with self.assertRaises(ValueError):
            self.novelty.calculate([0, 0], bank)


if __name__ == "__main__":
    unittest.main()
