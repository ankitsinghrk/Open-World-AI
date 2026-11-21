import unittest

from src.similarity import FeatureSimilarity


class TestFeatureSimilarity(unittest.TestCase):

    def setUp(self):
        self.similarity = FeatureSimilarity()

    def test_identical_vectors_have_cosine_one(self):
        self.assertAlmostEqual(self.similarity.cosine([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_have_cosine_zero(self):
        self.assertAlmostEqual(self.similarity.cosine([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors_have_cosine_minus_one(self):
        self.assertAlmostEqual(self.similarity.cosine([1, 2], [-1, -2]), -1.0)

    def test_normalized_identical_similarity_is_one(self):
        self.assertAlmostEqual(self.similarity.normalized([1, 2], [1, 2]), 1.0)

    def test_normalized_orthogonal_similarity_is_half(self):
        self.assertAlmostEqual(self.similarity.normalized([1, 0], [0, 1]), 0.5)

    def test_normalized_opposite_similarity_is_zero(self):
        self.assertAlmostEqual(self.similarity.normalized([1, 2], [-1, -2]), 0.0)

    def test_similarity_is_symmetric(self):
        first = [0.2, 0.4, 0.8]
        second = [0.7, 0.1, 0.3]
        self.assertAlmostEqual(self.similarity.normalized(first, second), self.similarity.normalized(second, first))

    def test_rejects_empty_vectors(self):
        with self.assertRaises(ValueError):
            self.similarity.cosine([], [1, 2])

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            self.similarity.cosine([1, 2], [1, 2, 3])

    def test_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            self.similarity.cosine([0, 0], [1, 2])

    def test_rejects_invalid_values(self):
        for invalid in (True, None, "0.5", float("nan"), float("inf")):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    self.similarity.cosine([1, invalid], [1, 2])


if __name__ == "__main__":
    unittest.main()
