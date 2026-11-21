"""Novelty estimation for a candidate memory item.

Novelty = 1 - maximum normalized cosine similarity to the reference bank.
"""

from src.similarity import FeatureSimilarity


class MemoryNovelty:
    """Calculate candidate novelty relative to existing memory records."""

    def __init__(self, similarity=None):
        self.similarity = similarity or FeatureSimilarity()

    @staticmethod
    def _get_features(record):
        if not isinstance(record, dict) or "item" not in record:
            raise ValueError("Each memory record must contain an item")
        item = record["item"]
        if not isinstance(item, dict) or "features" not in item:
            raise ValueError("Each memory item must contain features")
        return item["features"]

    def calculate(self, candidate_features, memory_records):
        """Return novelty in [0, 1]; an empty reference bank gives 1.0."""
        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        if not memory_records:
            self.similarity._validate_vector(candidate_features, "Candidate features")
            return 1.0

        similarities = []
        for record in memory_records:
            reference_features = self._get_features(record)
            similarities.append(
                self.similarity.normalized(candidate_features, reference_features)
            )
        return 1.0 - max(similarities)
