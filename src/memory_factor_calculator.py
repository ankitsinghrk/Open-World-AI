"""Calculate novelty and redundancy for an incoming memory candidate."""

from src.similarity import FeatureSimilarity


class MemoryFactorCalculator:
    """Derive novelty and redundancy from candidate features and memory bank.

    The candidate is not assumed to be stored yet. Every record in the bank
    must have the shape {"id": ..., "item": {"features": [...]}}.
    """

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
        """Return calculated novelty and redundancy, both in [0, 1].

        Empty bank: novelty=1.0 and redundancy=0.0.
        Otherwise:
          novelty = 1 - maximum similarity
          redundancy = mean similarity
        """
        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        if not memory_records:
            self.similarity._validate_vector(candidate_features, "Candidate features")
            return {"novelty": 1.0, "redundancy": 0.0}

        similarities = [
            self.similarity.normalized(
                candidate_features, self._get_features(record)
            )
            for record in memory_records
        ]

        return {
            "novelty": 1.0 - max(similarities),
            "redundancy": sum(similarities) / len(similarities),
        }
