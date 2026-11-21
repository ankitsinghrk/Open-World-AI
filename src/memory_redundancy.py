"""Estimate an item's redundancy relative to the current memory bank."""

from src.similarity import FeatureSimilarity


class MemoryRedundancy:
    """Calculate mean normalized similarity to all other memory items.

    Each record must have the shape {"id": ..., "item": {"features": [...]}}.
    The target item is excluded by its stable memory ID. A one-item bank has
    redundancy 0 because there are no other items to compare against.
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

    def calculate(self, memory_id, memory_records):
        """Return average similarity to every other item, in [0, 1]."""
        if not isinstance(memory_records, (list, tuple)):
            raise ValueError("Memory records must be a list or tuple")

        target_records = [r for r in memory_records if isinstance(r, dict) and r.get("id") == memory_id]
        if len(target_records) != 1:
            raise KeyError(f"Memory ID must identify exactly one record: {memory_id}")

        target_features = self._get_features(target_records[0])
        other_records = [r for r in memory_records if not (isinstance(r, dict) and r.get("id") == memory_id)]

        if not other_records:
            return 0.0

        similarities = []
        for record in other_records:
            other_features = self._get_features(record)
            similarities.append(self.similarity.normalized(target_features, other_features))

        return sum(similarities) / len(similarities)
