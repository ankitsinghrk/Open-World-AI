
import copy
from config import MEMORY_CAPACITY


class MemoryManager:
    """Stores memory items and assigns each item a stable internal ID."""

    def __init__(self, capacity=MEMORY_CAPACITY):

        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise ValueError(
                "Memory capacity must be a positive integer"
            )

        self.capacity = capacity
        self._records = []
        self._next_id = 1

    @property
    def memory(self):
        """Return an independent copy of stored items."""
        return copy.deepcopy([
            record["item"] for record in self._records
        ])

    def _create_memory_id(self):
        memory_id = f"mem_{self._next_id:06d}"
        self._next_id += 1
        return memory_id

    def _validate_index(self, index):

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(self._records)
        ):
            raise IndexError("Invalid memory index")

    def store(self, item):
        if item is None:
            raise ValueError("Item cannot be None")

        if len(self._records) >= self.capacity:
            raise ValueError("Memory is full")

        copied_item = copy.deepcopy(item)
        memory_id = self._create_memory_id()

        self._records.append({
            "id": memory_id,
            "item": copied_item
        })

        return True

    def discard(self, index):

        self._validate_index(index)

        removed_record = self._records.pop(index)

        return removed_record["item"]

    def replace(self, index, new_item):
        self._validate_index(index)

        if new_item is None:
            raise ValueError("New item cannot be None")

        copied_item = copy.deepcopy(new_item)
        memory_id = self._create_memory_id()

        old_item = self._records[index]["item"]

        self._records[index] = {
            "id": memory_id,
            "item": copied_item
        }

        return old_item

    def get_memory_id(self, index):

        self._validate_index(index)

        return self._records[index]["id"]

    def get_index_by_id(self, memory_id):

        for index, record in enumerate(self._records):

            if record["id"] == memory_id:
                return index

        raise KeyError(f"Unknown memory ID: {memory_id}")

    def get_memory_records(self):
        """Return independent copies of all memory records."""

        return copy.deepcopy(self._records)

    def get_memory_status(self):

        return {
            "used": len(self._records),
            "capacity": self.capacity
        }
