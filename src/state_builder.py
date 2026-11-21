import math

from config import INITIAL_LABEL_BUDGET
from config import INITIAL_COMPUTE_BUDGET


class StateBuilder:

    def calculate_memory_remaining(self, memory_status):
        if not isinstance(memory_status, dict):
            raise ValueError("Memory status must be a dictionary")

        used = memory_status.get("used")
        capacity = memory_status.get("capacity")

        if used is None or capacity is None:
            raise ValueError("Memory status needs used and capacity")

        if used < 0 or capacity <= 0 or used > capacity:
            raise ValueError("Invalid memory capacity values")

        remaining_ratio = 1 - (used / capacity)
        return remaining_ratio

    def calculate_label_remaining(self, resources):
        if not isinstance(resources, dict):
            raise ValueError("Resources must be a dictionary")

        remaining_labels = resources.get("labels")

        if remaining_labels is None:
            raise ValueError("Resources need labels")

        if INITIAL_LABEL_BUDGET <= 0:
            raise ValueError("INITIAL_LABEL_BUDGET must be greater than 0")

        if not 0 <= remaining_labels <= INITIAL_LABEL_BUDGET:
            raise ValueError("Resources must be within the initial label budget")

        remaining_ratio = remaining_labels / INITIAL_LABEL_BUDGET
        return remaining_ratio

    def calculate_compute_remaining(self, resources):
        if not isinstance(resources, dict):
            raise ValueError("Resources must be a dictionary")

        remaining_compute = resources.get("compute")

        if remaining_compute is None:
            raise ValueError("Resources need compute")

        if INITIAL_COMPUTE_BUDGET <= 0:
            raise ValueError("Initial compute budget must be greater than 0")

        if not 0 <= remaining_compute <= INITIAL_COMPUTE_BUDGET:
            raise ValueError("Invalid compute budget")

        remaining_ratio = remaining_compute / INITIAL_COMPUTE_BUDGET
        return remaining_ratio

    def build_state(self, features, prediction, uncertainty, novelty,
                    memory_status, resources, history):
        if not isinstance(features, list) or not features:
            raise ValueError("Features must be a non-empty list")

        for value in features:
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError("Features must contain finite numbers")

        for name, value in (("Uncertainty", uncertainty), ("Novelty", novelty)):
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")

        if not isinstance(memory_status, dict):
            raise ValueError("Memory status must be a dictionary")
        if not isinstance(resources, dict):
            raise ValueError("Resources must be a dictionary")
        if not isinstance(history, list):
            raise ValueError("History must be a list")

        used = memory_status.get("used")
        capacity = memory_status.get("capacity")
        if used is None or capacity is None:
            raise ValueError("Memory status needs used and capacity")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (used, capacity)):
            raise ValueError("Memory values must be finite numbers")
        if used < 0 or capacity <= 0 or used > capacity:
            raise ValueError("Invalid memory capacity values")

        labels = resources.get("labels")
        compute = resources.get("compute")
        if labels is None or compute is None:
            raise ValueError("Resources need labels and compute")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (labels, compute)):
            raise ValueError("Resource values must be finite numbers")

        return {
            "features": features,
            "prediction": prediction,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "memory_status": memory_status,
            "memory_remaining": self.calculate_memory_remaining(memory_status),
            "label_remaining": self.calculate_label_remaining(resources),
            "compute_remaining": self.calculate_compute_remaining(resources),
            "resources": resources,
            "history": history,
        }