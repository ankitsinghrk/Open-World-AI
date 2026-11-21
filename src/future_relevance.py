import copy
import math


class FutureRelevanceEstimator:
    """Estimate expected usefulness across a supplied future-task distribution."""

    def __init__(self, relevance_function, horizon_decay=1.0):
        if not callable(relevance_function):
            raise ValueError("Relevance function must be callable")
        if (isinstance(horizon_decay, bool)
                or not isinstance(horizon_decay, (int, float))
                or not math.isfinite(horizon_decay)
                or not 0.0 < horizon_decay <= 1.0):
            raise ValueError("Horizon decay must be in the range (0, 1]")
        self.relevance_function = relevance_function
        self.horizon_decay = float(horizon_decay)

    @staticmethod
    def _validate_tasks(future_tasks):
        if not isinstance(future_tasks, (list, tuple)) or not future_tasks:
            raise ValueError("Future tasks must be a non-empty list or tuple")

        checked = []
        seen = set()
        total = 0.0
        required = {"task_id", "probability", "horizon_step"}

        for task in future_tasks:
            if not isinstance(task, dict) or set(task) != required:
                raise ValueError("Each task needs task_id, probability, and horizon_step")

            task_id = task["task_id"]
            probability = task["probability"]
            step = task["horizon_step"]

            if not isinstance(task_id, str) or not task_id:
                raise ValueError("Task IDs must be non-empty strings")
            if task_id in seen:
                raise ValueError("Future task IDs must be unique")
            seen.add(task_id)

            if (isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(probability)
                    or not 0.0 <= probability <= 1.0):
                raise ValueError("Task probabilities must be between 0 and 1")

            if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                raise ValueError("Horizon steps must be positive integers")

            checked.append((task_id, float(probability), step))
            total += probability

        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Future task probabilities must sum to 1")
        return checked

    def estimate(self, memory_item, future_tasks):
        """Return a discounted expected relevance score in [0, 1]."""
        if memory_item is None:
            raise ValueError("Memory item cannot be empty")

        tasks = self._validate_tasks(future_tasks)
        weighted_score = 0.0
        total_weight = 0.0

        for task_id, probability, step in tasks:
            relevance = self.relevance_function(copy.deepcopy(memory_item), task_id)
            if (isinstance(relevance, bool)
                    or not isinstance(relevance, (int, float))
                    or not math.isfinite(relevance)
                    or not 0.0 <= relevance <= 1.0):
                raise ValueError("Task relevance must be a finite number from 0 to 1")

            weight = probability * (self.horizon_decay ** (step - 1))
            weighted_score += weight * relevance
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_score / total_weight
