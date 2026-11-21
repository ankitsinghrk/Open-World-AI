
"""Action manager for the two-stage Hybrid V1 architecture."""

from src.hybrid_decision_functions import (
    sample_actions,
    compute_actions,
    is_sample_action_feasible,
    is_compute_action_feasible,
)


class HybridActionManager:

    def __init__(self, memory_capacity, label_cost=1):

        if type(memory_capacity) is not int or memory_capacity <= 0:
            raise ValueError("Memory capacity must be a positive integer")

        if type(label_cost) is not int or label_cost <= 0:
            raise ValueError("Label cost must be a positive integer")

        self.memory_capacity = memory_capacity
        self.label_cost = label_cost

    def get_sample_feasible_actions(self, label_budget, memory_used):

        return [
            action
            for action in sample_actions()
            if is_sample_action_feasible(
                action=action,
                label_budget=label_budget,
                memory_used=memory_used,
                memory_capacity=self.memory_capacity,
                label_cost=self.label_cost,
            )
        ]

    def get_compute_feasible_actions(self, compute_budget, compute_costs):

        return [
            action
            for action in compute_actions()
            if is_compute_action_feasible(
                action=action,
                compute_budget=compute_budget,
                compute_costs=compute_costs,
            )
        ]

    def get_sample_action(self, action_id):

        if type(action_id) is not int or not 1 <= action_id <= 6:
            raise ValueError("Sample action ID must be between 1 and 6")

        return sample_actions()[action_id - 1]

    def get_compute_action(self, action_id):

        if type(action_id) is not int or not 1 <= action_id <= 4:
            raise ValueError("Compute action ID must be between 1 and 4")

        return compute_actions()[action_id - 1]

    def get_sample_feasible_mask(self, label_budget, memory_used):

        feasible = self.get_sample_feasible_actions(
            label_budget,
            memory_used,
        )

        mask = [False] * 6

        for action in feasible:
            mask[action["id"] - 1] = True

        if not any(mask):
            raise RuntimeError("No feasible sample action")

        return mask

    def get_compute_feasible_mask(self, compute_budget, compute_costs):

        feasible = self.get_compute_feasible_actions(
            compute_budget,
            compute_costs,
        )

        mask = [False] * 4

        for action in feasible:
            mask[action["id"] - 1] = True

        if not any(mask):
            raise RuntimeError("No feasible compute action")

        return mask
