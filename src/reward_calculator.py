
"""Calculate rewards for resource-bounded continual learning."""

import math

from src.decision_functions import operation_cost, v1_reward


class RewardCalculator:

    def __init__(
        self,
        forgetting_weight,
        label_cost_weight,
        compute_cost_weight,
        memory_cost_weight=0.0,
    ):
        self.forgetting_weight = self._validate_weight(
            forgetting_weight, "Forgetting weight"
        )
        self.label_cost_weight = self._validate_weight(
            label_cost_weight, "Label cost weight"
        )
        self.compute_cost_weight = self._validate_weight(
            compute_cost_weight, "Compute cost weight"
        )
        self.memory_cost_weight = self._validate_weight(
            memory_cost_weight, "Memory cost weight"
        )

    @staticmethod
    def _validate_weight(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a finite non-negative number"
            )
        return float(value)

    @staticmethod
    def _validate_score(value, name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be between 0 and 1")
        return float(value)

    @staticmethod
    def _validate_budget_usage(used, initial, name):
        if (
            isinstance(used, bool)
            or not isinstance(used, int)
            or used < 0
        ):
            raise ValueError(f"{name} usage must be a non-negative integer")

        if (
            isinstance(initial, bool)
            or not isinstance(initial, int)
            or initial <= 0
        ):
            raise ValueError(f"Initial {name} budget must be positive")

        if used > initial:
            raise ValueError(f"{name} usage exceeds initial budget")

        return used / initial

    def calculate(
        self,
        performance_before,
        performance_after,
        labels_used,
        compute_used,
        initial_label_budget,
        initial_compute_budget,
        previous_task_score_before=None,
        previous_task_score_after=None
    ):

        performance_before = self._validate_score(
            performance_before, "Performance before"
        )

        performance_after = self._validate_score(
            performance_after, "Performance after"
        )

        label_cost = self._validate_budget_usage(
            labels_used, initial_label_budget, "Label"
        )

        compute_cost = self._validate_budget_usage(
            compute_used, initial_compute_budget, "Compute"
        )

        if (
            (previous_task_score_before is None)
            != (previous_task_score_after is None)
        ):
            raise ValueError(
                "Both previous-task scores must be provided together"
            )

        if previous_task_score_before is None:
            forgetting = 0.0
        else:
            old_before = self._validate_score(
                previous_task_score_before,
                "Previous-task performance before"
            )

            old_after = self._validate_score(
                previous_task_score_after,
                "Previous-task performance after"
            )

            forgetting = max(0.0, old_before - old_after)

        performance_gain = performance_after - performance_before

        reward = (
            performance_gain
            - self.forgetting_weight * forgetting
            - self.label_cost_weight * label_cost
            - self.compute_cost_weight * compute_cost
        )

        return {
            "reward": reward,
            "performance_gain": performance_gain,
            "forgetting": forgetting,
            "label_cost": label_cost,
            "compute_cost": compute_cost
        }

    def calculate_v1(
        self,
        utility_before,
        utility_after,
        label_action,
        memory_operation_cost,
        compute_cost,
        label_cost,
    ):
        """Chapter 11 reward: immediate utility change minus operation cost."""
        utility_before = self._validate_score(utility_before, "Utility before")
        utility_after = self._validate_score(utility_after, "Utility after")

        if type(label_action) is not int or label_action not in (0, 1):
            raise ValueError("Label action must be 0 or 1")

        for name, value in (
            ("Label cost", label_cost),
            ("Memory operation cost", memory_operation_cost),
            ("Compute cost", compute_cost),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")

        cost = operation_cost(
            label_action=label_action,
            memory_operation_cost=float(memory_operation_cost),
            compute_cost=float(compute_cost),
            label_cost=float(label_cost),
            label_weight=self.label_cost_weight,
            memory_weight=self.memory_cost_weight,
            compute_weight=self.compute_cost_weight,
        )
        reward = v1_reward(utility_before, utility_after, cost)

        return {
            "reward": reward,
            "utility_change": utility_after - utility_before,
            "operation_cost": cost,
            "label_action": label_action,
            "label_cost": float(label_cost) * label_action,
            "memory_operation_cost": float(memory_operation_cost),
            "compute_cost": float(compute_cost),
        }
