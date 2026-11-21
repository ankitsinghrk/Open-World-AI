
from config import INITIAL_LABEL_BUDGET, INITIAL_COMPUTE_BUDGET


class ResourceManager:

    def __init__(
        self,
        label_budget=INITIAL_LABEL_BUDGET,
        compute_budget=INITIAL_COMPUTE_BUDGET
    ):

        if not isinstance(label_budget, int) or isinstance(label_budget, bool):
            raise ValueError("Invalid label budget")

        if not isinstance(compute_budget, int) or isinstance(compute_budget, bool):
            raise ValueError("Invalid compute budget")

        if label_budget < 0 or compute_budget < 0:
            raise ValueError("Budgets cannot be negative")

        self.label_budget = label_budget
        self.compute_budget = compute_budget
        self.initial_label_budget = label_budget
        self.initial_compute_budget = compute_budget

    def consume_label(self):

        if self.label_budget <= 0:
            raise ValueError("Label budget exhausted")

        self.label_budget -= 1
        return self.label_budget

    def consume_compute(self, cost):

        if not isinstance(cost, int) or isinstance(cost, bool):
            raise ValueError("Compute cost must be an integer")

        if cost < 0:
            raise ValueError("Compute cost cannot be negative")

        if cost > self.compute_budget:
            raise ValueError("Compute budget insufficient")

        self.compute_budget -= cost
        return self.compute_budget

    def get_resources(self):

        return {
            "labels": self.label_budget,
            "compute": self.compute_budget
        }


    def restore_label(self):

        if self.label_budget >= self.initial_label_budget:
            raise ValueError("Cannot restore beyond initial label budget")

        self.label_budget += 1

        return self.label_budget

    def restore_compute(self, amount):

        if not isinstance(amount, int) or isinstance(amount, bool):
            raise ValueError("Compute amount must be an integer")

        if amount <= 0:
            raise ValueError("Compute amount must be positive")

        if self.compute_budget + amount > self.initial_compute_budget:
            raise ValueError("Cannot restore beyond initial compute budget")

        self.compute_budget += amount

        return self.compute_budget
