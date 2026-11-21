from src.decision_functions import is_action_feasible, joint_actions


class ActionManager:

    def __init__(self):
        self.memory_operations = ["DISCARD", "STORE", "REPLACE"]
        self.compute_tiers = [0, 1, 2, 3]

    def generate_actions(self):
        return joint_actions()

    def get_feasible_actions(
        self,
        label_budget,
        compute_budget,
        memory_used,
        memory_capacity,
        compute_costs,
        label_cost=1,
    ):
        if label_budget < 0 or compute_budget < 0:
            raise ValueError("Budgets cannot be negative")

        if memory_capacity <= 0:
            raise ValueError("Memory capacity must be positive")

        if not 0 <= memory_used <= memory_capacity:
            raise ValueError("Invalid memory usage")

        if not isinstance(compute_costs, dict):
            raise ValueError("Compute costs must be a dictionary")

        for tier in self.compute_tiers:
            if tier not in compute_costs:
                raise ValueError("Missing compute cost for a tier")

            cost = compute_costs[tier]

            if (
                not isinstance(cost, int)
                or isinstance(cost, bool)
                or cost < 0
            ):
                raise ValueError("Invalid compute cost")

        if compute_costs[0] != 0:
            raise ValueError("Compute cost of tier 0 must be 0")

        for tier in (1, 2, 3):
            if compute_costs[tier] <= 0:
                raise ValueError("Positive compute tiers must cost more than 0")

        if (
            isinstance(label_cost, bool)
            or not isinstance(label_cost, int)
            or label_cost <= 0
        ):
            raise ValueError("Label cost must be a positive integer")

        feasible_actions = []

        for action in self.generate_actions():
            if is_action_feasible(
                action,
                label_budget=label_budget,
                compute_budget=compute_budget,
                memory_used=memory_used,
                memory_capacity=memory_capacity,
                compute_costs=compute_costs,
                label_cost=label_cost,
            ):
                feasible_actions.append(action)

        return feasible_actions


    def get_network_index(self, action_id):
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise ValueError("Action ID must be an integer")

        if not 1 <= action_id <= 24:
            raise ValueError("Invalid action ID")

        return action_id - 1

    def get_action_id(self, network_index):
        if not isinstance(network_index, int) or isinstance(network_index, bool):
            raise ValueError("Network index must be an integer")

        if not 0 <= network_index < 24:
            raise ValueError("Invalid network index")

        return network_index + 1

    def get_action(self, action_id):
        if not isinstance(action_id, int) or isinstance(action_id, bool):
          raise ValueError("Action ID must be an integer")

        if not 1 <= action_id <= 24:
           raise ValueError("Invalid action ID")

        return self.generate_actions()[action_id - 1]



    def get_feasible_mask(self, feasible_actions):
        mask = [False] * 24

        for action in feasible_actions:
            action_id = action.get("id")

            index = self.get_network_index(action_id)
            mask[index] = True

        if not any(mask):
            raise ValueError("No feasible actions available")

        return mask
