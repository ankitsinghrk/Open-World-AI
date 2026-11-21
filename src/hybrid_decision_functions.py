
"""Separate sample-level and batch-level actions for Hybrid V1."""

SAMPLE_MEMORY_OPERATIONS = ("DISCARD", "STORE", "REPLACE")
COMPUTE_TIERS = (0, 1, 2, 3)


def sample_action_id(label, memory):
    """Return a one-based ID for one of the six sample actions."""
    if label not in (0, 1):
        raise ValueError("Label action must be 0 or 1")

    if memory not in SAMPLE_MEMORY_OPERATIONS:
        raise ValueError("Invalid memory operation")

    memory_index = SAMPLE_MEMORY_OPERATIONS.index(memory)
    return label * 3 + memory_index + 1


def sample_actions():
    """Generate six sample-level actions."""
    actions = []

    for label in (0, 1):
        for memory in SAMPLE_MEMORY_OPERATIONS:
            actions.append({
                "id": sample_action_id(label, memory),
                "label": label,
                "memory": memory,
            })

    return actions


def compute_actions():
    """Generate four batch-level compute actions."""
    return [
        {"id": tier + 1, "compute": tier}
        for tier in COMPUTE_TIERS
    ]


def is_sample_action_feasible(
    action,
    label_budget,
    memory_used,
    memory_capacity,
    label_cost=1,
):
    if label_budget < 0:
        raise ValueError("Label budget cannot be negative")

    if not 0 <= memory_used <= memory_capacity:
        raise ValueError("Invalid memory usage")

    if label_cost <= 0:
        raise ValueError("Label cost must be positive")

    if action["label"] == 1 and label_budget < label_cost:
        return False

    if action["memory"] == "STORE":
        return memory_used < memory_capacity

    if action["memory"] == "REPLACE":
        return memory_used > 0

    return action["memory"] == "DISCARD"


def is_compute_action_feasible(action, compute_budget, compute_costs):
    if compute_budget < 0:
        raise ValueError("Compute budget cannot be negative")

    tier = action["compute"]

    if tier not in COMPUTE_TIERS:
        raise ValueError("Invalid compute tier")

    if not isinstance(compute_costs, dict):
        raise ValueError("Compute costs must be a dictionary")

    if tier not in compute_costs:
        raise ValueError("Missing compute cost")

    cost = compute_costs[tier]

    if (
        isinstance(cost, bool)
        or not isinstance(cost, int)
        or cost < 0
    ):
        raise ValueError("Invalid compute cost")

    if tier == 0 and cost != 0:
        raise ValueError("Tier 0 must have zero cost")

    if tier > 0 and cost <= 0:
        raise ValueError("Positive tiers must have positive cost")

    return cost <= compute_budget
