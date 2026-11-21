"""Chapter 11 joint-action, feasibility, and reward functions.

Action id:
    id(label, memory, compute) = 12 * label + 4 * memory + compute + 1
"""

MEMORY_OPERATIONS = ("DISCARD", "STORE", "REPLACE")
COMPUTE_TIERS = (0, 1, 2, 3)


def action_id(label, memory, compute):
    return 12 * label + 4 * memory + compute + 1


def joint_actions():
    """Return the 24 joint actions in network order."""
    actions = []
    for label in (0, 1):
        for memory in range(3):
            for compute in COMPUTE_TIERS:
                actions.append({
                    "id": action_id(label, memory, compute),
                    "label": label,
                    "memory": MEMORY_OPERATIONS[memory],
                    "compute": compute,
                })
    return actions


def is_action_feasible(
    action,
    label_budget,
    compute_budget,
    memory_used,
    memory_capacity,
    compute_costs,
    label_cost,
):
    """Hard feasibility from section 11.3.5.

    DISCARD does not need stored memory: it drops the current item.
    STORE needs free capacity. REPLACE needs a stored item.
    Tier 0 has zero compute cost, so it is not budget-limited.
    """
    if action["label"] == 1 and label_budget < label_cost:
        return False

    compute_tier = action["compute"]
    compute_cost = compute_costs[compute_tier]
    if compute_tier > 0 and compute_cost > compute_budget:
        return False

    if action["memory"] == "STORE" and memory_used >= memory_capacity:
        return False

    if action["memory"] == "REPLACE" and memory_used <= 0:
        return False

    return True


def operation_cost(
    label_action,
    compute_cost,
    label_cost,
    memory_operation_cost,
    label_weight,
    memory_weight,
    compute_weight,
):
    """Resource-operation cost Ct = λL cL ℓ + λM cM(μ) + λC CC(c)."""
    return (
        label_weight * label_cost * label_action
        + memory_weight * memory_operation_cost
        + compute_weight * compute_cost
    )


def v1_reward(utility_before, utility_after, cost):
    """Rt = ΔUt − Ct."""
    return (utility_after - utility_before) - cost
