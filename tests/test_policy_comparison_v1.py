import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from config import COMPUTE_COSTS, INITIAL_COMPUTE_BUDGET, INITIAL_LABEL_BUDGET
from src.action_manager import ActionManager
from src.memory_manager import MemoryManager
from experiments.open_world_policy_comparison_v1 import (
    build_heuristic_schedule,
    build_comparison_action_mask,
    evenly_spaced_positions,
    fixed_heuristic_action,
    V1RingReplayBuffer,
)
from experiments.aggregate_policy_comparison_v1 import mean_sd_ci


class FakeEnvironment:
    def __init__(self, mask):
        self.mask = mask

    def build_action_mask(self):
        return list(self.mask)


class PolicyComparisonV1Tests(unittest.TestCase):
    def setUp(self):
        self.actions = ActionManager()
        self.base_mask = [True] * 24

    def test_aggregator_reports_paired_t_interval_for_ten_seeds(self):
        result = mean_sd_ci([float(i) for i in range(10)])
        self.assertEqual(result["n"], 10)
        self.assertAlmostEqual(result["mean"], 4.5)
        self.assertGreater(result["ci95_high"], result["mean"])
        self.assertLess(result["ci95_low"], result["mean"])

    def test_aggregator_rejects_wrong_seed_count(self):
        with self.assertRaises(ValueError):
            mean_sd_ci([1.0, 2.0, 3.0])

    def test_v1_ring_replay_buffer_fifo_eviction_and_sampling(self):
        buffer = V1RingReplayBuffer(capacity=3)
        for i in range(4):
            buffer.add([float(i)], i, float(i), [float(i + 1)], False, [True] * 24)
        self.assertEqual(len(buffer), 3)
        stored = {item[1] for item in buffer.sample(3)}
        self.assertEqual(stored, {1, 2, 3})
        with self.assertRaises(ValueError):
            buffer.sample(4)

    def test_even_spacing_is_deterministic_and_left_anchored(self):
        self.assertEqual(evenly_spaced_positions(10, 3), {0, 3, 6})
        self.assertEqual(evenly_spaced_positions(4, 9), {0, 1, 2, 3})
        self.assertEqual(evenly_spaced_positions(10, 0), set())

    def test_shared_mask_prevents_unlabelled_memory_storage(self):
        memory = MemoryManager(capacity=10)
        mask = build_comparison_action_mask(
            FakeEnvironment(self.base_mask), self.actions, memory
        )
        for idx, action in enumerate(self.actions.generate_actions()):
            if action["label"] == 0 and action["memory"] in ("STORE", "REPLACE"):
                self.assertFalse(mask[idx])

    def test_shared_mask_prevents_compute_with_empty_memory_without_label(self):
        memory = MemoryManager(capacity=10)
        mask = build_comparison_action_mask(
            FakeEnvironment(self.base_mask), self.actions, memory
        )
        for idx, action in enumerate(self.actions.generate_actions()):
            if action["label"] == 0 and action["compute"] > 0:
                self.assertFalse(mask[idx])

    def test_shared_mask_prevents_label_waste_action(self):
        memory = MemoryManager(capacity=10)
        mask = build_comparison_action_mask(
            FakeEnvironment(self.base_mask), self.actions, memory
        )
        for idx, action in enumerate(self.actions.generate_actions()):
            if action["label"] == 1 and action["memory"] == "DISCARD" and action["compute"] == 0:
                self.assertFalse(mask[idx])

    def test_mask_is_nonempty_when_memory_empty_and_budgets_available(self):
        memory = MemoryManager(capacity=10)
        mask = build_comparison_action_mask(
            FakeEnvironment(self.base_mask), self.actions, memory
        )
        self.assertTrue(any(mask))
        self.assertTrue(any(
            valid and action["label"] == 1 and action["memory"] == "STORE"
            for valid, action in zip(mask, self.actions.generate_actions())
        ))

    def test_tier_one_compute_schedule_respects_budget_cap(self):
        schedule = build_heuristic_schedule(10000)
        self.assertEqual(len(schedule["label_positions"]), INITIAL_LABEL_BUDGET)
        self.assertEqual(
            len(schedule["tier1_compute_positions"]),
            INITIAL_COMPUTE_BUDGET // COMPUTE_COSTS[1],
        )
        self.assertEqual(schedule["label_positions"][0], 0)
        self.assertEqual(schedule["tier1_compute_positions"], [2000, 4000, 6000, 8000])

    def test_fixed_heuristic_actions_are_feasible_for_smoke_horizon(self):
        schedule = build_heuristic_schedule(50)
        memory_used = 0
        labels_remaining = INITIAL_LABEL_BUDGET
        compute_remaining = INITIAL_COMPUTE_BUDGET
        for step in range(50):
            # Use a stable permissive mask plus the explicit V1 constraints.
            mem = MemoryManager(capacity=10)
            for j in range(memory_used):
                mem.store({"features": [0.0], "label": 1})
            mask = build_comparison_action_mask(
                FakeEnvironment(self.base_mask), self.actions, mem
            )
            index = fixed_heuristic_action(
                step, schedule,
                {"labels": labels_remaining, "compute": compute_remaining},
                memory_used, 10, self.actions, mask,
            )
            action = self.actions.get_action(index + 1)
            self.assertTrue(mask[index])
            if action["label"]:
                labels_remaining -= 1
                memory_used = min(10, memory_used + 1)
            compute_remaining -= COMPUTE_COSTS[action["compute"]]
        self.assertLessEqual(labels_remaining, INITIAL_LABEL_BUDGET)
        self.assertGreaterEqual(compute_remaining, 0)


if __name__ == "__main__":
    unittest.main()
