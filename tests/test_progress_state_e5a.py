import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.open_world_progress_state_e5a import (
    V1RingReplayBuffer,
    normalized_stream_progress,
    build_state_for_mode,
    build_paired_dqn_controller,
    select_masked_dqn_action,
)


class ProgressStateE5ATests(unittest.TestCase):

    def test_progress_first_position_is_zero(self):
        self.assertEqual(normalized_stream_progress(0, 100), 0.0)

    def test_progress_last_position_is_one(self):
        self.assertEqual(normalized_stream_progress(99, 100), 1.0)

    def test_progress_midpoint(self):
        self.assertAlmostEqual(
            normalized_stream_progress(50, 101), 0.5
        )

    def test_single_observation_stream_returns_zero(self):
        self.assertEqual(normalized_stream_progress(0, 1), 0.0)

    def test_progress_rejects_invalid_positions(self):
        for step, length in [(-1, 10), (10, 10), (0, 0)]:
            with self.subTest(step=step, length=length):
                with self.assertRaises(ValueError):
                    normalized_stream_progress(step, length)

    def test_progress_rejects_boolean_inputs(self):
        with self.assertRaises(ValueError):
            normalized_stream_progress(True, 10)

        with self.assertRaises(ValueError):
            normalized_stream_progress(0, True)

    def test_baseline_state_dimension_is_preserved(self):
        base = np.zeros(139, dtype=np.float64)

        result = build_state_for_mode(
            base, "baseline", 20, 100
        )

        self.assertEqual(len(result), 139)
        self.assertEqual(result, base.tolist())

    def test_progress_state_adds_exactly_one_feature(self):
        base = np.zeros(139, dtype=np.float64)

        result = build_state_for_mode(
            base, "progress", 50, 101
        )

        self.assertEqual(len(result), 140)
        self.assertAlmostEqual(result[-1], 0.5)

    def test_progress_state_preserves_original_features(self):
        base = np.arange(139, dtype=np.float64)

        result = build_state_for_mode(
            base, "progress", 25, 101
        )

        self.assertEqual(result[:-1], base.tolist())
        self.assertAlmostEqual(result[-1], 0.25)

    def test_state_builder_rejects_non_finite_values(self):
        base = np.zeros(139, dtype=np.float64)
        base[10] = np.nan

        with self.assertRaises(ValueError):
            build_state_for_mode(base, "progress", 0, 100)

    def test_state_builder_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            build_state_for_mode(
                np.zeros(139), "unknown", 0, 100
            )

    def test_paired_controller_initialization_is_equivalent(self):
        seed = 42

        baseline, baseline_meta = build_paired_dqn_controller(
            139, "baseline", seed
        )

        progress, progress_meta = build_paired_dqn_controller(
            139, "progress", seed
        )

        self.assertEqual(
            baseline.online_network.network[0].in_features, 139
        )
        self.assertEqual(
            progress.online_network.network[0].in_features, 140
        )

        self.assertTrue(
            progress_meta["shared_weight_equivalence"]
        )
        self.assertTrue(
            progress_meta["progress_input_weights_zero_at_initialization"]
        )

        self.assertEqual(
            baseline_meta["base_controller_initial_sha256"],
            progress_meta["projected_controller_initial_sha256"],
        )

        # Check functional equivalence for several progress values.
        generator = torch.Generator().manual_seed(12345)
        state = torch.randn(139, generator=generator)

        baseline.online_network.eval()
        progress.online_network.eval()

        with torch.no_grad():
            expected = baseline.online_network(state)

            for p in (0.0, 0.37, 1.0):
                augmented = torch.cat(
                    [state, torch.tensor([p])]
                )
                actual = progress.online_network(augmented)

                torch.testing.assert_close(
                    actual, expected, rtol=1e-6, atol=1e-6
                )

    def test_progress_controller_target_matches_online_at_initialization(self):
        agent, _ = build_paired_dqn_controller(
            139, "progress", 42
        )

        for online, target in zip(
            agent.online_network.parameters(),
            agent.target_network.parameters(),
        ):
            self.assertTrue(torch.equal(online, target))

    def test_masked_action_selection_never_returns_infeasible_action(self):
        agent, _ = build_paired_dqn_controller(
            139, "baseline", 42
        )

        state = np.zeros(139, dtype=np.float32)
        mask = torch.zeros(24, dtype=torch.bool)
        mask[4] = True
        mask[17] = True

        selected = select_masked_dqn_action(
            agent, state, mask, 1.0, random.Random(42)
        )

        self.assertIn(selected, [4, 17])

    def test_action_selection_rejects_empty_mask(self):
        agent, _ = build_paired_dqn_controller(
            139, "baseline", 42
        )

        with self.assertRaises(ValueError):
            select_masked_dqn_action(
                agent,
                np.zeros(139, dtype=np.float32),
                torch.zeros(24, dtype=torch.bool),
                0.5,
                random.Random(42),
            )

    def test_replay_rng_is_reproducible(self):
        first = V1RingReplayBuffer(10, rng_seed=123)
        second = V1RingReplayBuffer(10, rng_seed=123)

        for index in range(10):
            experience = (
                [float(index)],
                index,
                float(index),
                [float(index + 1)],
                False,
                [True] * 24,
            )
            first.add(*experience)
            second.add(*experience)

        self.assertEqual(
            first.sample(5),
            second.sample(5),
        )


if __name__ == "__main__":
    unittest.main()
