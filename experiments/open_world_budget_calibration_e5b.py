"""Fixed-schedule compute-budget calibration for Open-World AI.

Experiment ID: CDQN-BUDGET-CALIBRATION-01-E5B

Compares fixed task-learner optimizer-update budgets under a common,
predeclared annotation schedule and persistent-memory protocol. No adaptive
controller is instantiated in this calibration experiment.

This runner is separate from the frozen V1 and E5-A runners.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import open_world_causal_dqn_pilot_v2 as v2
from config import (
    COMPUTE_COSTS,
    INITIAL_COMPUTE_BUDGET,
    INITIAL_LABEL_BUDGET,
    LABEL_COST,
    MEMORY_CAPACITY,
)
from src.action_manager import ActionManager
from src.dqn_agent import DQNNetworks
from src.dqn_trainer import DQNTrainer
from src.environment.environment import Environment
from src.feature_extractor import FeatureExtractor
from src.hybrid_batch_trainer import HybridBatchTrainer
from src.memory_manager import MemoryManager
from src.prediction_provider import PredictionProvider
from src.replay_buffer import ReplayBuffer
from src.resource_manager import ResourceManager
from src.reward_calculator import RewardCalculator
from src.state_builder import StateBuilder
from src.state_encoder import StateEncoder

EXPERIMENT_ID = "CDQN-BUDGET-CALIBRATION-01-E5B"
V2_ID = "CDQN-PILOT-01-V2"
EXPECTED_V2_SHA256 = "6a2863258198f79eca09c682fe3523ab78f0d4b01b83591b0f0cb2e665870a61"
BUDGET_CREDITS = (4, 40, 100)
SCHEDULER_VERSION = "fixed_collision_free_after_memory_warmup_v0.2"


class V1RingReplayBuffer:
    """V1-local FIFO replay buffer with O(1) append and uniform sampling.

    It preserves the V2 replay-buffer semantics: bounded FIFO replacement and
    uniform sampling without replacement. Unlike V2's deque-to-list conversion,
    this ring representation samples directly by index, avoiding a full buffer
    copy on every controller update. It is local to this experiment; V2's
    ReplayBuffer class and source remain unchanged.
    """

    def __init__(self, capacity: int, rng_seed: int = 0):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity = capacity
        self._items: list[Any] = [None] * capacity
        self._size = 0
        self._next = 0
        if not isinstance(rng_seed, int) or isinstance(rng_seed, bool) or rng_seed < 0:
            raise ValueError('rng_seed must be a non-negative integer')
        self._rng = random.Random(rng_seed)

    def add(self, state, action, reward, next_state, done, next_feasible_mask):
        experience = (state, action, reward, next_state, done, next_feasible_mask)
        self._items[self._next] = copy.deepcopy(experience)
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return self._size

    def sample(self, batch_size: int):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        if batch_size > self._size:
            raise ValueError("Not enough experiences in replay buffer")
        indices = self._rng.sample(range(self._size), batch_size)
        return [self._items[index] for index in indices]

    def __len__(self):
        return self._size


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()




def normalized_stream_progress(
    step_index: int,
    full_stream_length: int,
) -> float:
    """Return zero-based normalized position in the complete stream."""
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or not isinstance(full_stream_length, int)
        or isinstance(full_stream_length, bool)
    ):
        raise ValueError("Stream position and length must be integers")

    if full_stream_length < 1:
        raise ValueError("Full stream length must be positive")

    if not 0 <= step_index < full_stream_length:
        raise ValueError("Step index is outside the full stream")

    if full_stream_length == 1:
        return 0.0

    return float(step_index / (full_stream_length - 1))


def build_state_for_mode(
    base_state,
    state_mode: str,
    step_index: int,
    full_stream_length: int,
) -> list[float]:
    """Preserve the base state or append normalized stream progress."""
    state = np.asarray(base_state, dtype=np.float64)

    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Base state must be a finite one-dimensional vector")

    result = state.tolist()

    if state_mode == "baseline":
        return result

    if state_mode == "progress":
        progress = normalized_stream_progress(
            step_index,
            full_stream_length,
        )
        return result + [progress]

    raise ValueError("state_mode must be 'baseline' or 'progress'")


def build_paired_dqn_controller(
    base_state_dim: int,
    state_mode: str,
    seed: int,
) -> tuple[DQNNetworks, dict[str, Any]]:
    """Initialize baseline and progress controllers from identical weights.

    The progress controller receives one additional input weight column.
    That column is initialized to zero, so its initial Q-function equals
    the baseline Q-function for every progress value.
    """
    if state_mode not in ("baseline", "progress"):
        raise ValueError("Invalid state mode")

    if (
        not isinstance(base_state_dim, int)
        or isinstance(base_state_dim, bool)
        or base_state_dim <= 0
    ):
        raise ValueError("base_state_dim must be positive")

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    controller_seed = seed + 1_000_003

    # Isolate controller initialization from task-model RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(controller_seed)

        baseline_agent = DQNNetworks(
            state_dim=base_state_dim,
            action_dim=24,
            hidden_dim=128,
        )

        baseline_state = {
            name: tensor.detach().clone()
            for name, tensor in
            baseline_agent.online_network.state_dict().items()
        }

        baseline_hash = state_dict_hash(baseline_state)

        if state_mode == "baseline":
            return baseline_agent, {
                "controller_init_seed": controller_seed,
                "base_controller_initial_sha256": baseline_hash,
                "projected_controller_initial_sha256": baseline_hash,
                "shared_weight_equivalence": True,
                "progress_input_weights_zero_at_initialization": None,
            }

        progress_agent = DQNNetworks(
            state_dim=base_state_dim + 1,
            action_dim=24,
            hidden_dim=128,
        )

        expanded_state = {
            name: tensor.detach().clone()
            for name, tensor in
            progress_agent.online_network.state_dict().items()
        }

        first_weight_name = "network.0.weight"

        for name, base_tensor in baseline_state.items():
            if name == first_weight_name:
                if expanded_state[name].shape[1] != base_state_dim + 1:
                    raise RuntimeError("Unexpected augmented input dimension")

                expanded_state[name][:, :base_state_dim].copy_(base_tensor)
                expanded_state[name][:, base_state_dim].zero_()
            else:
                if expanded_state[name].shape != base_tensor.shape:
                    raise RuntimeError(
                        f"Unexpected parameter shape for {name}"
                    )
                expanded_state[name].copy_(base_tensor)

        progress_agent.online_network.load_state_dict(expanded_state)

        # Ensure the target network starts from the augmented online network.
        progress_agent.update_target_network()

        projected_state = {}
        for name, tensor in progress_agent.online_network.state_dict().items():
            if name == first_weight_name:
                projected_state[name] = (
                    tensor[:, :base_state_dim].detach().clone()
                )
            else:
                projected_state[name] = tensor.detach().clone()

        projected_hash = state_dict_hash(projected_state)

        equivalent = all(
            torch.equal(baseline_state[name], projected_state[name])
            for name in baseline_state
        )

        progress_weights_zero = bool(
            torch.count_nonzero(
                progress_agent.online_network.network[0].weight[:, base_state_dim]
            ).item() == 0
        )

        if not equivalent or not progress_weights_zero:
            raise RuntimeError(
                "Paired controller initialization verification failed"
            )

        return progress_agent, {
            "controller_init_seed": controller_seed,
            "base_controller_initial_sha256": baseline_hash,
            "projected_controller_initial_sha256": projected_hash,
            "shared_weight_equivalence": equivalent,
            "progress_input_weights_zero_at_initialization": progress_weights_zero,
        }


def select_masked_dqn_action(
    agent: DQNNetworks,
    state,
    feasible_mask,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Masked DQN action selection using an independent, fixed-draw RNG."""
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise ValueError("epsilon must be numeric")

    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")

    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    mask_tensor = torch.as_tensor(feasible_mask)

    expected_dim = agent.online_network.network[0].in_features

    if state_tensor.shape != (expected_dim,):
        raise ValueError("State dimension does not match DQN input")

    if not torch.isfinite(state_tensor).all():
        raise ValueError("State contains non-finite values")

    if mask_tensor.shape != (24,) or mask_tensor.dtype != torch.bool:
        raise ValueError("Feasible mask must be 24 Boolean values")

    feasible_indices = torch.where(mask_tensor)[0].tolist()

    if not feasible_indices:
        raise ValueError("At least one action must be feasible")

    # Exactly one exploration draw per step.
    exploration_draw = rng.random()

    if exploration_draw < epsilon:
        # Exactly one additional draw when exploring, independent
        # of the number of feasible actions.
        choice_draw = rng.random()
        slot = min(
            int(choice_draw * len(feasible_indices)),
            len(feasible_indices) - 1,
        )
        return int(feasible_indices[slot])

    with torch.no_grad():
        q_values = agent.online_network(state_tensor)
        masked_q_values = agent.online_network.mask_q_values(
            q_values,
            mask_tensor,
        )
        return int(torch.argmax(masked_q_values).item())



def build_comparison_action_mask(
    environment: Environment,
    action_manager: ActionManager,
    memory: MemoryManager,
) -> list[bool]:
    """Shared V1 feasibility mask applied identically to all policy arms.

    Adds three productive-action constraints to the V2 resource-feasibility
    mask and retains V2's pilot convention that DISCARD is masked when memory
    is empty. STORE/REPLACE are prohibited without a label because the current
    execution path does not provide a later way to label stored unlabelled
    records. Positive compute without a label is prohibited while memory is
    empty because no replay data would exist. Label + DISCARD + tier 0 is also
    prohibited because it spends a label without learning or retaining it.
    """
    mask = list(environment.build_action_mask())
    if len(mask) != 24 or any(type(x) is not bool for x in mask):
        raise RuntimeError("Environment returned an invalid 24-action mask")

    memory_empty = memory.get_memory_status()["used"] == 0
    actions = action_manager.generate_actions()
    for index, action in enumerate(actions):
        if action["label"] == 0 and action["memory"] in ("STORE", "REPLACE"):
            mask[index] = False
        if memory_empty and action["memory"] == "DISCARD":
            mask[index] = False
        if memory_empty and action["label"] == 0 and action["compute"] > 0:
            mask[index] = False
        if (
            action["label"] == 1
            and action["memory"] == "DISCARD"
            and action["compute"] == 0
        ):
            mask[index] = False

    if not any(mask):
        raise RuntimeError(
            "No feasible actions remain under the shared comparison mask"
        )
    return mask


def evenly_spaced_positions(total_steps: int, count: int) -> set[int]:
    """Deterministic left-anchored schedule; includes step 0 when count > 0."""
    if total_steps < 0 or count < 0:
        raise ValueError("total_steps and count must be non-negative")
    if total_steps == 0 or count == 0:
        return set()
    count = min(total_steps, count)
    # Integer arithmetic avoids platform-dependent floating rounding.
    return {(i * total_steps) // count for i in range(count)}


def build_heuristic_schedule(
    total_steps: int,
    label_budget: int = INITIAL_LABEL_BUDGET,
    compute_budget: int = INITIAL_COMPUTE_BUDGET,
) -> dict[str, list[int]]:
    """Predeclare label and tier-1 compute positions before a run begins."""
    label_count = min(total_steps, label_budget)
    label_positions = evenly_spaced_positions(total_steps, label_count)

    tier1_cost = COMPUTE_COSTS[1]
    compute_count = min(total_steps, compute_budget // tier1_cost)
    # Predeclare compute events at interior, evenly spaced stream positions.
    # This avoids front-loading the entire task-compute budget at stream start.
    compute_positions = {
        ((i + 1) * total_steps) // (compute_count + 1)
        for i in range(compute_count)
    }
    compute_positions = {p for p in compute_positions if 0 <= p < total_steps}

    return {
        "label_positions": sorted(label_positions),
        "tier1_compute_positions": sorted(compute_positions),
    }


def fixed_heuristic_action(
    step_index: int,
    schedule: dict[str, list[int]],
    resources: dict[str, int],
    memory_used: int,
    memory_capacity: int,
    action_manager: ActionManager,
    feasible_mask: list[bool],
) -> int:
    """Return a predeclared action index; raises if it violates the mask."""
    label_due = (
        step_index in schedule["label_positions"]
        and resources["labels"] >= LABEL_COST
    )
    compute_due = (
        step_index in schedule["tier1_compute_positions"]
        and resources["compute"] >= COMPUTE_COSTS[1]
    )

    if label_due:
        memory_operation = "STORE" if memory_used < memory_capacity else "REPLACE"
        label_action = 1
    else:
        memory_operation = "DISCARD"
        label_action = 0

    compute_tier = 1 if compute_due else 0
    candidates = action_manager.generate_actions()
    candidate = next(
        a for a in candidates
        if a["label"] == label_action
        and a["memory"] == memory_operation
        and a["compute"] == compute_tier
    )
    action_index = action_manager.get_network_index(candidate["id"])
    if not feasible_mask[action_index]:
        raise RuntimeError(
            "Predeclared fixed-heuristic action is infeasible at step "
            f"{step_index + 1}: {candidate}"
        )
    return action_index


def _check_v2_unchanged() -> str:
    v2_path = PROJECT_ROOT / "experiments" / "open_world_causal_dqn_pilot_v2.py"
    if not v2_path.is_file():
        raise FileNotFoundError(f"Frozen V2 source is missing: {v2_path}")
    actual = sha256_file(v2_path)
    if actual != EXPECTED_V2_SHA256:
        raise RuntimeError(
            "Frozen V2 source hash mismatch. Do not run comparison until the "
            "original V2 file is restored or the expected hash is reviewed. "
            f"Expected {EXPECTED_V2_SHA256}; found {actual}."
        )
    return actual


def _prepare_stream(
    data_dir: Path, seed: int, max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, tuple[np.ndarray, np.ndarray]],
           list[dict[str, Any]], dict[str, str], str]:
    paths = [data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset batches:\n" + "\n".join(missing))

    batches = [v2.base.load_batch(p) for p in paths]
    prepared, _, _ = v2.make_partitions(batches, seed)

    stream: list[dict[str, Any]] = []
    label_map: dict[str, int] = {}
    batch_validation: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    stream_rng = np.random.default_rng(seed * 1_000_003 + 902)
    split_manifest: list[dict[str, Any]] = []

    for part in prepared:
        batch_number = part["batch"]
        learn_idx = np.asarray(part["learning_idx"]).copy()
        stream_rng.shuffle(learn_idx)
        for row_index in learn_idx:
            sample_id = f"b{batch_number}_r{int(row_index)}"
            label_map[sample_id] = int(part["y_raw"][row_index])
            stream.append({
                "sample_id": sample_id,
                "batch": batch_number,
                "row_index": int(row_index),
                "raw": part["x_raw"][row_index].astype(np.float64),
                "scaled": part["x_scaled"][row_index].astype(np.float32),
            })
        val_idx = np.asarray(part["validation_idx"])
        batch_validation[batch_number] = (
            part["x_scaled"][val_idx].astype(np.float32),
            part["y_raw"][val_idx].astype(np.int64),
        )
        split_manifest.append({
            "batch": batch_number,
            "learning_rows": np.asarray(part["learning_idx"]).astype(int).tolist(),
            "validation_rows": np.asarray(part["validation_idx"]).astype(int).tolist(),
            "heldout_rows": np.asarray(part["test_idx"]).astype(int).tolist(),
        })

    full_stream_length = len(stream)
    if max_steps > 0:
        stream = stream[:max_steps]
    if not stream:
        raise ValueError("No learning-stream samples available")

    data_hashes = {p.name: sha256_file(p) for p in paths}
    stream_signature = stable_json_hash([s["sample_id"] for s in stream])
    return (
        stream, label_map, batch_validation, prepared, data_hashes,
        stable_json_hash({"splits": split_manifest, "stream_signature": stream_signature,
                          "full_stream_length": full_stream_length}),
    )


def _evaluate_stage_validation(
    model: torch.nn.Module,
    batch_validation: dict[int, tuple[np.ndarray, np.ndarray]],
    upto_batch: int,
) -> list[dict[str, Any]]:
    output = []
    for batch in range(1, upto_batch + 1):
        x, y = batch_validation[batch]
        metrics = v2.evaluate_model(model, x, y)
        output.append({"batch": batch, **metrics})
    return output


def _heldout_test_metrics(
    model: torch.nn.Module,
    prepared: list[dict[str, Any]],
) -> dict[str, Any]:
    per_batch = {}
    all_y, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for part in prepared:
            idx = np.asarray(part["test_idx"])
            x = part["x_scaled"][idx].astype(np.float32)
            y = part["y_raw"][idx].astype(np.int64)
            logits = model(torch.as_tensor(x, dtype=torch.float32))
            pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64) + 1
            per_batch[str(part["batch"])] = {
                "samples": int(len(y)),
                "accuracy": float(np.mean(pred == y)),
                "macro_f1_fixed_six": v2.macro_f1_fixed_six(y, pred),
            }
            all_y.append(y)
            all_pred.append(pred)
    pooled_y = np.concatenate(all_y)
    pooled_pred = np.concatenate(all_pred)
    return {
        "pooled": {
            "samples": int(len(pooled_y)),
            "accuracy": float(np.mean(pooled_pred == pooled_y)),
            "macro_f1_fixed_six": v2.macro_f1_fixed_six(pooled_y, pooled_pred),
        },
        "per_batch": per_batch,
    }


def _forgetting_proxy(stage_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Validation-only stage-boundary forgetting proxy, not historical E2/E3 F."""
    by_batch: dict[int, list[float]] = {}
    for row in stage_rows:
        by_batch.setdefault(int(row["batch"]), []).append(
            float(row["macro_f1_fixed_six"])
        )
    result = {}
    for batch, scores in by_batch.items():
        if len(scores) >= 2:
            result[str(batch)] = max(scores) - scores[-1]
    return result


def build_budget_calibration_schedule(
    total_steps: int,
    compute_credits: int,
) -> dict[str, Any]:
    """Create a predeclared, collision-free label/compute event schedule.

    Each compute event is tier 1, whose current pilot mapping is one task
    optimizer update. All compute events occur after the first MEMORY_CAPACITY
    label events, and never coincide with a label event.
    """
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if compute_credits not in BUDGET_CREDITS:
        raise ValueError(f"compute_credits must be one of {BUDGET_CREDITS}")
    if total_steps < INITIAL_LABEL_BUDGET:
        raise ValueError("Stream is too short for the declared label schedule")
    if COMPUTE_COSTS[1] <= 0 or v2.COMPUTE_TO_UPDATES[1] != 1:
        raise RuntimeError("Tier-1 cost/update mapping is inconsistent")

    label_positions = set(
        evenly_spaced_positions(
            total_steps,
            min(total_steps, INITIAL_LABEL_BUDGET),
        )
    )
    if len(label_positions) != INITIAL_LABEL_BUDGET:
        raise RuntimeError("Unexpected number of label positions")

    first_memory_labels = sorted(label_positions)[:MEMORY_CAPACITY]
    warmup_end_index = max(first_memory_labels) + 1
    eligible_positions = [
        index
        for index in range(warmup_end_index, total_steps)
        if index not in label_positions
    ]
    if len(eligible_positions) < compute_credits:
        raise ValueError("Not enough eligible positions for compute events")

    # Include the first eligible position and the last stream position, with
    # integer interpolation over the eligible-position list.
    compute_positions = [
        eligible_positions[
            (i * (len(eligible_positions) - 1)) // (compute_credits - 1)
        ]
        for i in range(compute_credits)
    ]

    if len(set(compute_positions)) != compute_credits:
        raise RuntimeError("Compute schedule contains duplicate positions")
    if set(compute_positions) & label_positions:
        raise RuntimeError("Compute and label schedules overlap")
    if min(compute_positions) < warmup_end_index:
        raise RuntimeError("Compute event occurs before memory warm-up")

    return {
        "label_positions_zero_based": sorted(label_positions),
        "compute_positions_zero_based": compute_positions,
        "warmup_end_index_zero_based": warmup_end_index,
        "warmup_complete_after_step": warmup_end_index,
        "compute_credits": compute_credits,
        "tier": 1,
        "tier_cost": COMPUTE_COSTS[1],
        "optimizer_updates_per_event": v2.COMPUTE_TO_UPDATES[1],
        "initial_compute_budget": compute_credits * COMPUTE_COSTS[1],
        "label_compute_overlap_count": 0,
        "replay_pool_size_at_compute_events": MEMORY_CAPACITY,
        "scheduler_version": SCHEDULER_VERSION,
    }


def _fixed_schedule_action(
    action_manager: ActionManager,
    label_action: int,
    memory_operation: str,
    compute_tier: int,
) -> dict[str, Any]:
    matches = [
        action for action in action_manager.generate_actions()
        if action["label"] == label_action
        and action["memory"] == memory_operation
        and action["compute"] == compute_tier
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one joint action for the fixed schedule; "
            f"found {len(matches)} for {(label_action, memory_operation, compute_tier)}"
        )
    return matches[0]


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.compute_credits not in BUDGET_CREDITS:
        raise ValueError(f"compute_credits must be one of {BUDGET_CREDITS}")
    if args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    v2_hash = _check_v2_unchanged()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    (stream, label_map, batch_validation, prepared, data_hashes,
     split_signature) = _prepare_stream(data_dir, args.seed, args.max_steps)
    full_stream_length = sum(len(part["learning_idx"]) for part in prepared)
    is_full_stream = len(stream) == full_stream_length

    schedule = build_budget_calibration_schedule(
        full_stream_length,
        args.compute_credits,
    )
    label_positions = set(schedule["label_positions_zero_based"])
    compute_positions = set(schedule["compute_positions_zero_based"])

    initial_compute_budget = schedule["initial_compute_budget"]
    resources = ResourceManager(
        label_budget=INITIAL_LABEL_BUDGET,
        compute_budget=initial_compute_budget,
    )
    label_provider = v2.LearningOnlyLabelProvider(label_map)
    memory = MemoryManager(capacity=MEMORY_CAPACITY)
    action_manager = ActionManager()

    feature_extractor = FeatureExtractor(
        lambda row: np.asarray(row, dtype=np.float64).tolist()
    )
    environment = Environment(
        feature_extractor=feature_extractor,
        state_builder=StateBuilder(),
        state_encoder=StateEncoder(v2.N_FEATURES, history_size=v2.HISTORY_SIZE),
        memory_manager=memory,
        action_manager=action_manager,
        resource_manager=resources,
        reward_calculator=None,
        prediction_provider=None,
    )

    learner_model = v2.base.SmallMLP()
    task_initial_hash = state_dict_hash(learner_model.state_dict())
    hybrid_trainer = HybridBatchTrainer(
        learner_model,
        learning_rate=args.learning_rate,
        device="cpu",
        input_dim=v2.N_FEATURES,
        num_classes=v2.N_CLASSES,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "budget_calibration_e5b"
        / f"credits_{args.compute_credits}"
        / f"seed_{args.seed}"
        / (f"smoke_{args.max_steps}" if args.max_steps > 0 else "full")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    labels_acquired = 0
    task_optimizer_updates = 0
    compute_events_processed = 0
    memory_operation_counts = {"DISCARD": 0, "STORE": 0, "REPLACE": 0}
    timing = {
        "action_execution_seconds": 0.0,
        "task_training_seconds": 0.0,
        "stage_validation_seconds": 0.0,
        "heldout_test_seconds": 0.0,
        "stream_loop_wall_seconds": 0.0,
    }
    end_to_end_started = time.perf_counter()

    for step_number, sample in enumerate(stream):
        step_started = time.perf_counter()
        batch_number = int(sample["batch"])
        scaled_features = np.asarray(sample["scaled"], dtype=np.float32)
        sample_id = sample["sample_id"]

        label_due = step_number in label_positions
        compute_due = step_number in compute_positions
        if label_due and compute_due:
            raise RuntimeError("Scheduled label and compute events overlap")

        memory_used_before = memory.get_memory_status()["used"]
        if label_due:
            memory_operation = (
                "STORE" if memory_used_before < MEMORY_CAPACITY else "REPLACE"
            )
            label_action = 1
            compute_tier = 0
        elif compute_due:
            memory_operation = "DISCARD"
            label_action = 0
            compute_tier = 1
        else:
            memory_operation = "DISCARD"
            label_action = 0
            compute_tier = 0

        action = _fixed_schedule_action(
            action_manager,
            label_action,
            memory_operation,
            compute_tier,
        )
        feasible_mask = build_comparison_action_mask(
            environment,
            action_manager,
            memory,
        )
        network_index = action_manager.get_network_index(action["id"])
        if not feasible_mask[network_index]:
            raise RuntimeError(
                f"Fixed schedule selected an infeasible action at step {step_number + 1}: {action}"
            )

        labelled_replay: list[dict[str, Any]] = []
        replay_sample_ids: list[str] = []
        if compute_due:
            for record in memory.get_memory_records():
                item = record.get("item", {})
                if isinstance(item, dict) and item.get("label") is not None:
                    labelled_replay.append({
                        "id": record["id"],
                        "item": {
                            "features": item["features"],
                            "label": int(item["label"]),
                        },
                    })
                    replay_sample_ids.append(str(item.get("sample_id", record["id"])))
            if len(labelled_replay) != MEMORY_CAPACITY:
                raise RuntimeError(
                    "Compute event requires a full labelled replay memory: "
                    f"expected {MEMORY_CAPACITY}, found {len(labelled_replay)}"
                )

        item_for_memory = {
            "features": scaled_features.tolist(),
            "sample_id": sample_id,
            "batch": batch_number,
        }
        action_started = time.perf_counter()
        execution = environment.execute_agent_action(
            action=action,
            item=item_for_memory if memory_operation in ("STORE", "REPLACE") else None,
            index=0 if memory_operation in ("DISCARD", "REPLACE") else None,
            sample_id=sample_id if label_action == 1 else None,
            label_provider=label_provider if label_action == 1 else None,
        )
        action_seconds = time.perf_counter() - action_started
        timing["action_execution_seconds"] += action_seconds
        memory_operation_counts[memory_operation] += 1

        acquired_label = execution["acquired_label"]
        if label_due:
            if acquired_label is None:
                raise RuntimeError("Scheduled annotation event did not acquire a label")
            labels_acquired += 1
        elif acquired_label is not None:
            raise RuntimeError("Unscheduled label acquisition occurred")

        update_result = {
            "completed_optimizer_updates": 0,
            "mean_loss": None,
            "replay_presentations": 0,
            "total_presentations": 0,
        }
        task_seconds = 0.0
        if compute_due:
            task_started = time.perf_counter()
            update_result = hybrid_trainer.train_batch(
                current_features=[],
                current_labels=[],
                replay_records=labelled_replay,
                optimizer_updates=1,
                stage=batch_number,
                seed=args.seed + step_number,
            )
            task_seconds = time.perf_counter() - task_started
            timing["task_training_seconds"] += task_seconds
            if update_result["completed_optimizer_updates"] != 1:
                raise RuntimeError(
                    "A scheduled compute event failed to complete exactly one optimizer update"
                )
            if update_result["current_presentations"] != 0:
                raise RuntimeError("Replay-only update unexpectedly used current samples")
            task_optimizer_updates += int(update_result["completed_optimizer_updates"])
            compute_events_processed += 1

        rows.append({
            "step": step_number + 1,
            "batch": batch_number,
            "sample_id": sample_id,
            "label_event": int(label_due),
            "compute_event": int(compute_due),
            "action_id": action["id"],
            "label_action": label_action,
            "memory_operation": memory_operation,
            "compute_tier": compute_tier,
            "acquired_label": acquired_label if acquired_label is not None else "",
            "replay_pool_size": len(labelled_replay),
            "replay_pool_sample_ids": json.dumps(replay_sample_ids),
            "replay_presentations": int(update_result["replay_presentations"]),
            "task_optimizer_updates": int(update_result["completed_optimizer_updates"]),
            "task_mean_loss": update_result["mean_loss"] if update_result["mean_loss"] is not None else "",
            "memory_used": memory.get_memory_status()["used"],
            "labels_remaining": resources.get_resources()["labels"],
            "compute_remaining": resources.get_resources()["compute"],
            "action_execution_seconds": action_seconds,
            "task_training_seconds": task_seconds,
        })

        has_next = step_number + 1 < len(stream)
        boundary = has_next and int(stream[step_number + 1]["batch"]) != batch_number
        final_full_boundary = (step_number == len(stream) - 1) and is_full_stream
        if boundary or final_full_boundary:
            stage_started = time.perf_counter()
            stage_eval = _evaluate_stage_validation(
                hybrid_trainer.model,
                batch_validation,
                batch_number,
            )
            timing["stage_validation_seconds"] += time.perf_counter() - stage_started
            for metric in stage_eval:
                stage_rows.append({"after_batch": batch_number, **metric})

        timing["stream_loop_wall_seconds"] += time.perf_counter() - step_started

    heldout_metrics = None
    if is_full_stream:
        test_started = time.perf_counter()
        heldout_metrics = _heldout_test_metrics(hybrid_trainer.model, prepared)
        timing["heldout_test_seconds"] += time.perf_counter() - test_started

    end_to_end_seconds = time.perf_counter() - end_to_end_started

    steps_path = output_dir / "steps.csv"
    with steps_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stage_path = output_dir / "stage_validation_trajectory.csv"
    if stage_rows:
        with stage_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage_rows)

    task_model_path = output_dir / "task_model_final.pt"
    torch.save(hybrid_trainer.model.state_dict(), task_model_path)

    memory_final = memory.get_memory_status()["used"]
    resources_final = resources.get_resources()
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "v2_id": V2_ID,
        "status": "completed_full_stream_single_seed" if is_full_stream else "completed_smoke_run",
        "seed": args.seed,
        "steps_processed": len(rows),
        "full_stream_steps": full_stream_length,
        "full_stream_evaluated": bool(is_full_stream),
        "policy": "fixed_reference_schedule",
        "adaptive_controller_used": False,
        "budget_condition": {
            "compute_update_credits": args.compute_credits,
            "tier": 1,
            "cost_per_event": COMPUTE_COSTS[1],
            "initial_compute_budget": initial_compute_budget,
            "planned_compute_events": args.compute_credits,
            "planned_task_optimizer_updates": args.compute_credits,
            "compute_events_processed": compute_events_processed,
            "completed_task_optimizer_updates": task_optimizer_updates,
            "compute_budget_remaining": resources_final["compute"],
            "compute_budget_consumed": initial_compute_budget - resources_final["compute"],
            "unit_interpretation": "provisional tier-1 optimizer-update credits; not FLOPs, energy, or hardware-independent compute",
        },
        "annotation_memory_protocol": {
            "initial_label_budget": INITIAL_LABEL_BUDGET,
            "labels_acquired": labels_acquired,
            "labels_remaining": resources_final["labels"],
            "memory_capacity_records": MEMORY_CAPACITY,
            "memory_final_used": memory_final,
            "memory_operations": memory_operation_counts,
            "label_event_steps_one_based": [p + 1 for p in schedule["label_positions_zero_based"]],
            "memory_warmup_complete_after_step": schedule["warmup_complete_after_step"],
            "compute_event_steps_one_based": [p + 1 for p in schedule["compute_positions_zero_based"]],
            "label_compute_overlap_count": schedule["label_compute_overlap_count"],
            "replay_pool_records_per_compute_event": MEMORY_CAPACITY,
            "replay_sampling": "10 records sampled with replacement per optimizer update by HybridBatchTrainer's replay-only path",
            "replacement_rule": "fixed memory slot 0, matching the existing pilot environment",
        },
        "scheduler": schedule,
        "task_learner_configuration": {
            "architecture": "SmallMLP 128-64-6",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "device": "CPU",
            "training_source": "persistent labelled memory only at scheduled compute events",
        },
        "task_initialization": {
            "initial_task_model_sha256": task_initial_hash,
        },
        "stage_validation_trajectory": stage_rows,
        "validation_forgetting_proxy_macro_f1": _forgetting_proxy(stage_rows),
        "heldout_test": heldout_metrics,
        "data_provenance": {
            "split_manifest_sha256": split_signature,
            "dataset_batch_sha256": data_hashes,
            "v2_source_sha256": v2_hash,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing_seconds": {
            **timing,
            "end_to_end_run_seconds": end_to_end_seconds,
        },
        "limitations": [
            "This is a fixed-schedule compute-dose calibration, not an adaptive RL policy comparison.",
            "Tier-1 costs and update credits are provisional proxies, not measured FLOPs, energy, or hardware-independent compute.",
            "Six known classes only; unknown-class discovery/open-set recognition is not evaluated.",
            "Memory capacity is measured in records, not bytes or RAM.",
            "The first ten scheduled labels fill persistent memory before any compute event; all task updates are replay-only.",
            "Replay sampling is with replacement; each optimizer update draws ten presentations from the ten-record memory pool.",
            "The fixed memory replacement rule overwrites slot 0; victim selection is not evaluated.",
            "The 100-label schedule uses labels from the existing stratified learning split; held-out test labels are not used for training.",
            "A smoke run is only an execution-path check and is not evidence of a budget-performance relationship.",
        ],
        "artifacts": {
            "steps_csv": str(steps_path),
            "stage_validation_csv": str(stage_path) if stage_rows else None,
            "task_model": str(task_model_path),
            "controller_checkpoint": None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument(
        "--compute-credits",
        type=int,
        choices=BUDGET_CREDITS,
        required=True,
        help="Number of predeclared tier-1 task-optimizer update credits: 4, 40, or 100.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2010,
        help="Positive value runs a smoke prefix; 0 processes the full stream.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Task-learner Adam learning rate.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/gas_sensor_array_drift/Dataset",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.output_dir is None:
        cli_args.output_dir = (
            f"experiments/results/budget_calibration_e5b/credits_{cli_args.compute_credits}/seed_{cli_args.seed}"
            + (f"_smoke_{cli_args.max_steps}" if cli_args.max_steps > 0 else "_full")
        )
    run_one(cli_args)
