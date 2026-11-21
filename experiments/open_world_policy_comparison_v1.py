"""Controlled policy comparison for Open-World AI.

Experiment ID: CDQN-POLICY-COMP-01-V1

Compares three policies through the unchanged CDQN-PILOT-01-V2 environment
semantics: learned feasibility-masked DQN, uniformly random feasible action,
and a deterministic predeclared schedule heuristic.

This runner is intentionally a separate experiment. It imports shared helpers
from the frozen V2 pilot but never edits that source file.
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

EXPERIMENT_ID = "CDQN-POLICY-COMP-01-V1"
V2_ID = "CDQN-PILOT-01-V2"
EXPECTED_V2_SHA256 = "6a2863258198f79eca09c682fe3523ab78f0d4b01b83591b0f0cb2e665870a61"
POLICIES = ("dqn", "random_feasible", "fixed_heuristic")


class V1RingReplayBuffer:
    """V1-local FIFO replay buffer with O(1) append and uniform sampling.

    It preserves the V2 replay-buffer semantics: bounded FIFO replacement and
    uniform sampling without replacement. Unlike V2's deque-to-list conversion,
    this ring representation samples directly by index, avoiding a full buffer
    copy on every controller update. It is local to this experiment; V2's
    ReplayBuffer class and source remain unchanged.
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity = capacity
        self._items: list[Any] = [None] * capacity
        self._size = 0
        self._next = 0

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
        indices = random.sample(range(self._size), batch_size)
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


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    v2_hash = _check_v2_unchanged()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    checkpoint = Path(args.prediction_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Prediction checkpoint not found: {checkpoint}")

    (stream, label_map, batch_validation, prepared, data_hashes,
     split_signature) = _prepare_stream(data_dir, args.seed, args.max_steps)
    full_stream_length = sum(len(part["learning_idx"]) for part in prepared)
    is_full_stream = len(stream) == full_stream_length

    label_provider = v2.LearningOnlyLabelProvider(label_map)
    prediction_provider = PredictionProvider(str(checkpoint))
    memory = MemoryManager(capacity=MEMORY_CAPACITY)
    resources = ResourceManager(
        label_budget=INITIAL_LABEL_BUDGET,
        compute_budget=INITIAL_COMPUTE_BUDGET,
    )
    action_manager = ActionManager()
    feature_extractor = FeatureExtractor(
        lambda row: np.asarray(row, dtype=np.float64).tolist()
    )
    state_builder = StateBuilder()
    state_encoder = StateEncoder(v2.N_FEATURES, history_size=v2.HISTORY_SIZE)

    # This is initialized before the DQN controller so task initialization is
    # byte-identical across all three policies for the same seed.
    learner_model = v2.base.SmallMLP()
    task_initial_hash = state_dict_hash(learner_model.state_dict())
    hybrid_trainer = HybridBatchTrainer(
        learner_model,
        learning_rate=args.learning_rate,
        device="cpu",
        input_dim=v2.N_FEATURES,
        num_classes=v2.N_CLASSES,
    )
    reward_calculator = RewardCalculator(
        forgetting_weight=0.0,
        label_cost_weight=args.label_cost_weight,
        compute_cost_weight=args.compute_cost_weight,
        memory_cost_weight=args.memory_cost_weight,
    )
    environment = Environment(
        feature_extractor=feature_extractor,
        state_builder=state_builder,
        state_encoder=state_encoder,
        memory_manager=memory,
        action_manager=action_manager,
        resource_manager=resources,
        reward_calculator=reward_calculator,
        prediction_provider=prediction_provider,
    )

    dqn_agent = dqn_trainer = replay_buffer = None
    if args.policy == "dqn":
        replay_buffer = V1RingReplayBuffer(capacity=args.dqn_replay_capacity)
        dqn_agent = DQNNetworks(
            state_dim=state_encoder.state_dim,
            action_dim=24,
            hidden_dim=128,
        )
        dqn_trainer = DQNTrainer(
            dqn_agent,
            replay_buffer,
            batch_size=args.dqn_batch_size,
            gamma=args.gamma,
            target_update_frequency=args.target_update_frequency,
        )

    # A separate RNG prevents random-policy draws from changing task-training
    # randomness or any unrelated Python random stream.
    policy_rng = random.Random(args.seed + 91_337)
    heuristic_schedule = build_heuristic_schedule(
        full_stream_length, INITIAL_LABEL_BUDGET, INITIAL_COMPUTE_BUDGET
    )

    history: list[float] = []
    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    dqn_losses: list[float] = []
    acquired_labels = 0
    task_updates = 0
    memory_operation_counts = {"DISCARD": 0, "STORE": 0, "REPLACE": 0}
    timing = {
        "state_prediction_seconds": 0.0,
        "step_validation_seconds": 0.0,
        "stage_validation_seconds": 0.0,
        "heldout_test_seconds": 0.0,
        "policy_selection_seconds": 0.0,
        "action_execution_seconds": 0.0,
        "task_training_seconds": 0.0,
        "dqn_training_seconds": 0.0,
        "stream_loop_wall_seconds": 0.0,
    }
    end_to_end_started = time.perf_counter()

    for step_number, sample in enumerate(stream):
        step_started = time.perf_counter()
        batch_number = int(sample["batch"])
        raw_features = sample["raw"]
        scaled_features = sample["scaled"]
        sample_id = sample["sample_id"]

        current_resources = resources.get_resources()
        state_started = time.perf_counter()
        current_state = v2.build_pilot_state(
            environment, raw_features, scaled_features,
            current_resources, history[-v2.HISTORY_SIZE:],
        )
        timing["state_prediction_seconds"] += time.perf_counter() - state_started
        current_mask = build_comparison_action_mask(
            environment, action_manager, memory
        )

        validation_x, validation_y = batch_validation[batch_number]
        val_started = time.perf_counter()
        utility_before = v2.evaluate_model(
            hybrid_trainer.model, validation_x, validation_y
        )["macro_f1_fixed_six"]
        validation_seconds = time.perf_counter() - val_started

        progress = step_number / max(1, len(stream) - 1)
        epsilon_now = args.epsilon_start + (
            args.epsilon_end - args.epsilon_start
        ) * progress
        policy_started = time.perf_counter()
        if args.policy == "dqn":
            assert dqn_agent is not None
            selected_index = dqn_agent.select_action(
                current_state, current_mask, epsilon=epsilon_now
            )
        elif args.policy == "random_feasible":
            feasible_indices = [i for i, valid in enumerate(current_mask) if valid]
            selected_index = policy_rng.choice(feasible_indices)
        else:
            selected_index = fixed_heuristic_action(
                step_number,
                heuristic_schedule,
                resources.get_resources(),
                memory.get_memory_status()["used"],
                MEMORY_CAPACITY,
                action_manager,
                current_mask,
            )
        policy_seconds = time.perf_counter() - policy_started
        timing["policy_selection_seconds"] += policy_seconds

        selected_action = action_manager.get_action(selected_index + 1)
        pre_action_records = memory.get_memory_records()
        labelled_replay = []
        for record in pre_action_records:
            item = record.get("item", {})
            if isinstance(item, dict) and item.get("label") is not None:
                labelled_replay.append({
                    "id": record["id"],
                    "item": {
                        "features": item["features"],
                        "label": int(item["label"]),
                    },
                })

        item_for_memory = {
            "features": scaled_features.tolist(),
            "sample_id": sample_id,
            "batch": batch_number,
        }
        action_kwargs = {
            "item": item_for_memory
            if selected_action["memory"] in ("STORE", "REPLACE") else None,
            "index": 0 if selected_action["memory"] in ("DISCARD", "REPLACE") else None,
            "sample_id": sample_id if selected_action["label"] == 1 else None,
            "label_provider": label_provider if selected_action["label"] == 1 else None,
        }
        action_started = time.perf_counter()
        execution = environment.execute_agent_action(
            action=selected_action, **action_kwargs
        )
        action_seconds = time.perf_counter() - action_started
        timing["action_execution_seconds"] += action_seconds
        memory_operation_counts[selected_action["memory"]] += 1

        acquired_label = execution["acquired_label"]
        if acquired_label is not None:
            acquired_labels += 1
            current_features = [scaled_features.tolist()]
            current_labels = [int(acquired_label)]
        else:
            current_features = []
            current_labels = []

        task_started = time.perf_counter()
        tier_updates = v2.COMPUTE_TO_UPDATES[selected_action["compute"]]
        update_result = hybrid_trainer.train_batch(
            current_features=current_features,
            current_labels=current_labels,
            replay_records=labelled_replay,
            optimizer_updates=tier_updates,
            stage=batch_number,
            seed=args.seed + step_number,
        )
        task_seconds = time.perf_counter() - task_started
        timing["task_training_seconds"] += task_seconds
        task_updates += int(update_result["completed_optimizer_updates"])

        val_started = time.perf_counter()
        utility_after = v2.evaluate_model(
            hybrid_trainer.model, validation_x, validation_y
        )["macro_f1_fixed_six"]
        validation_seconds += time.perf_counter() - val_started
        timing["step_validation_seconds"] += validation_seconds

        utility_delta = float(utility_after - utility_before)
        reward_result = environment.calculate_v1_reward(
            utility_before=utility_before,
            utility_after=utility_after,
            action=selected_action,
        )

        done = step_number == len(stream) - 1
        if done:
            next_state = list(current_state)
            next_mask = [False] * 24
        else:
            next_sample = stream[step_number + 1]
            next_state_started = time.perf_counter()
            next_state = v2.build_pilot_state(
                environment,
                next_sample["raw"],
                next_sample["scaled"],
                resources.get_resources(),
                (history + [utility_delta])[-v2.HISTORY_SIZE:],
            )
            timing["state_prediction_seconds"] += time.perf_counter() - next_state_started
            next_mask = build_comparison_action_mask(
                environment, action_manager, memory
            )

        dqn_loss = None
        if args.policy == "dqn":
            assert replay_buffer is not None and dqn_trainer is not None
            environment.store_dqn_experience(
                replay_buffer=replay_buffer,
                state=current_state,
                action_index=selected_index,
                reward_result=reward_result,
                next_state=next_state,
                done=done,
                next_feasible_mask=next_mask,
            )
            dqn_started = time.perf_counter()
            dqn_loss = dqn_trainer.train()
            timing["dqn_training_seconds"] += time.perf_counter() - dqn_started
            if dqn_loss is not None:
                dqn_losses.append(float(dqn_loss))

        history.append(utility_delta)
        rows.append({
            "step": step_number + 1,
            "policy": args.policy,
            "epsilon": epsilon_now if args.policy == "dqn" else "",
            "batch": batch_number,
            "sample_id": sample_id,
            "action_id": selected_action["id"],
            "label_action": selected_action["label"],
            "memory_operation": selected_action["memory"],
            "compute_tier": selected_action["compute"],
            "acquired_label": acquired_label if acquired_label is not None else "",
            "utility_before_macro_f1": utility_before,
            "utility_after_macro_f1": utility_after,
            "utility_delta": utility_delta,
            "reward": float(reward_result["reward"]),
            "task_optimizer_updates": int(update_result["completed_optimizer_updates"]),
            "task_mean_loss": update_result["mean_loss"] if update_result["mean_loss"] is not None else "",
            "dqn_loss": dqn_loss if dqn_loss is not None else "",
            "memory_used": memory.get_memory_status()["used"],
            "labels_remaining": resources.get_resources()["labels"],
            "compute_remaining": resources.get_resources()["compute"],
            "replay_buffer_size": len(replay_buffer) if replay_buffer is not None else 0,
            "policy_selection_seconds": policy_seconds,
            "action_execution_seconds": action_seconds,
            "task_training_seconds": task_seconds,
            "validation_seconds_step": validation_seconds,
        })

        # Measure old-batch retention only at chronological batch boundaries.
        has_next = step_number + 1 < len(stream)
        boundary = has_next and int(stream[step_number + 1]["batch"]) != batch_number
        final_full_boundary = done and is_full_stream
        if boundary or final_full_boundary:
            stage_started = time.perf_counter()
            stage_eval = _evaluate_stage_validation(
                hybrid_trainer.model, batch_validation, batch_number
            )
            elapsed = time.perf_counter() - stage_started
            timing["stage_validation_seconds"] += elapsed
            for metric in stage_eval:
                stage_rows.append({"after_batch": batch_number, **metric})

        timing["stream_loop_wall_seconds"] += time.perf_counter() - step_started

    heldout_metrics = None
    if is_full_stream:
        test_started = time.perf_counter()
        heldout_metrics = _heldout_test_metrics(hybrid_trainer.model, prepared)
        timing["heldout_test_seconds"] += time.perf_counter() - test_started

    end_to_end_seconds = time.perf_counter() - end_to_end_started
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    dqn_training_steps = dqn_trainer.training_steps if dqn_trainer is not None else 0
    dqn_model_path = None
    if dqn_agent is not None:
        dqn_model_path = output_dir / "dqn_controller_final.pt"
        torch.save({
            "online_network": dqn_agent.online_network.state_dict(),
            "target_network": dqn_agent.target_network.state_dict(),
        }, dqn_model_path)

    stage_by_batch: dict[str, list[float]] = {}
    for entry in stage_rows:
        stage_by_batch.setdefault(str(entry["batch"]), []).append(
            float(entry["macro_f1_fixed_six"])
        )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "v2_id": V2_ID,
        "policy": args.policy,
        "status": "completed_full_stream_single_seed" if is_full_stream else "completed_smoke_run",
        "seed": args.seed,
        "steps_processed": len(rows),
        "full_stream_steps": full_stream_length,
        "full_stream_evaluated": bool(is_full_stream),
        "label_budget_initial": INITIAL_LABEL_BUDGET,
        "labels_acquired": acquired_labels,
        "label_budget_remaining": resources.get_resources()["labels"],
        "compute_budget_initial": INITIAL_COMPUTE_BUDGET,
        "compute_budget_remaining": resources.get_resources()["compute"],
        "compute_budget_consumed": INITIAL_COMPUTE_BUDGET - resources.get_resources()["compute"],
        "compute_budget_scope": "task-learner compute tier costs only; policy/controller, state-prediction and evaluator costs are measured separately",
        "task_optimizer_updates": task_updates,
        "compute_update_mapping": v2.COMPUTE_TO_UPDATES,
        "compute_update_mapping_status": "provisional pilot mapping; not FLOPs, energy, or hardware-independent compute",
        "memory_capacity_examples": MEMORY_CAPACITY,
        "memory_final_used": memory.get_memory_status()["used"],
        "memory_operations": memory_operation_counts,
        "dqn_training_steps": dqn_training_steps,
        "dqn_mean_training_loss": float(np.mean(dqn_losses)) if dqn_losses else None,
        "epsilon_schedule": {
            "start": args.epsilon_start,
            "end": args.epsilon_end,
            "shape": "linear across processed stream; DQN only",
        },
        "fixed_heuristic_schedule": heuristic_schedule if args.policy == "fixed_heuristic" else None,
        "task_learner_configuration": {
            "architecture": "SmallMLP 128-64-6",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "device": "CPU",
        },
        "dqn_controller_configuration": ({
            "architecture": "DQN 128-hidden, 24 actions",
            "gamma": args.gamma,
            "learning_rate": 0.001,
            "replay_capacity": args.dqn_replay_capacity,
            "batch_size": args.dqn_batch_size,
            "target_update_frequency": args.target_update_frequency,
            "optimization_frequency": "every stream step after replay warm-up",
        } if args.policy == "dqn" else None),
        "reward_weights": {
            "label": args.label_cost_weight,
            "memory": args.memory_cost_weight,
            "compute": args.compute_cost_weight,
            "forgetting_weight": 0.0,
        },
        "stage_validation_trajectory": stage_rows,
        "validation_forgetting_proxy_macro_f1": _forgetting_proxy(stage_rows),
        "heldout_test": heldout_metrics,
        "same_seed_pairing": {
            "initial_task_model_sha256": task_initial_hash,
            "split_manifest_sha256": split_signature,
            "dataset_batch_sha256": data_hashes,
            "prediction_checkpoint_sha256": sha256_file(checkpoint),
            "v2_source_sha256": v2_hash,
            "comparison_runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing_seconds": {
            **timing,
            "end_to_end_run_seconds": end_to_end_seconds,
            "timing_note": "Per-component timers are nested within the stream-loop/end-to-end timers; do not add them to infer wall time.",
        },
        "limitations": [
            "Six known classes only; unknown-class discovery/open-set recognition not evaluated.",
            "Memory capacity is measured in records, not bytes or RAM.",
            "Compute tiers are provisional proxy costs; controller/evaluator compute is not debited from task budget.",
            "Reward has no explicit forgetting penalty; forgetting is reported separately using validation-only stage-boundary measurements.",
            "The current environment uses fixed REPLACE slot 0; the policy does not select a memory victim. DISCARD drops the incoming sample and leaves persistent memory unchanged.",
            "The stratified split construction uses class labels; held-out test labels are never used for training, reward, or policy selection.",
            "The 10% per-batch validation subset is repeatedly used for the within-run reward; this is a pilot evaluation protocol and may permit adaptation to validation feedback.",
            "V1 uses a local ring replay-buffer implementation with FIFO eviction and uniform sampling without replacement; this is a data-structure optimization, not a change to V2.",
            "A smoke run is only an execution-path check and is not evidence of policy effectiveness.",
        ],
        "artifacts": {
            "steps_csv": str(steps_path),
            "stage_validation_csv": str(stage_path) if stage_rows else None,
            "task_model": str(task_model_path),
            "dqn_controller": str(dqn_model_path) if dqn_model_path else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=40,
                        help="Positive value runs a smoke prefix; 0 processes the full stream.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Task-learner optimizer learning rate")
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-replay-capacity", type=int, default=5000)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    parser.add_argument("--label-cost-weight", type=float, default=0.01)
    parser.add_argument("--compute-cost-weight", type=float, default=0.0001)
    parser.add_argument("--memory-cost-weight", type=float, default=0.001)
    parser.add_argument("--data-dir", default="data/raw/gas_sensor_array_drift/Dataset")
    parser.add_argument(
        "--prediction-checkpoint",
        default="experiments/results/static_baseline_100_seed42/model_and_scaler.pt",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.output_dir is None:
        cli_args.output_dir = (
            f"experiments/results/policy_comp_v1/{cli_args.policy}/seed_{cli_args.seed}"
            + (f"_smoke_{cli_args.max_steps}" if cli_args.max_steps > 0 else "")
        )
    run_one(cli_args)
