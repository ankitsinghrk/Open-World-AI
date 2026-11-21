"""Causal DQN pilot for resource-bounded continual learning.

Pilot ID: CDQN-PILOT-01-V2

This is an exploratory integration pilot, not a benchmark replacement.
It keeps the historical E2 scripts unchanged.

Causal order per stream sample:
  observe -> build state/mask -> select -> execute resource action
  -> train continual learner using acquired label and pre-action replay
  -> re-observe next sample -> evaluate the SAME current-batch validation set
  -> reward -> store DQN transition -> train DQN.

Important protocol notes:
- Per-batch 80/20 split follows the E2 chronological split.
- The 80% stream pool is split again into learning and reward-validation rows.
- Validation labels are evaluator-only; label provider exposes learning rows only.
- The 20% E2 test partition is never used for reward or model selection.
- Scaler is fitted on Batch 1 learning rows only (pilot-specific, no validation fit).
- Frozen PredictionProvider is used only for state prediction; the separate task
  learner is trained online by HybridBatchTrainer.
- Novelty is fixed at 0.0 in this pilot; novel-class discovery is NOT evaluated.
- Compute tier -> optimizer updates is a provisional pilot mapping, not a
  measured equivalence: {0: 0, 1: 1, 2: 2, 3: 4}.
- DISCARD in the current Environment implementation requires an existing memory
  index, although the generic feasibility function allows it when memory is empty.
  The pilot masks DISCARD while memory is empty to avoid invalid execution.
"""

from __future__ import annotations

import argparse
import csv
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

import open_world_static_100_baseline_v2 as base
import open_world_e2_temporal_baseline as e2
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
from src.dqn_training_loop import DQNTrainingLoop
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

PILOT_ID = "CDQN-PILOT-01-V2"
N_FEATURES = 128
N_CLASSES = 6
HISTORY_SIZE = 5
COMPUTE_TO_UPDATES = {0: 0, 1: 1, 2: 2, 3: 4}


class LearningOnlyLabelProvider:
    """Reveal labels only for rows in the learning partition."""

    def __init__(self, label_by_id: dict[str, int]):
        self._labels = dict(label_by_id)
        self.requests: list[str] = []

    def get_label(self, sample_id: str) -> int:
        self.requests.append(sample_id)
        if sample_id not in self._labels:
            raise KeyError(
                f"Label request outside learning partition: {sample_id}"
            )
        return int(self._labels[sample_id])


def macro_f1_fixed_six(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over all six known classes; absent-class F1 is zero."""
    scores = []
    for class_id in range(1, N_CLASSES + 1):
        tp = int(np.sum((y_true == class_id) & (y_pred == class_id)))
        fp = int(np.sum((y_true != class_id) & (y_pred == class_id)))
        fn = int(np.sum((y_true == class_id) & (y_pred != class_id)))
        denom = 2 * tp + fp + fn
        scores.append((2.0 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def build_pilot_state(
    environment: Environment,
    raw_observation: np.ndarray,
    scaled_state_features: np.ndarray,
    resources: dict[str, int],
    history: list[float],
) -> list[float]:
    """Build DQN state with normalized sensor features but raw prediction input.

    PredictionProvider applies its own checkpoint scaler, so it must receive raw
    sensor features. The DQN state receives the pilot's frozen Batch-1 scaler
    output to avoid feeding large raw sensor magnitudes into the Q-network.
    """
    prediction_result = environment.predict_observation(
        np.asarray(raw_observation, dtype=np.float64).tolist()
    )
    return environment.build_agent_state(
        observation=np.asarray(raw_observation, dtype=np.float64).tolist(),
        prediction=prediction_result["prediction"],
        uncertainty=prediction_result["uncertainty"],
        novelty=0.0,
        resources=resources,
        history=history,
        extracted_features=np.asarray(
            scaled_state_features, dtype=np.float64
        ).tolist(),
    )

def build_pilot_action_mask(
    environment: Environment, action_manager: ActionManager, memory: MemoryManager
) -> list[bool]:
    """Apply environment feasibility plus a label-preservation constraint.

    A label request combined with DISCARD and compute tier 0 is prohibited: it
    would spend a label while neither training on it nor retaining it for replay.
    """
    mask = list(environment.build_action_mask())
    memory_empty = memory.get_memory_status()["used"] == 0
    for index, action in enumerate(action_manager.generate_actions()):
        if memory_empty and action["memory"] == "DISCARD":
            mask[index] = False
        if (
            action["label"] == 1
            and action["memory"] == "DISCARD"
            and action["compute"] == 0
        ):
            mask[index] = False
    if not any(mask):
        raise RuntimeError("No feasible pilot actions remain after pilot constraints")
    return mask


def evaluate_model(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        raise ValueError("Validation partition is empty")
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64) + 1
    accuracy = float(np.mean(predictions == y))
    return {
        "accuracy": accuracy,
        "macro_f1_fixed_six": macro_f1_fixed_six(y, predictions),
    }


def make_partitions(
    batches: list[tuple[np.ndarray, np.ndarray]], seed: int
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Build independent stream/validation/test partitions for each batch."""
    prepared: list[dict[str, Any]] = []
    scaler_min = scaler_max = None

    for batch_number, (x_raw, y_raw) in enumerate(batches, start=1):
        split_seed = seed * 10_000 + batch_number * 97 + 13
        stream_idx, test_idx = e2.stratified_split(
            y_raw, seed=split_seed, test_fraction=0.20
        )
        stream_y = y_raw[stream_idx]
        learning_local, validation_local = e2.stratified_split(
            stream_y,
            seed=split_seed + 1_000_003,
            test_fraction=0.10,
        )
        learning_idx = stream_idx[learning_local]
        validation_idx = stream_idx[validation_local]

        if len(learning_idx) == 0 or len(validation_idx) == 0:
            raise ValueError(f"Empty learning/validation split in batch {batch_number}")

        if batch_number == 1:
            _, scaler_min, scaler_max = base.fit_scaler(x_raw[learning_idx])

        prepared.append({
            "batch": batch_number,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "learning_idx": learning_idx,
            "validation_idx": validation_idx,
            "test_idx": test_idx,
        })

    assert scaler_min is not None and scaler_max is not None

    for part in prepared:
        x_raw = part["x_raw"]
        part["x_scaled"] = base.transform_with_scaler(
            x_raw, scaler_min, scaler_max
        )
    return prepared, scaler_min, scaler_max


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_paths = [data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [str(path) for path in data_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset batches:\n" + "\n".join(missing))

    checkpoint = Path(args.prediction_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Prediction checkpoint not found: {checkpoint}")

    batches = [base.load_batch(path) for path in data_paths]
    prepared, scaler_min, scaler_max = make_partitions(batches, args.seed)

    # One chronological stream containing only learning-partition rows.
    stream: list[dict[str, Any]] = []
    label_map: dict[str, int] = {}
    batch_validation: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    stream_rng = np.random.default_rng(args.seed * 1_000_003 + 902)
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

        val_idx = part["validation_idx"]
        batch_validation[batch_number] = (
            part["x_scaled"][val_idx].astype(np.float32),
            part["y_raw"][val_idx].astype(np.int64),
        )

    if args.max_steps > 0:
        stream = stream[:args.max_steps]

    if not stream:
        raise ValueError("No stream samples available")

    label_provider = LearningOnlyLabelProvider(label_map)

    # The prediction checkpoint is frozen and separate from the trainable learner.
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
    state_encoder = StateEncoder(N_FEATURES, history_size=HISTORY_SIZE)

    # Task learner starts from a fresh random initialization.
    learner_model = base.SmallMLP()
    hybrid_trainer = HybridBatchTrainer(
        learner_model,
        learning_rate=args.learning_rate,
        device="cpu",
        input_dim=N_FEATURES,
        num_classes=N_CLASSES,
    )
    # Reward weights are explicit pilot parameters, not inferred from prior results.
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

    replay_buffer = ReplayBuffer(capacity=args.dqn_replay_capacity)
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
    loop = DQNTrainingLoop(
        dqn_agent, environment, replay_buffer, dqn_trainer
    )

    history: list[float] = []
    rows: list[dict[str, Any]] = []
    learner_update_count = 0
    acquired_label_count = 0
    timing_totals = {
        "state_prediction_seconds": 0.0,
        "validation_seconds": 0.0,
        "policy_selection_seconds": 0.0,
        "action_execution_seconds": 0.0,
        "task_training_seconds": 0.0,
        "dqn_training_seconds": 0.0,
        "stream_loop_seconds": 0.0,
    }

    for step_number, sample in enumerate(stream):
        step_started = time.perf_counter()
        batch_number = sample["batch"]
        raw_features = sample["raw"]
        scaled_features = sample["scaled"]
        sample_id = sample["sample_id"]

        # Build state from raw input for the frozen predictor, but use the
        # pilot-fitted scaled vector as the DQN's 128 sensor-state features.
        current_resources = resources.get_resources()
        state_started = time.perf_counter()
        current_state = build_pilot_state(
            environment, raw_features, scaled_features,
            current_resources, history[-HISTORY_SIZE:],
        )
        timing_totals["state_prediction_seconds"] += time.perf_counter() - state_started
        current_mask = build_pilot_action_mask(environment, action_manager, memory)

        validation_x, validation_y = batch_validation[batch_number]
        validation_started = time.perf_counter()
        utility_before = evaluate_model(
            hybrid_trainer.model, validation_x, validation_y
        )["macro_f1_fixed_six"]
        validation_seconds = time.perf_counter() - validation_started

        progress = step_number / max(1, len(stream) - 1)
        epsilon_now = (
            args.epsilon_start
            + (args.epsilon_end - args.epsilon_start) * progress
        )
        policy_started = time.perf_counter()
        selected = loop.select_action(
            current_state,
            current_mask,
            epsilon=epsilon_now,
        )
        policy_selection_seconds = time.perf_counter() - policy_started
        timing_totals["policy_selection_seconds"] += policy_selection_seconds
        # Optional integration-only override: action 18 = acquire label,
        # STORE the labelled item, and spend compute tier 1 (one task update).
        # This is NOT used in the full exploratory run.
        if args.force_first_learning_action and step_number == 0:
            forced_action = action_manager.get_action(18)
            forced_index = action_manager.get_network_index(18)
            if not current_mask[forced_index]:
                raise RuntimeError("Forced integration-smoke action is infeasible")
            selected = {"action_index": forced_index, "action": forced_action}
        action = selected["action"]

        # Snapshot replay BEFORE this action; newly stored records are not replayed
        # in the same transition's learner update.
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
            "item": item_for_memory if action["memory"] in ("STORE", "REPLACE") else None,
            "index": 0 if action["memory"] in ("DISCARD", "REPLACE") else None,
            "sample_id": sample_id if action["label"] == 1 else None,
            "label_provider": label_provider if action["label"] == 1 else None,
        }
        action_started = time.perf_counter()
        execution = loop.execute_action(selected, **action_kwargs)
        action_execution_seconds = time.perf_counter() - action_started
        timing_totals["action_execution_seconds"] += action_execution_seconds

        # Apply selected action's real learner effect.
        acquired_label = execution["acquired_label"]
        tier_updates = COMPUTE_TO_UPDATES[action["compute"]]
        current_features: list[list[float]] = []
        current_labels: list[int] = []
        if acquired_label is not None:
            acquired_label_count += 1
            current_features.append(scaled_features.tolist())
            current_labels.append(int(acquired_label))

        task_train_started = time.perf_counter()
        update_result = hybrid_trainer.train_batch(
            current_features=current_features,
            current_labels=current_labels,
            replay_records=labelled_replay,
            optimizer_updates=tier_updates,
            stage=batch_number,
            seed=args.seed + step_number,
        )
        task_training_seconds = time.perf_counter() - task_train_started
        timing_totals["task_training_seconds"] += task_training_seconds
        learner_update_count += update_result["completed_optimizer_updates"]

        validation_started = time.perf_counter()
        utility_after = evaluate_model(
            hybrid_trainer.model, validation_x, validation_y
        )["macro_f1_fixed_six"]
        validation_seconds += time.perf_counter() - validation_started
        timing_totals["validation_seconds"] += validation_seconds

        # Fresh next observation/state after execution and task-model update.
        done = step_number == len(stream) - 1
        if done:
            next_state = list(current_state)
            next_mask = [False] * 24
        else:
            next_sample = stream[step_number + 1]
            state_started = time.perf_counter()
            next_state = build_pilot_state(
                environment,
                next_sample["raw"],
                next_sample["scaled"],
                resources.get_resources(),
                (history + [utility_after - utility_before])[-HISTORY_SIZE:],
            )
            timing_totals["state_prediction_seconds"] += time.perf_counter() - state_started
            next_mask = build_pilot_action_mask(environment, action_manager, memory)

        reward_result = environment.calculate_v1_reward(
            utility_before=utility_before,
            utility_after=utility_after,
            action=action,
        )
        environment.store_dqn_experience(
            replay_buffer=replay_buffer,
            state=current_state,
            action_index=selected["action_index"],
            reward_result=reward_result,
            next_state=next_state,
            done=done,
            next_feasible_mask=next_mask,
        )
        dqn_started = time.perf_counter()
        dqn_loss = dqn_trainer.train()
        dqn_training_seconds = time.perf_counter() - dqn_started
        timing_totals["dqn_training_seconds"] += dqn_training_seconds

        utility_delta = utility_after - utility_before
        stream_loop_seconds = time.perf_counter() - step_started
        timing_totals["stream_loop_seconds"] += stream_loop_seconds
        history.append(float(utility_delta))
        rows.append({
            "step": step_number + 1,
            "epsilon": epsilon_now,
            "batch": batch_number,
            "sample_id": sample_id,
            "action_id": action["id"],
            "label_action": action["label"],
            "memory_operation": action["memory"],
            "compute_tier": action["compute"],
            "acquired_label": acquired_label,
            "utility_before_macro_f1": utility_before,
            "utility_after_macro_f1": utility_after,
            "utility_delta": utility_delta,
            "reward": reward_result["reward"],
            "task_optimizer_updates": update_result["completed_optimizer_updates"],
            "task_mean_loss": update_result["mean_loss"],
            "dqn_loss": dqn_loss,
            "memory_used": memory.get_memory_status()["used"],
            "labels_remaining": resources.get_resources()["labels"],
            "compute_remaining": resources.get_resources()["compute"],
            "replay_buffer_size": len(replay_buffer),
            "state_prediction_seconds": timing_totals["state_prediction_seconds"],
            "validation_seconds_step": validation_seconds,
            "action_execution_seconds": action_execution_seconds,
            "task_training_seconds": task_training_seconds,
            "dqn_training_seconds": dqn_training_seconds,
            "policy_selection_seconds": policy_selection_seconds,
            "stream_loop_seconds": stream_loop_seconds,
            "max_abs_scaled_state_feature": float(
                np.max(np.abs(np.asarray(current_state[:N_FEATURES], dtype=np.float64)))
            ),
        })

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "pilot_steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pilot_id": PILOT_ID,
        "status": "completed_smoke_run" if args.max_steps > 0 else "completed_single_seed_exploratory_run",
        "seed": args.seed,
        "steps": len(rows),
        "epsilon_schedule": {
            "start": args.epsilon_start,
            "end": args.epsilon_end,
            "shape": "linear over processed stream",
        },
        "force_first_learning_action": bool(args.force_first_learning_action),
        "labels_acquired": acquired_label_count,
        "task_optimizer_updates": learner_update_count,
        "dqn_training_steps": dqn_trainer.training_steps,
        "dqn_replay_size": len(replay_buffer),
        "memory_final_used": memory.get_memory_status()["used"],
        "label_budget_remaining": resources.get_resources()["labels"],
        "compute_budget_remaining": resources.get_resources()["compute"],
        "compute_budget_scope": (
            "task-learner compute action costs only; DQN controller optimization, "
            "state prediction, and validation inference are not debited from this "
            "budget and are separately timed below"
        ),
        "controller_and_evaluation_wall_time_seconds": {
            "state_prediction_total": timing_totals["state_prediction_seconds"],
            "validation_total": timing_totals["validation_seconds"],
            "dqn_training_total": timing_totals["dqn_training_seconds"],
            "policy_selection_total": timing_totals["policy_selection_seconds"],
        },
        "task_training_wall_time_seconds": timing_totals["task_training_seconds"],
        "action_execution_wall_time_seconds": timing_totals["action_execution_seconds"],
        "stream_loop_wall_time_seconds": timing_totals["stream_loop_seconds"],
        "prediction_checkpoint": str(checkpoint),
        "data_directory": str(data_dir),
        "validation_protocol": "per-batch 10% stratified subset of the 80% stream pool; evaluator-only",
        "test_protocol": "20% per-batch held-out partition; not used for reward",
        "scaler_protocol": "fit on Batch 1 learning partition only; frozen across batches",
        "novelty_protocol": "fixed at 0.0; novelty detection and novel-class discovery not evaluated",
        "compute_update_mapping": COMPUTE_TO_UPDATES,
        "compute_update_mapping_status": "provisional pilot assumption; not a measured compute equivalence",
        "reward_weights": {
            "label_cost_weight": args.label_cost_weight,
            "compute_cost_weight": args.compute_cost_weight,
            "memory_cost_weight": args.memory_cost_weight,
        },
        "label_requests_outside_learning_partition": 0,
        "notes": [
            "The frozen PredictionProvider checkpoint is used only for state prediction.",
            "The task learner starts from a fresh random SmallMLP initialization.",
            "The 20% held-out test partition is not used for reward or model selection.",
            "Results are exploratory and are not directly comparable to E2/E3/E4.",
            "The pilot does not establish optimality or open-world novel-class discovery.",
            "DQN state sensor features use the pilot-fitted frozen scaler; the frozen prediction checkpoint separately receives raw observations and applies its own scaler.",
            "Labels cannot be acquired with DISCARD and compute tier 0 in this pilot because that action would immediately waste the label.",
            "The compute budget constrains task-learner compute tiers only; controller and evaluator wall time are reported separately and not yet budget-constrained.",
        ],
    }
    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nStep log: {csv_path}")
    print(f"Summary:  {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CDQN-PILOT-01-V2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=25,
                        help="Positive value runs a smoke subset; 0 runs the full stream.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--force-first-learning-action",
        action="store_true",
        help="Integration smoke only: force one label+STORE+tier-1 action on step 1.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-replay-capacity", type=int, default=5000)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    parser.add_argument("--label-cost-weight", type=float, default=0.01)
    parser.add_argument("--compute-cost-weight", type=float, default=0.0001)
    parser.add_argument("--memory-cost-weight", type=float, default=0.001)
    parser.add_argument(
        "--data-dir", default="data/raw/gas_sensor_array_drift/Dataset"
    )
    parser.add_argument(
        "--prediction-checkpoint",
        default="experiments/results/static_baseline_100_seed42/model_and_scaler.pt",
    )
    parser.add_argument(
        "--output-dir", default="experiments/results/causal_dqn_pilot_01_v2"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pilot(parse_args())
