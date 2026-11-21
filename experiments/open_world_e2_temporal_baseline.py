"""E2 temporal continual-learning baseline comparison.

Run from the project root:
  python .\\experiments\\open_world_e2_temporal_baseline.py

Two controlled strategies are compared under equal label and training-exposure
budgets:
  1) static_batch1: acquire 100 labels from Batch 1, train once, never update;
  2) online_no_replay: acquire 10 labels per chronological batch and fine-tune
     only on the current batch's 10 labels, without replay.

For each seed, every batch is split into an 80% stream/training pool and a 20%
held-out test set using a deterministic class-wise split. A common feature scaler
is fitted using only the unlabeled features in Batch 1's 80% training pool and
then frozen for both strategies and all later batches. This is an explicitly
unsupervised initial-batch preprocessing assumption.

Both strategies use the same 128-64-6 MLP, Adam learning rate 0.001, batch size
10, 240 optimizer updates, and 2,400 training-example presentations:
  static: 24 epochs over 100 labelled examples (10 updates/epoch);
  online: 24 epochs over each incoming set of 10 labelled examples (1 update
          per epoch, 24 updates/batch, across 10 batches).

This is a benchmark baseline, not the Open-World AI agent. No production source
files are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import open_world_static_100_baseline_v2 as base

EXPERIMENT_ID = "E2-TEMP-01"
DEFAULT_SEEDS = list(range(42, 52))
LABEL_BUDGET = 100
LABELS_PER_BATCH = 10
TRAIN_FRACTION = 0.80
TEST_FRACTION = 0.20
EPOCHS_PER_LEARNING_PHASE = 24
BATCH_SIZE = 10
LEARNING_RATE = 0.001
EXPECTED_UPDATES_PER_STRATEGY = 240
EXPECTED_EXAMPLE_PRESENTATIONS = 2400
T_CRITICAL_DF9_95 = 2.262  # two-sided 95% Student-t critical value for n=10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def stratified_split(
    labels: np.ndarray, *, seed: int, test_fraction: float = TEST_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted train/test row indices; keep at least one train row/class."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_id in sorted(int(v) for v in np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_id)
        class_indices = class_indices.copy()
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            test_count = 0
        else:
            test_count = max(1, int(round(len(class_indices) * test_fraction)))
            test_count = min(test_count, len(class_indices) - 1)
        test_parts.append(class_indices[:test_count])
        train_parts.append(class_indices[test_count:])
    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    test_nonempty = [part for part in test_parts if len(part) > 0]
    if not test_nonempty:
        raise ValueError("Stratified split produced an empty test set")
    test_indices = np.sort(np.concatenate(test_nonempty)).astype(np.int64)
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Train and test partitions must both be non-empty")
    if np.intersect1d(train_indices, test_indices).size:
        raise ArithmeticError("Train/test split contains overlapping row indices")
    return train_indices, test_indices


def select_without_replacement(
    pool_indices: np.ndarray, count: int, *, rng: np.random.Generator, context: str
) -> np.ndarray:
    if len(pool_indices) < count:
        raise ValueError(
            f"Not enough training-pool rows for {context}: need {count}, got {len(pool_indices)}"
        )
    selected = rng.choice(pool_indices, size=count, replace=False)
    return np.asarray(selected, dtype=np.int64)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(class_id): int(np.sum(labels == class_id)) for class_id in range(1, 7)}


def train_static_model(
    x_train: np.ndarray, y_train: np.ndarray, *, seed: int
) -> tuple[nn.Module, list[float], int, int]:
    """Train once on all 100 labels: 24 epochs x 10 minibatches = 240 updates."""
    if len(y_train) != LABEL_BUDGET:
        raise ValueError(f"Static strategy requires exactly {LABEL_BUDGET} labels")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = base.SmallMLP().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device="cpu")
    train_y = torch.as_tensor(y_train - 1, dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_001)
    epoch_losses: list[float] = []
    steps = 0
    presentations = 0
    for _epoch in range(EPOCHS_PER_LEARNING_PHASE):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        weighted_loss = 0.0
        seen = 0
        for start in range(0, len(train_x), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            n = len(indices)
            weighted_loss += float(loss.detach().item()) * n
            seen += n
            steps += 1
            presentations += n
        epoch_losses.append(weighted_loss / seen)
    return model, epoch_losses, steps, presentations


def train_online_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_phase: np.ndarray,
    y_phase: np.ndarray,
    *,
    seed: int,
) -> tuple[list[float], int, int]:
    """Fine-tune on exactly the current phase's 10 labels; no past data/replay."""
    if len(y_phase) != LABELS_PER_BATCH:
        raise ValueError(f"Online phase requires exactly {LABELS_PER_BATCH} labels")
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.as_tensor(x_phase, dtype=torch.float32, device="cpu")
    train_y = torch.as_tensor(y_phase - 1, dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epoch_losses: list[float] = []
    steps = 0
    presentations = 0
    for _epoch in range(EPOCHS_PER_LEARNING_PHASE):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        # Ten examples and batch size ten: exactly one full-batch optimizer step.
        indices = permutation
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x[indices]), train_y[indices])
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.detach().item()))
        steps += 1
        presentations += len(indices)
    return epoch_losses, steps, presentations


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    return base.predict(model, x)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return base.calculate_metrics(y, predict(model, x))


def pooled_metrics(
    model: nn.Module,
    test_x: list[np.ndarray],
    test_y: list[np.ndarray],
    end_batch: int,
) -> dict[str, Any]:
    x = np.concatenate(test_x[:end_batch])
    y = np.concatenate(test_y[:end_batch])
    return evaluate(model, x, y)


def mean_sd_ci(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) == 10:
        half_width: float | None = T_CRITICAL_DF9_95 * sd / math.sqrt(10)
        low, high = mean - half_width, mean + half_width
    else:
        low = high = None
    return {
        "n": int(len(arr)),
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": low,
        "ci95_high": high,
    }


def run_one_seed(
    *,
    seed: int,
    batches: list[tuple[np.ndarray, np.ndarray]],
    data_paths: list[Path],
    output_dir: Path,
    runner_path: Path,
    helper_path: Path,
) -> dict[str, Any]:
    # Independent, reproducible stratified splits for each chronological batch.
    train_indices_by_batch: list[np.ndarray] = []
    test_indices_by_batch: list[np.ndarray] = []
    for batch_index, (_, labels) in enumerate(batches, start=1):
        split_seed = seed * 10_000 + batch_index * 97 + 13
        train_idx, test_idx = stratified_split(labels, seed=split_seed)
        if len(train_idx) < LABELS_PER_BATCH:
            raise ValueError(f"Batch {batch_index} has fewer than 10 training-pool rows")
        train_indices_by_batch.append(train_idx)
        test_indices_by_batch.append(test_idx)

    # A single common scaler is fitted only on Batch 1's 80% training pool.
    # No held-out rows or future-batch rows participate in fitting it.
    x_batch1, _ = batches[0]
    _, scaler_min, scaler_max = base.fit_scaler(x_batch1[train_indices_by_batch[0]])
    constant_features = (scaler_max - scaler_min) == 0

    train_x_by_batch: list[np.ndarray] = []
    train_y_by_batch: list[np.ndarray] = []
    test_x_by_batch: list[np.ndarray] = []
    test_y_by_batch: list[np.ndarray] = []
    for batch_index, (x_raw, y_raw) in enumerate(batches):
        train_idx = train_indices_by_batch[batch_index]
        test_idx = test_indices_by_batch[batch_index]
        train_x_by_batch.append(
            base.transform_with_scaler(x_raw[train_idx], scaler_min, scaler_max)
        )
        train_y_by_batch.append(y_raw[train_idx].copy())
        test_x_by_batch.append(
            base.transform_with_scaler(x_raw[test_idx], scaler_min, scaler_max)
        )
        test_y_by_batch.append(y_raw[test_idx].copy())

    # Strategy-specific sampling RNG streams keep the two label-allocation
    # procedures reproducible without coupling their draws to one another.
    static_label_rng = np.random.default_rng(seed * 1_000_003 + 71)
    online_label_rng = np.random.default_rng(seed * 1_000_003 + 72)
    static_pool = train_indices_by_batch[0]
    static_selected = select_without_replacement(
        static_pool, LABEL_BUDGET, rng=static_label_rng, context="static Batch 1 strategy"
    )
    online_selected_by_batch = [
        select_without_replacement(
            pool, LABELS_PER_BATCH, rng=online_label_rng, context=f"online Batch {i}"
        )
        for i, pool in enumerate(train_indices_by_batch, start=1)
    ]

    # Convert selected global row indices into offsets within each train pool.
    static_lookup = {int(row): i for i, row in enumerate(train_indices_by_batch[0])}
    static_offsets = np.asarray([static_lookup[int(row)] for row in static_selected])
    static_x = train_x_by_batch[0][static_offsets]
    static_y = train_y_by_batch[0][static_offsets]

    online_phase_data: list[tuple[np.ndarray, np.ndarray]] = []
    online_selected_labels: list[np.ndarray] = []
    for batch_index, selected_rows in enumerate(online_selected_by_batch):
        lookup = {int(row): i for i, row in enumerate(train_indices_by_batch[batch_index])}
        offsets = np.asarray([lookup[int(row)] for row in selected_rows])
        phase_x = train_x_by_batch[batch_index][offsets]
        phase_y = train_y_by_batch[batch_index][offsets]
        online_phase_data.append((phase_x, phase_y))
        online_selected_labels.append(phase_y)

    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Strategy A: 100 Batch 1 labels, trained once, then frozen.
    static_model, static_losses, static_steps, static_presentations = train_static_model(
        static_x, static_y, seed=seed
    )

    # Strategy B: 10 new labels per batch, current-batch-only updates, no replay.
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    online_model = base.SmallMLP().to("cpu")
    online_optimizer = torch.optim.Adam(online_model.parameters(), lr=LEARNING_RATE)
    online_phase_losses: list[list[float]] = []
    online_steps = 0
    online_presentations = 0
    online_trajectory: list[dict[str, Any]] = []
    online_accuracy_history: list[list[float]] = [[] for _ in range(10)]
    static_per_batch: list[dict[str, Any]] = []

    for stage in range(1, 11):
        phase_x, phase_y = online_phase_data[stage - 1]
        phase_seed = seed * 1_000_033 + stage * 10_007
        phase_losses, phase_steps, phase_presentations = train_online_phase(
            online_model,
            online_optimizer,
            phase_x,
            phase_y,
            seed=phase_seed,
        )
        online_phase_losses.append(phase_losses)
        online_steps += phase_steps
        online_presentations += phase_presentations

        stage_online_metrics: dict[str, Any] = {}
        stage_static_metrics: dict[str, Any] = {}
        for eval_batch in range(1, stage + 1):
            idx = eval_batch - 1
            static_metrics = evaluate(
                static_model, test_x_by_batch[idx], test_y_by_batch[idx]
            )
            online_metrics = evaluate(
                online_model, test_x_by_batch[idx], test_y_by_batch[idx]
            )
            stage_static_metrics[f"batch{eval_batch}"] = static_metrics
            stage_online_metrics[f"batch{eval_batch}"] = online_metrics
            online_accuracy_history[idx].append(float(online_metrics["accuracy"]))
            if stage == eval_batch:
                static_per_batch.append(static_metrics)

        static_pooled = pooled_metrics(
            static_model, test_x_by_batch, test_y_by_batch, stage
        )
        online_pooled = pooled_metrics(
            online_model, test_x_by_batch, test_y_by_batch, stage
        )
        online_trajectory.append(
            {
                "after_batch": stage,
                "static_current_batch": stage_static_metrics[f"batch{stage}"],
                "online_current_batch": stage_online_metrics[f"batch{stage}"],
                "static_pooled_seen_batches": static_pooled,
                "online_pooled_seen_batches": online_pooled,
                "online_per_seen_batch": stage_online_metrics,
            }
        )

    static_final_by_batch = {
        f"batch{i}": evaluate(static_model, test_x_by_batch[i - 1], test_y_by_batch[i - 1])
        for i in range(1, 11)
    }
    online_final_by_batch = {
        f"batch{i}": evaluate(online_model, test_x_by_batch[i - 1], test_y_by_batch[i - 1])
        for i in range(1, 11)
    }
    static_final_pooled = pooled_metrics(static_model, test_x_by_batch, test_y_by_batch, 10)
    online_final_pooled = pooled_metrics(online_model, test_x_by_batch, test_y_by_batch, 10)

    per_task_forgetting: dict[str, float] = {}
    for task_index in range(9):  # Exclude Batch 10: no later update can forget it.
        history = online_accuracy_history[task_index]
        if not history:
            continue
        per_task_forgetting[f"batch{task_index + 1}"] = float(max(history) - history[-1])
    mean_forgetting = float(np.mean(list(per_task_forgetting.values()))) if per_task_forgetting else 0.0

    # The online strategy's current-batch post-update metrics, one per stage.
    online_current_accuracy = [
        float(stage_row["online_current_batch"]["accuracy"])
        for stage_row in online_trajectory
    ]
    online_current_macro_f1 = [
        float(stage_row["online_current_batch"]["macro_f1_all_six_classes"])
        for stage_row in online_trajectory
    ]
    static_current_accuracy = [
        float(stage_row["static_current_batch"]["accuracy"])
        for stage_row in online_trajectory
    ]
    static_current_macro_f1 = [
        float(stage_row["static_current_batch"]["macro_f1_all_six_classes"])
        for stage_row in online_trajectory
    ]

    if static_steps != EXPECTED_UPDATES_PER_STRATEGY:
        raise ArithmeticError(f"Static strategy had {static_steps} updates, expected 240")
    if online_steps != EXPECTED_UPDATES_PER_STRATEGY:
        raise ArithmeticError(f"Online strategy had {online_steps} updates, expected 240")
    if static_presentations != EXPECTED_EXAMPLE_PRESENTATIONS:
        raise ArithmeticError("Static example-presentation budget mismatch")
    if online_presentations != EXPECTED_EXAMPLE_PRESENTATIONS:
        raise ArithmeticError("Online example-presentation budget mismatch")

    selected_csv = run_dir / "selected_training_rows.csv"
    with selected_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "batch", "batch_row_number_1_based", "class_id", "class_name"])
        batch1_labels = batches[0][1]
        for row in static_selected:
            row_i = int(row)
            label = int(batch1_labels[row_i])
            writer.writerow(["static_batch1", 1, row_i + 1, label, base.CLASS_NAMES[label]])
        for batch_index, selected_rows in enumerate(online_selected_by_batch, start=1):
            batch_labels = batches[batch_index - 1][1]
            for row in selected_rows:
                row_i = int(row)
                label = int(batch_labels[row_i])
                writer.writerow(["online_no_replay", batch_index, row_i + 1, label, base.CLASS_NAMES[label]])

    checkpoint_payload = {
        "model_state_dict": static_model.state_dict(),
        "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
        "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
        "strategy": "static_batch1",
        "seed": seed,
        "training_batch_rows_1_based": (static_selected + 1).astype(int).tolist(),
        "optimizer_updates": static_steps,
        "training_example_presentations": static_presentations,
    }
    torch.save(checkpoint_payload, run_dir / "static_model_and_scaler.pt")
    checkpoint_payload_online = {
        "model_state_dict": online_model.state_dict(),
        "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
        "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
        "strategy": "online_no_replay",
        "seed": seed,
        "training_batch_rows_1_based": [
            (rows + 1).astype(int).tolist() for rows in online_selected_by_batch
        ],
        "optimizer_updates": online_steps,
        "training_example_presentations": online_presentations,
    }
    torch.save(checkpoint_payload_online, run_dir / "online_model_and_scaler.pt")

    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "protocol": {
            "dataset": "UCI Gas Sensor Array Drift Dataset",
            "batches": 10,
            "train_test_split": "class-wise deterministic 80/20 split within each batch; test rows excluded from scaler fitting and training",
            "scaler": "feature-wise min-max to [-1, 1], fitted on unlabeled features in Batch 1 training pool only; frozen for all strategies and batches; no clipping",
            "static_strategy": "100 labels uniformly sampled from Batch 1 training pool; trained once; no later updates",
            "online_strategy": "10 labels uniformly sampled from each batch training pool; update using only current batch labels; no replay",
            "label_budget_per_strategy": LABEL_BUDGET,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs_static": EPOCHS_PER_LEARNING_PHASE,
            "epochs_per_online_batch": EPOCHS_PER_LEARNING_PHASE,
            "optimizer_updates_per_strategy": EXPECTED_UPDATES_PER_STRATEGY,
            "training_example_presentations_per_strategy": EXPECTED_EXAMPLE_PRESENTATIONS,
            "loss": "cross entropy",
            "model": "128 -> 64 -> ReLU -> 6 MLP (same SmallMLP helper)",
            "device": "CPU",
            "validation_or_checkpoint_selection": "none; final model after prescribed schedule",
        },
        "data_audit": {
            "raw_batch_sizes": [int(len(y)) for _, y in batches],
            "train_pool_sizes": [int(len(idx)) for idx in train_indices_by_batch],
            "test_sizes": [int(len(idx)) for idx in test_indices_by_batch],
            "test_class_counts": [class_counts(y) for y in test_y_by_batch],
            "static_selected_class_counts": class_counts(static_y),
            "online_selected_class_counts_by_batch": [class_counts(y) for y in online_selected_labels],
            "constant_scaler_features_zero_based": np.flatnonzero(constant_features).astype(int).tolist(),
        },
        "training_audit": {
            "static_optimizer_updates": int(static_steps),
            "online_optimizer_updates": int(online_steps),
            "static_training_example_presentations": int(static_presentations),
            "online_training_example_presentations": int(online_presentations),
            "static_epoch_mean_losses": static_losses,
            "online_epoch_losses_by_batch": online_phase_losses,
        },
        "static_final_metrics_by_batch": static_final_by_batch,
        "online_final_metrics_by_batch": online_final_by_batch,
        "static_final_pooled_metrics": static_final_pooled,
        "online_final_pooled_metrics": online_final_pooled,
        "online_forgetting": {
            "definition": "For each of Batches 1-9: maximum post-update held-out accuracy observed from its acquisition stage through Batch 10, minus final held-out accuracy after Batch 10; Batch 10 excluded because no subsequent update can affect it.",
            "per_batch": per_task_forgetting,
            "mean_batches_1_to_9": mean_forgetting,
            "accuracy_history_by_batch": online_accuracy_history,
        },
        "stage_trajectory": online_trajectory,
        "current_batch_post_update": {
            "static_accuracy_by_batch": static_current_accuracy,
            "online_accuracy_by_batch": online_current_accuracy,
            "static_macro_f1_by_batch": static_current_macro_f1,
            "online_macro_f1_by_batch": online_current_macro_f1,
        },
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "runner_script_sha256": sha256_file(runner_path),
            "helper_script_sha256": sha256_file(helper_path),
            "dataset_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in data_paths
            ],
        },
    }
    write_json(run_dir / "metrics.json", result)

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "evaluation_batch", "samples", "accuracy", "macro_f1_all_six_classes", "macro_f1_present_classes_only"])
        for batch_name in [f"batch{i}" for i in range(1, 11)]:
            for strategy, metric_map in [
                ("static_batch1", static_final_by_batch),
                ("online_no_replay", online_final_by_batch),
            ]:
                metrics = metric_map[batch_name]
                writer.writerow([strategy, batch_name, metrics["samples"], metrics["accuracy"], metrics["macro_f1_all_six_classes"], metrics["macro_f1_present_classes_only"]])

    return {
        "seed": int(seed),
        "static_accuracy": float(static_final_pooled["accuracy"]),
        "static_macro_f1": float(static_final_pooled["macro_f1_all_six_classes"]),
        "online_accuracy": float(online_final_pooled["accuracy"]),
        "online_macro_f1": float(online_final_pooled["macro_f1_all_six_classes"]),
        "paired_accuracy_difference_online_minus_static": float(online_final_pooled["accuracy"] - static_final_pooled["accuracy"]),
        "paired_macro_f1_difference_online_minus_static": float(online_final_pooled["macro_f1_all_six_classes"] - static_final_pooled["macro_f1_all_six_classes"]),
        "online_mean_forgetting_batches1to9": mean_forgetting,
        "static_mean_current_batch_accuracy": float(np.mean(static_current_accuracy)),
        "online_mean_current_batch_accuracy": float(np.mean(online_current_accuracy)),
        "static_mean_current_batch_macro_f1": float(np.mean(static_current_macro_f1)),
        "online_mean_current_batch_macro_f1": float(np.mean(online_current_macro_f1)),
        "output_dir": str(run_dir),
    }


def generate_report(
    *, output_dir: Path, seeds: list[int], records: list[dict[str, Any]],
    summary: dict[str, Any], stage_summary: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {EXPERIMENT_ID}: Temporal Continual-Learning Baseline",
        "",
        "## 1. Experiment identification",
        f"- Experiment ID: `{EXPERIMENT_ID}`",
        "- Status: completed; this report is produced only after all requested seeds succeed.",
        f"- Output directory: `{output_dir}`",
        "",
        "## 2. Research objective",
        "Compare a frozen static model with a no-replay online fine-tuning baseline on the chronological Gas Sensor Array Drift batch stream. Measure current-batch adaptation, final stream performance, and loss of previously acquired batch performance.",
        "",
        "## 3. Hypotheses",
        "- H1: Online no-replay updates may improve performance on newly arriving batches relative to a frozen model.",
        "- H2: Updating only on the current batch may reduce performance on earlier batches (catastrophic-forgetting signal).",
        "- H3: Equal label and training-exposure budgets may produce a trade-off between adaptation to recent batches and retention of earlier batches.",
        "",
        "## 4. Experimental setup",
        f"- Dataset: UCI Gas Sensor Array Drift; raw batch sizes: `{summary['dataset']['raw_batch_sizes']}`; total rows: {summary['dataset']['total_rows']:,}.",
        f"- Seeds: `{seeds}` ({len(seeds)} paired runs).",
        "- Per-batch split: deterministic class-wise 80/20 train-pool/held-out test split; same split used for both strategies within each seed.",
        "- Label budget: 100 per strategy. Static receives 100 from Batch 1; online receives 10 from each of Batches 1–10.",
        "- Scaler: feature-wise min-max to [-1, 1], fitted on unlabeled features in Batch 1's 80% train pool only, then frozen; held-out and future batches are not used to fit it.",
        "- Model: 128 → 64 → ReLU → 6 MLP; Adam; learning rate 0.001; cross-entropy; CPU; batch size 10.",
        "- Compute/exposure normalization: both strategies receive exactly 240 optimizer updates and 2,400 example presentations.",
        "- Static schedule: 24 epochs over its 100 labels (10 updates per epoch); then frozen.",
        "- Online schedule: 24 epochs over each current batch's 10 labels (24 updates per batch); no replay or access to prior labelled examples during later updates.",
        "- No validation-based selection or checkpoint selection; final scheduled models evaluated.",
        "",
        "## 5. Methodology",
        "For each seed, each of the ten raw batches is split independently by class. A common scaler is fitted on Batch 1's training-pool features only. The static strategy samples 100 labels from Batch 1 and trains once. The online strategy samples 10 labels from each batch in chronological order and fine-tunes only on that batch's selected labels. After each online update, both models are evaluated on held-out test sets of all batches seen so far. At the end, both are evaluated on the concatenated held-out sets from all ten batches. Online forgetting for each of Batches 1–9 is defined as the maximum post-update accuracy seen after that batch's acquisition minus its final accuracy after Batch 10; Batch 10 is excluded because no later update can affect it.",
        "",
        "## 6. Results",
        "### 6.1 Across-seed final pooled held-out performance",
        "| Strategy | Accuracy mean | SD | 95% CI | Macro-F1 mean | SD | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, key in [("Static Batch 1", "static"), ("Online no replay", "online")]:
        a = summary["outcomes"][f"{key}_accuracy"]
        f = summary["outcomes"][f"{key}_macro_f1"]
        a_ci = f"[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]" if a["ci95_low"] is not None else "not calculated"
        f_ci = f"[{f['ci95_low']:.4f}, {f['ci95_high']:.4f}]" if f["ci95_low"] is not None else "not calculated"
        lines.append(f"| {strategy} | {a['mean']:.4f} | {a['sample_sd']:.4f} | {a_ci} | {f['mean']:.4f} | {f['sample_sd']:.4f} | {f_ci} |")
    lines.extend([
        "",
        "### 6.2 Paired online-minus-static differences",
        "| Metric | Mean paired difference | SD | 95% CI |",
        "|---|---:|---:|---:|",
    ])
    for metric, key in [("Accuracy", "paired_accuracy_difference"), ("Macro-F1", "paired_macro_f1_difference")]:
        item = summary["paired_differences"][key]
        ci = f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]" if item["ci95_low"] is not None else "not calculated"
        lines.append(f"| {metric} | {item['mean']:.4f} | {item['sample_sd']:.4f} | {ci} |")
    lines.extend([
        "",
        "### 6.3 Mean current-batch post-update performance by stream position",
        "| Batch | Static accuracy | Online accuracy | Static Macro-F1 | Online Macro-F1 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in stage_summary:
        lines.append(f"| {row['batch']} | {row['static_accuracy_mean']:.4f} | {row['online_accuracy_mean']:.4f} | {row['static_macro_f1_mean']:.4f} | {row['online_macro_f1_mean']:.4f} |")
    forgetting = summary["outcomes"]["online_mean_forgetting_batches1to9"]
    ci = f"[{forgetting['ci95_low']:.4f}, {forgetting['ci95_high']:.4f}]" if forgetting["ci95_low"] is not None else "not calculated"
    accuracy_difference = summary["paired_differences"]["paired_accuracy_difference"]
    macro_difference = summary["paired_differences"]["paired_macro_f1_difference"]
    if accuracy_difference["mean"] > 0:
        accuracy_direction = "higher"
    elif accuracy_difference["mean"] < 0:
        accuracy_direction = "lower"
    else:
        accuracy_direction = "equal"
    accuracy_ci = (
        f"[{accuracy_difference['ci95_low']:.4f}, {accuracy_difference['ci95_high']:.4f}]"
        if accuracy_difference["ci95_low"] is not None else "not calculated"
    )
    macro_ci = (
        f"[{macro_difference['ci95_low']:.4f}, {macro_difference['ci95_high']:.4f}]"
        if macro_difference["ci95_low"] is not None else "not calculated"
    )
    conclusion_text = (
        f"Across the {len(seeds)} paired seeds, online no-replay final pooled accuracy was {accuracy_direction} "
        f"than static by a mean of {accuracy_difference['mean']:.4f} (95% CI {accuracy_ci}); "
        f"the paired Macro-F1 difference was {macro_difference['mean']:.4f} (95% CI {macro_ci}). "
        f"The online mean forgetting estimate for Batches 1–9 was {forgetting['mean']:.4f} "
        f"(95% CI {ci}). These are protocol-specific baseline findings, not evidence that the Open-World AI policy works."
    )
    lines.extend([
        "",
        f"- Online mean forgetting across Batches 1–9: {forgetting['mean']:.4f} (SD {forgetting['sample_sd']:.4f}; 95% CI {ci}).",
        f"- Mean current-batch accuracy across ten positions: static {summary['outcomes']['static_mean_current_batch_accuracy']['mean']:.4f}; online {summary['outcomes']['online_mean_current_batch_accuracy']['mean']:.4f}.",
        "",
        "## 7. Statistical analysis",
        "The ten seeds are paired: each static and online comparison uses the same seed-specific train/test splits and matching model-initialization seed; the two strategies use separate, reproducible label-selection RNG streams. Means, sample standard deviations, and two-sided 95% Student-t confidence intervals are computed across seeds (df=9 for the default ten-seed protocol). Paired intervals are reported for online-minus-static differences. These intervals quantify variation across the specified random-seed procedure; they do not establish causal effects across other datasets or environments. No unpaired significance test or claim of statistical significance is made.",
        "",
        "## 8. Observations and interpretation",
        "Interpret the current-batch trajectory together with final pooled performance and the forgetting metric. A higher current-batch score alone does not show better continual learning if prior-batch performance falls. The frozen static model is a no-adaptation reference; online no-replay is a simple fine-tuning reference, not a memory-aware or resource-aware policy.",
        "",
        "## 9. Limitations — what this experiment does not establish",
        "- The ten batch files are coarse chronological groups, not equal-duration or strictly consecutive time windows.",
        "- The task is six-class classification under temporal/domain drift; it is not class-incremental or open-set novel-class discovery.",
        "- The frozen scaler uses unlabeled features from the full Batch 1 training pool. This is an unsupervised initial-batch preprocessing assumption; it is not a purely label-only protocol.",
        "- The online baseline has no replay, memory selection, uncertainty estimation, active label acquisition, compute-tier allocation, or Open-World AI policy.",
        "- The online learner receives 10 labels at every batch, including batches where some classes may be absent or sparse; selected class counts are recorded per run.",
        "- The static baseline is intentionally front-loaded with all 100 labels from Batch 1, whereas online labels arrive progressively; this is the intended timing comparison, not identical information availability at every time point.",
        "- Equal optimizer updates and example presentations control a simple compute proxy, not exact wall-clock energy or hardware FLOPs.",
        "- Held-out samples are used only for evaluation; the results remain specific to this dataset, model, split rule, seed set, and training schedule.",
        "",
        "## 10. Conclusion",
        conclusion_text,
        "This experiment does not validate the full Open-World AI architecture or its research novelty.",
        "",
        "## 11. Reproducibility record",
        f"- Python: {summary['provenance']['python_version']}; NumPy: {summary['provenance']['numpy_version']}; PyTorch: {summary['provenance']['torch_version']}.",
        f"- Platform: `{summary['provenance']['platform']}`.",
        f"- Runner SHA-256: `{summary['provenance']['runner_script_sha256']}`.",
        f"- Static-baseline helper SHA-256: `{summary['provenance']['helper_script_sha256']}`.",
        f"- Data directory: `{summary['dataset']['directory']}`.",
        "- Each seed directory contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `static_model_and_scaler.pt`, and `online_model_and_scaler.pt`.",
        "- Combined files: `summary.json`, `summary.csv`, `stage_summary.csv`, and `research_report.md`.",
        "",
        "## 12. Next experiment",
        "If online no-replay exhibits forgetting, use this E2 result as the reference point for a controlled replay/memory baseline (for example, fixed-size reservoir replay) under the same label and compute budgets. Only after that should the project test utility-based memory selection or the learned resource-allocation policy. If online no-replay does not show clear adaptation or forgetting, first inspect per-batch metrics and verify the stream/split protocol before adding policy complexity.",
        "",
        "---",
        "Integrity note: all outcome values in this report are computed from per-seed saved run records. No metrics are fabricated or filled in for unsuccessful runs.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the E2 temporal static-vs-online continual-learning baseline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "experiments" / "results" / "e2_temporal_baseline_seeds42-51",
    )
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    if len(args.seeds) != 10:
        parser.error("E2 protocol is locked to exactly 10 distinct seeds for paired 95% confidence intervals")

    data_paths = [args.data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in data_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s):\n" + "\n".join(map(str, missing)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. Choose a new --output-dir; existing results will not be overwritten."
        )

    print("Loading and validating all ten dataset batches...")
    batches = [base.load_batch(path) for path in data_paths]
    raw_batch_sizes = [int(len(y)) for _, y in batches]
    for i, count in enumerate(raw_batch_sizes, start=1):
        print(f"Batch {i}: {count} rows validated")
    total_rows = int(sum(raw_batch_sizes))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    helper_path = Path(base.__file__).resolve()
    runner_path = Path(__file__).resolve()
    records: list[dict[str, Any]] = []
    for position, seed in enumerate(args.seeds, start=1):
        print(f"\n[{position}/{len(args.seeds)}] seed={seed}: running static and online strategies")
        record = run_one_seed(
            seed=seed,
            batches=batches,
            data_paths=data_paths,
            output_dir=args.output_dir,
            runner_path=runner_path,
            helper_path=helper_path,
        )
        records.append(record)
        print(
            f"Static accuracy={record['static_accuracy']:.6f}; "
            f"online accuracy={record['online_accuracy']:.6f}; "
            f"online forgetting={record['online_mean_forgetting_batches1to9']:.6f}"
        )
        print(f"Saved: {record['output_dir']}")

    def get_values(key: str) -> list[float]:
        return [float(row[key]) for row in records]

    outcome_keys = [
        "static_accuracy", "static_macro_f1", "online_accuracy", "online_macro_f1",
        "paired_accuracy_difference_online_minus_static",
        "paired_macro_f1_difference_online_minus_static",
        "online_mean_forgetting_batches1to9",
        "static_mean_current_batch_accuracy", "online_mean_current_batch_accuracy",
        "static_mean_current_batch_macro_f1", "online_mean_current_batch_macro_f1",
    ]
    outcomes = {key: mean_sd_ci(get_values(key)) for key in outcome_keys}

    # Aggregate current-batch metrics by stream position across seeds.
    stage_summary: list[dict[str, Any]] = []
    for batch_position in range(10):
        stage_summary.append({
            "batch": batch_position + 1,
            "static_accuracy_mean": float(np.mean([
                # Reload per-seed metrics to aggregate the batch-specific current score.
                json.loads((Path(row["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))["current_batch_post_update"]["static_accuracy_by_batch"][batch_position]
                for row in records
            ])),
            "online_accuracy_mean": float(np.mean([
                json.loads((Path(row["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))["current_batch_post_update"]["online_accuracy_by_batch"][batch_position]
                for row in records
            ])),
            "static_macro_f1_mean": float(np.mean([
                json.loads((Path(row["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))["current_batch_post_update"]["static_macro_f1_by_batch"][batch_position]
                for row in records
            ])),
            "online_macro_f1_mean": float(np.mean([
                json.loads((Path(row["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))["current_batch_post_update"]["online_macro_f1_by_batch"][batch_position]
                for row in records
            ])),
        })

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "seeds": args.seeds,
        "dataset": {
            "name": "UCI Gas Sensor Array Drift Dataset",
            "directory": str(args.data_dir.resolve()),
            "raw_batch_sizes": raw_batch_sizes,
            "total_rows": total_rows,
        },
        "protocol": {
            "label_budget_per_strategy": LABEL_BUDGET,
            "optimizer_updates_per_strategy": EXPECTED_UPDATES_PER_STRATEGY,
            "example_presentations_per_strategy": EXPECTED_EXAMPLE_PRESENTATIONS,
            "static_epochs": EPOCHS_PER_LEARNING_PHASE,
            "online_epochs_per_batch": EPOCHS_PER_LEARNING_PHASE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "outcomes": outcomes,
        "paired_differences": {
            "paired_accuracy_difference": outcomes["paired_accuracy_difference_online_minus_static"],
            "paired_macro_f1_difference": outcomes["paired_macro_f1_difference_online_minus_static"],
        },
        "seed_records": records,
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "runner_script_sha256": sha256_file(runner_path),
            "helper_script_sha256": sha256_file(helper_path),
            "dataset_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in data_paths
            ],
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_summary[0].keys()))
        writer.writeheader()
        writer.writerows(stage_summary)

    report = generate_report(
        output_dir=args.output_dir,
        seeds=args.seeds,
        records=records,
        summary=summary,
        stage_summary=stage_summary,
    )
    (args.output_dir / "research_report.md").write_text(report, encoding="utf-8")

    print("\nE2 TEMPORAL BASELINE COMPLETE")
    print(f"Completed seeds: {len(records)}/{len(args.seeds)}")
    print(f"Static pooled accuracy mean: {outcomes['static_accuracy']['mean']:.6f}")
    print(f"Online pooled accuracy mean: {outcomes['online_accuracy']['mean']:.6f}")
    print(f"Paired accuracy difference: {outcomes['paired_accuracy_difference_online_minus_static']['mean']:.6f}")
    print(f"Online mean forgetting (Batches 1-9): {outcomes['online_mean_forgetting_batches1to9']['mean']:.6f}")
    print(f"Summary JSON: {args.output_dir / 'summary.json'}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")
    print(f"Stage summary CSV: {args.output_dir / 'stage_summary.csv'}")
    print(f"Research report: {args.output_dir / 'research_report.md'}")
    print("Each seed folder contains metrics.json, metrics.csv, selected_training_rows.csv, and both model checkpoints.")


if __name__ == "__main__":
    main()
