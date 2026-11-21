"""E3 fixed-capacity reservoir replay baseline.

Run from the project root:
  python .\\experiments\\

Compares the E2 online-no-replay reference against a simple online learner with
uniform reservoir replay. It reuses E2's deterministic train/test splits,
feature scaler, model architecture, labelled-row selections, seed-specific
initialization, evaluation sets, and metric definitions.

Per seed:
  * 10 newly acquired labels arrive in each of the 10 chronological batches.
  * Total label budget = 100.
  * Replay memory capacity = 20 labelled examples, updated by standard
    reservoir sampling after each batch has been trained.
  * Batch 1 has no past data to replay: all 10 current examples are used in
    each of its 24 updates.
  * Batches 2-10 use 5 current examples + 5 sampled past-memory examples per
    update. Each of the 10 current examples is included exactly 12 times
    across the 24 updates for that batch; replay examples are sampled with
    replacement from the prior reservoir.
  * Total = 240 optimizer updates and 2,400 example presentations per seed.

This is a fixed random-replay baseline, not the Open-World AI memory-utility
policy. No production src/ files are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import open_world_e2_temporal_baseline as e2

EXPERIMENT_ID = "E3-RESERVOIR-01"
DEFAULT_SEEDS = list(range(42, 52))
MEMORY_CAPACITY = 20
CURRENT_PER_UPDATE = 5
REPLAY_PER_UPDATE = 5
UPDATES_PER_BATCH = e2.EPOCHS_PER_LEARNING_PHASE  # 24
EXPECTED_UPDATES = e2.EXPECTED_UPDATES_PER_STRATEGY  # 240
EXPECTED_PRESENTATIONS = e2.EXPECTED_EXAMPLE_PRESENTATIONS  # 2,400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def row_selection_by_batch(
    *, seed: int, train_indices_by_batch: list[np.ndarray]
) -> list[np.ndarray]:
    """Reproduce E2's online labelled-row selections exactly."""
    rng = np.random.default_rng(seed * 1_000_003 + 72)
    return [
        e2.select_without_replacement(
            pool, e2.LABELS_PER_BATCH, rng=rng, context=f"E3 online Batch {i}"
        )
        for i, pool in enumerate(train_indices_by_batch, start=1)
    ]



def verify_e2_selection_reference(
    e2_seed_dir: Path, selected_by_batch: list[np.ndarray]
) -> None:
    """Require exact ordered E2/E3 row-selection equality."""
    selected_file = e2_seed_dir / "selected_training_rows.csv"

    if not selected_file.is_file():
        raise FileNotFoundError(
            f"Missing E2 selection audit: {selected_file}"
        )

    expected_batches = set(range(1, 11))
    saved: dict[int, list[int]] = {
        i: [] for i in range(1, 11)
    }

    with selected_file.open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "strategy",
            "batch",
            "batch_row_number_1_based",
        }

        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "E2 selection CSV is missing required columns"
            )

        for line_number, row in enumerate(reader, start=2):
            if row.get("strategy") != "online_no_replay":
                continue

            try:
                batch_id = int(row["batch"])
                row_number = int(
                    row["batch_row_number_1_based"]
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid batch/row value at CSV line {line_number}"
                )

            if batch_id not in expected_batches:
                raise ValueError(
                    f"Unexpected batch ID {batch_id} "
                    f"at CSV line {line_number}"
                )

            if row_number < 1:
                raise ValueError(
                    f"Invalid 1-based row number at CSV line {line_number}"
                )

            saved[batch_id].append(row_number)

    if len(selected_by_batch) != 10:
        raise ValueError(
            f"E3 has {len(selected_by_batch)} batches; expected 10"
        )

    for batch_id, selected_rows in enumerate(
        selected_by_batch, start=1
    ):
        observed = saved[batch_id]
        expected = [
            int(x) + 1 for x in selected_rows
        ]

        if len(observed) != e2.LABELS_PER_BATCH:
            raise ValueError(
                f"E2 Batch {batch_id} has {len(observed)} "
                f"online selections; expected {e2.LABELS_PER_BATCH}"
            )

        if len(set(observed)) != len(observed):
            raise ValueError(
                f"E2 Batch {batch_id} contains duplicate selected rows"
            )

        if len(set(expected)) != len(expected):
            raise ValueError(
                f"E3 Batch {batch_id} contains duplicate selected rows"
            )

        if observed != expected:
            if set(observed) == set(expected):
                raise ValueError(
                    f"E2/E3 row order mismatch in Batch {batch_id}"
                )

            raise ValueError(
                f"E2/E3 selected row membership mismatch "
                f"in Batch {batch_id}"
            )



def make_current_schedule(
    rng: np.random.Generator, n_current: int = e2.LABELS_PER_BATCH
) -> list[np.ndarray]:
    """24 groups of five; every current row is used exactly 12 times."""
    if n_current != 10 or UPDATES_PER_BATCH != 24 or CURRENT_PER_UPDATE != 5:
        raise ValueError("Current-sample schedule is locked to 10 rows, 24 updates, 5 per update")
    schedule: list[np.ndarray] = []
    for _ in range(UPDATES_PER_BATCH // 2):
        perm = rng.permutation(n_current)
        schedule.append(perm[:CURRENT_PER_UPDATE])
        schedule.append(perm[CURRENT_PER_UPDATE:])
    if len(schedule) != UPDATES_PER_BATCH:
        raise ArithmeticError("Incorrect number of current-sample minibatches")
    counts = np.bincount(np.concatenate(schedule), minlength=n_current)
    if not np.all(counts == UPDATES_PER_BATCH // 2):
        raise ArithmeticError("Each current example must be presented exactly 12 times per batch")
    return schedule


def update_reservoir(
    memory: list[tuple[np.ndarray, int, int, int]],
    *,
    new_items: list[tuple[np.ndarray, int, int, int]],
    seen_before: int,
    rng: np.random.Generator,
) -> int:
    """Update a uniform reservoir with new labelled records.

    Record tuple: (scaled feature vector, label, source batch number, source
    row number 1-based). Returns the cumulative number of labelled examples
    observed by the reservoir.
    """
    seen = int(seen_before)
    for item in new_items:
        seen += 1
        if len(memory) < MEMORY_CAPACITY:
            memory.append((item[0].copy(), int(item[1]), int(item[2]), int(item[3])))
        else:
            replacement_index = int(rng.integers(0, seen))
            if replacement_index < MEMORY_CAPACITY:
                memory[replacement_index] = (
                    item[0].copy(), int(item[1]), int(item[2]), int(item[3])
                )
    if len(memory) > MEMORY_CAPACITY:
        raise ArithmeticError("Reservoir exceeded its fixed capacity")
    return seen


def train_replay_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    current_x: np.ndarray,
    current_y: np.ndarray,
    memory: list[tuple[np.ndarray, int, int, int]],
    *,
    stage: int,
    seed: int,
) -> tuple[list[float], int, int, int, int]:
    """Run 24 updates, with exact 2,400-exposure budget over the whole stream."""
    if len(current_y) != e2.LABELS_PER_BATCH:
        raise ValueError("Each stream batch must contribute exactly 10 labelled examples")
    loss_fn = nn.CrossEntropyLoss()
    current_tensor = torch.as_tensor(current_x, dtype=torch.float32, device="cpu")
    current_targets = torch.as_tensor(current_y - 1, dtype=torch.long, device="cpu")
    current_rng = np.random.default_rng(seed * 1_000_033 + stage * 10_007 + 301)
    replay_rng = np.random.default_rng(seed * 1_000_037 + stage * 10_009 + 401)
    epoch_losses: list[float] = []
    steps = current_presentations = replay_presentations = 0

    if stage == 1:
        if memory:
            raise ValueError("Batch 1 replay memory must be empty before first learning phase")
        # Warm-up: no previous labelled examples exist, so use all ten current rows.
        torch_generator = torch.Generator(device="cpu").manual_seed(seed * 101 + stage)
        for _ in range(UPDATES_PER_BATCH):
            order = torch.randperm(len(current_tensor), generator=torch_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(current_tensor[order]), current_targets[order])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            steps += 1
            current_presentations += len(current_tensor)
    else:
        if len(memory) == 0:
            raise ValueError(f"Replay memory unexpectedly empty before Batch {stage}")
        current_schedule = make_current_schedule(current_rng)
        replay_features = np.stack([item[0] for item in memory]).astype(np.float32)
        replay_labels = np.asarray([item[1] for item in memory], dtype=np.int64)
        for current_indices in current_schedule:
            replay_indices = replay_rng.integers(0, len(memory), size=REPLAY_PER_UPDATE)
            batch_x_np = np.concatenate(
                [current_x[current_indices], replay_features[replay_indices]], axis=0
            )
            batch_y_np = np.concatenate(
                [current_y[current_indices], replay_labels[replay_indices]], axis=0
            )
            batch_x = torch.as_tensor(batch_x_np, dtype=torch.float32, device="cpu")
            batch_y = torch.as_tensor(batch_y_np - 1, dtype=torch.long, device="cpu")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            steps += 1
            current_presentations += CURRENT_PER_UPDATE
            replay_presentations += REPLAY_PER_UPDATE

    if steps != UPDATES_PER_BATCH:
        raise ArithmeticError(f"Batch {stage} had {steps} updates, expected {UPDATES_PER_BATCH}")
    return epoch_losses, steps, current_presentations, replay_presentations, current_presentations + replay_presentations


def memory_snapshot(
    memory: list[tuple[np.ndarray, int, int, int]], *, after_batch: int, seen: int
) -> dict[str, Any]:
    batch_counts = {str(i): 0 for i in range(1, 11)}
    class_counts = {str(i): 0 for i in range(1, 7)}
    for _, label, source_batch, _ in memory:
        batch_counts[str(source_batch)] += 1
        class_counts[str(int(label))] += 1
    return {
        "after_batch": after_batch,
        "seen_labelled_examples": seen,
        "memory_size": len(memory),
        "source_batch_counts": batch_counts,
        "class_counts": class_counts,
    }


def run_one_seed(
    *,
    seed: int,
    batches: list[tuple[np.ndarray, np.ndarray]],
    data_paths: list[Path],
    output_dir: Path,
    e2_results_dir: Path,
    runner_path: Path,
    helper_path: Path,
    e2_runner_path: Path,
) -> dict[str, Any]:
    train_indices_by_batch: list[np.ndarray] = []
    test_indices_by_batch: list[np.ndarray] = []
    for batch_index, (_, labels) in enumerate(batches, start=1):
        split_seed = seed * 10_000 + batch_index * 97 + 13
        train_idx, test_idx = e2.stratified_split(labels, seed=split_seed)
        if len(train_idx) < e2.LABELS_PER_BATCH:
            raise ValueError(f"Batch {batch_index} has fewer than 10 training-pool rows")
        train_indices_by_batch.append(train_idx)
        test_indices_by_batch.append(test_idx)

    x_batch1, _ = batches[0]
    _, scaler_min, scaler_max = e2.base.fit_scaler(x_batch1[train_indices_by_batch[0]])
    constant_features = (scaler_max - scaler_min) == 0

    train_x_by_batch: list[np.ndarray] = []
    train_y_by_batch: list[np.ndarray] = []
    test_x_by_batch: list[np.ndarray] = []
    test_y_by_batch: list[np.ndarray] = []
    for batch_index, (x_raw, y_raw) in enumerate(batches):
        train_idx = train_indices_by_batch[batch_index]
        test_idx = test_indices_by_batch[batch_index]
        train_x_by_batch.append(e2.base.transform_with_scaler(x_raw[train_idx], scaler_min, scaler_max))
        train_y_by_batch.append(y_raw[train_idx].copy())
        test_x_by_batch.append(e2.base.transform_with_scaler(x_raw[test_idx], scaler_min, scaler_max))
        test_y_by_batch.append(y_raw[test_idx].copy())

    selected_by_batch = row_selection_by_batch(
        seed=seed, train_indices_by_batch=train_indices_by_batch
    )
    e2_seed_dir = e2_results_dir / f"seed_{seed}"
    verify_e2_selection_reference(e2_seed_dir, selected_by_batch)

    phase_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for batch_index, selected_rows in enumerate(selected_by_batch):
        lookup = {int(row): i for i, row in enumerate(train_indices_by_batch[batch_index])}
        offsets = np.asarray([lookup[int(row)] for row in selected_rows], dtype=np.int64)
        phase_data.append((
            train_x_by_batch[batch_index][offsets],
            train_y_by_batch[batch_index][offsets],
            selected_rows,
        ))

    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(seed)  # Same initialization as E2 online-no-replay baseline.
    torch.set_num_threads(1)
    model = e2.base.SmallMLP().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=e2.LEARNING_RATE)

    memory: list[tuple[np.ndarray, int, int, int]] = []
    reservoir_rng = np.random.default_rng(seed * 1_000_039 + 503)
    seen_labels = 0
    memory_history: list[dict[str, Any]] = []
    phase_losses: list[list[float]] = []
    current_presentations_by_batch: list[int] = []
    replay_presentations_by_batch: list[int] = []
    optimizer_steps = total_presentations = 0
    trajectory: list[dict[str, Any]] = []
    accuracy_history: list[list[float]] = [[] for _ in range(10)]
    selected_labels_by_batch: list[np.ndarray] = []

    for stage, (phase_x, phase_y, selected_rows) in enumerate(phase_data, start=1):
        losses, steps, current_pres, replay_pres, presentations = train_replay_batch(
            model, optimizer, phase_x, phase_y, memory, stage=stage, seed=seed
        )
        phase_losses.append(losses)
        optimizer_steps += steps
        total_presentations += presentations
        current_presentations_by_batch.append(current_pres)
        replay_presentations_by_batch.append(replay_pres)
        selected_labels_by_batch.append(phase_y.copy())

        stage_metrics: dict[str, Any] = {}
        for eval_batch in range(1, stage + 1):
            idx = eval_batch - 1
            metrics = e2.evaluate(model, test_x_by_batch[idx], test_y_by_batch[idx])
            stage_metrics[f"batch{eval_batch}"] = metrics
            accuracy_history[idx].append(float(metrics["accuracy"]))
        pooled = e2.pooled_metrics(model, test_x_by_batch, test_y_by_batch, stage)
        trajectory.append({
            "after_batch": stage,
            "current_batch_metrics": stage_metrics[f"batch{stage}"],
            "pooled_seen_batches_metrics": pooled,
            "per_seen_batch_metrics": stage_metrics,
        })

        # Add the just-trained batch's labelled records only after its updates;
        # future stages can replay them, but the current stage cannot replay itself.
        new_items = [
            (phase_x[i], int(phase_y[i]), stage, int(selected_rows[i]) + 1)
            for i in range(len(phase_y))
        ]
        seen_labels = update_reservoir(
            memory, new_items=new_items, seen_before=seen_labels, rng=reservoir_rng
        )
        memory_history.append(memory_snapshot(memory, after_batch=stage, seen=seen_labels))

    final_by_batch = {
        f"batch{i}": e2.evaluate(model, test_x_by_batch[i - 1], test_y_by_batch[i - 1])
        for i in range(1, 11)
    }
    final_pooled = e2.pooled_metrics(model, test_x_by_batch, test_y_by_batch, 10)
    per_task_forgetting = {
        f"batch{i}": float(max(accuracy_history[i - 1]) - accuracy_history[i - 1][-1])
        for i in range(1, 10)
    }
    mean_forgetting = float(np.mean(list(per_task_forgetting.values())))
    mean_current_accuracy = float(np.mean([
        float(row["current_batch_metrics"]["accuracy"]) for row in trajectory
    ]))
    mean_current_macro_f1 = float(np.mean([
        float(row["current_batch_metrics"]["macro_f1_all_six_classes"]) for row in trajectory
    ]))

    if optimizer_steps != EXPECTED_UPDATES:
        raise ArithmeticError(f"E3 updates={optimizer_steps}, expected {EXPECTED_UPDATES}")
    if total_presentations != EXPECTED_PRESENTATIONS:
        raise ArithmeticError(
            f"E3 presentations={total_presentations}, expected {EXPECTED_PRESENTATIONS}"
        )
    expected_current = 10 * 24 + 9 * 24 * 5
    expected_replay = 9 * 24 * 5
    if sum(current_presentations_by_batch) != expected_current:
        raise ArithmeticError("Current-example exposure count does not match protocol")
    if sum(replay_presentations_by_batch) != expected_replay:
        raise ArithmeticError("Replay-example exposure count does not match protocol")
    if len(memory) != MEMORY_CAPACITY or seen_labels != e2.LABEL_BUDGET:
        raise ArithmeticError("Final replay memory/label accounting mismatch")

    with (run_dir / "selected_training_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "batch", "batch_row_number_1_based", "class_id", "class_name"])
        for batch_id, (phase_y, selected_rows) in enumerate(zip(selected_labels_by_batch, selected_by_batch), start=1):
            for label, row_index in zip(phase_y, selected_rows):
                writer.writerow(["online_reservoir_replay", batch_id, int(row_index) + 1, int(label), e2.base.CLASS_NAMES[int(label)]])

    with (run_dir / "memory_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["after_batch", "seen_labelled_examples", "memory_size", "source_batch_counts", "class_counts"],
        )
        writer.writeheader()
        for snap in memory_history:
            writer.writerow({
                "after_batch": snap["after_batch"],
                "seen_labelled_examples": snap["seen_labelled_examples"],
                "memory_size": snap["memory_size"],
                "source_batch_counts": json.dumps(snap["source_batch_counts"], sort_keys=True),
                "class_counts": json.dumps(snap["class_counts"], sort_keys=True),
            })

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
        "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
        "strategy": "online_fixed_reservoir_replay",
        "seed": seed,
        "memory_capacity": MEMORY_CAPACITY,
        "memory_records": [
            {
                "features": torch.as_tensor(item[0], dtype=torch.float32),
                "label": int(item[1]),
                "source_batch": int(item[2]),
                "source_row_1_based": int(item[3]),
            }
            for item in memory
        ],
        "optimizer_updates": optimizer_steps,
        "training_example_presentations": total_presentations,
    }
    torch.save(checkpoint, run_dir / "reservoir_replay_model_and_memory.pt")

    e2_seed_record = next(
        row for row in json.loads((e2_results_dir / "summary.json").read_text(encoding="utf-8"))["seed_records"]
        if int(row["seed"]) == seed
    )
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "seed": int(seed),
        "protocol": {
            "dataset": "UCI Gas Sensor Array Drift Dataset",
            "label_budget": e2.LABEL_BUDGET,
            "labels_per_batch": e2.LABELS_PER_BATCH,
            "memory_type": "uniform reservoir sampling over previously acquired labelled examples",
            "memory_capacity_examples": MEMORY_CAPACITY,
            "replay_batch_composition_after_batch1": "5 current + 5 replay examples per update",
            "batch1_warmup_composition": "10 current examples; no earlier memory exists",
            "updates_per_batch": UPDATES_PER_BATCH,
            "optimizer_updates_total": optimizer_steps,
            "example_presentations_total": total_presentations,
            "current_example_presentations": sum(current_presentations_by_batch),
            "replay_example_presentations": sum(replay_presentations_by_batch),
            "optimizer": "Adam",
            "learning_rate": e2.LEARNING_RATE,
            "batch_size": e2.BATCH_SIZE,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "device": "CPU",
            "same_e2_label_selections": True,
            "same_e2_train_test_splits": True,
            "same_e2_model_initialization_seed": True,
            "scaler": "same E2 scaler protocol: Batch 1 training-pool features only; frozen thereafter; no clipping",
        },
        "data_audit": {
            "train_pool_sizes": [int(len(v)) for v in train_indices_by_batch],
            "test_sizes": [int(len(v)) for v in test_indices_by_batch],
            "selected_class_counts_by_batch": [e2.class_counts(y) for y in selected_labels_by_batch],
            "constant_scaler_features_zero_based": np.flatnonzero(constant_features).astype(int).tolist(),
        },
        "training_audit": {
            "optimizer_updates": optimizer_steps,
            "example_presentations": total_presentations,
            "current_presentations_by_batch": current_presentations_by_batch,
            "replay_presentations_by_batch": replay_presentations_by_batch,
            "phase_losses": phase_losses,
        },
        "final_metrics_by_batch": final_by_batch,
        "final_pooled_metrics": final_pooled,
        "forgetting": {
            "definition": "For each Batch 1-9, maximum post-update held-out accuracy from its acquisition stage through Batch 10 minus final held-out accuracy after Batch 10; Batch 10 excluded.",
            "per_batch": per_task_forgetting,
            "mean_batches_1_to_9": mean_forgetting,
            "accuracy_history_by_batch": accuracy_history,
        },
        "stage_trajectory": trajectory,
        "memory_history": memory_history,
        "e2_no_replay_reference": {
            "accuracy": float(e2_seed_record["online_accuracy"]),
            "macro_f1": float(e2_seed_record["online_macro_f1"]),
            "mean_forgetting_batches1to9": float(e2_seed_record["online_mean_forgetting_batches1to9"]),
            "seed_output_dir": e2_seed_record["output_dir"],
        },
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "runner_script_sha256": sha256_file(runner_path),
            "e2_runner_script_sha256": sha256_file(e2_runner_path),
            "e2_helper_script_sha256": e2.sha256_file(helper_path),
            "e2_summary_sha256": sha256_file(e2_results_dir / "summary.json"),
            "dataset_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": e2.sha256_file(path)}
                for path in data_paths
            ],
        },
    }
    write_json(run_dir / "metrics.json", result)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "samples", "accuracy", "macro_f1_all_six_classes", "macro_f1_present_classes_only"])
        for batch_name, metrics in final_by_batch.items():
            writer.writerow([batch_name, metrics["samples"], metrics["accuracy"], metrics["macro_f1_all_six_classes"], metrics["macro_f1_present_classes_only"]])
        writer.writerow(["pooled_future_batches_1_to_10", final_pooled["samples"], final_pooled["accuracy"], final_pooled["macro_f1_all_six_classes"], final_pooled["macro_f1_present_classes_only"]])

    return {
        "seed": int(seed),
        "reservoir_accuracy": float(final_pooled["accuracy"]),
        "reservoir_macro_f1": float(final_pooled["macro_f1_all_six_classes"]),
        "reservoir_mean_forgetting": mean_forgetting,
        "reservoir_current_batch_accuracy_mean": mean_current_accuracy,
        "reservoir_current_batch_macro_f1_mean": mean_current_macro_f1,
        "e2_no_replay_accuracy": float(e2_seed_record["online_accuracy"]),
        "e2_no_replay_macro_f1": float(e2_seed_record["online_macro_f1"]),
        "e2_no_replay_mean_forgetting": float(e2_seed_record["online_mean_forgetting_batches1to9"]),
        "paired_accuracy_difference_replay_minus_no_replay": float(final_pooled["accuracy"] - e2_seed_record["online_accuracy"]),
        "paired_macro_f1_difference_replay_minus_no_replay": float(final_pooled["macro_f1_all_six_classes"] - e2_seed_record["online_macro_f1"]),
        "paired_forgetting_difference_replay_minus_no_replay": float(mean_forgetting - e2_seed_record["online_mean_forgetting_batches1to9"]),
        "optimizer_updates": optimizer_steps,
        "example_presentations": total_presentations,
        "memory_capacity": MEMORY_CAPACITY,
        "output_dir": str(run_dir),
    }


def generate_report(
    *, output_dir: Path, seeds: list[int], summary: dict[str, Any], stage_summary: list[dict[str, Any]]
) -> str:
    out = summary["outcomes"]
    a = out["paired_accuracy_difference_replay_minus_no_replay"]
    f = out["paired_macro_f1_difference_replay_minus_no_replay"]
    g = out["paired_forgetting_difference_replay_minus_no_replay"]
    forgetting = out["reservoir_mean_forgetting"]
    lines = [
        f"# {EXPERIMENT_ID}: Fixed-Capacity Reservoir Replay Baseline",
        "",
        "## 1. Experiment identification",
        f"- Status: {summary['status']}; completed seeds: {len(seeds)}.",
        f"- Output directory: `{output_dir}`.",
        "",
        "## 2. Research objective",
        "Test whether a small fixed-size random reservoir of previously labelled examples reduces forgetting relative to E2 online fine-tuning without replay, under matched label selections, train/test splits, initialization seeds, optimizer updates, and total example presentations.",
        "",
        "## 3. Hypotheses",
        "- H1: Reservoir replay may reduce peak-to-final forgetting on earlier batches.",
        "- H2: Replay may improve final pooled Macro-F1 and/or accuracy versus E2 no-replay.",
        "- H3: Any retention benefit may come with a trade-off in current-batch adaptation because half of each post-warm-up minibatch is replay data.",
        "",
        "## 4. Experimental setup",
        f"- Dataset: UCI Gas Sensor Array Drift; seeds: `{seeds}`.",
        "- Same deterministic 80/20 class-wise split, Batch 1 training-pool scaler, model, initialization seeds, and selected 10 labelled rows per batch as the E2 online baseline.",
        f"- Label budget: 100 total (10 per batch); reservoir capacity: {MEMORY_CAPACITY} labelled examples.",
        "- Replay: standard uniform reservoir sampling; after Batch 1 warm-up, each update uses 5 current + 5 replay examples.",
        "- Batch 1 warm-up: 10 current examples per update because no past examples exist yet.",
        f"- Training: 24 updates per batch, {EXPECTED_UPDATES} total updates, {EXPECTED_PRESENTATIONS} total example presentations; Adam, learning rate 0.001, batch size 10, CPU.",
        "- No validation-based checkpoint selection; held-out test data are evaluation only.",
        "",
        "## 5. Methodology",
        "For each seed, reproduce E2's per-batch train/test partitions and exact online labelled-row selections. Initialize the same MLP with the same seed. Process batches chronologically. During Batch 1 warm-up, each of its 10 labels appears in all 24 updates because no past memory exists. After each batch has been trained, insert its labelled examples into a capacity-20 uniform reservoir. From Batch 2 onward, each optimizer update combines five current-batch examples with five examples sampled with replacement from the prior reservoir. The current-example schedule ensures every one of the ten current labels appears exactly 12 times in each later batch's 24 updates. Evaluate all seen held-out batches after each phase and calculate peak-to-final forgetting on Batches 1-9.",
        "",
        "## 6. Results",
        "### 6.1 Final pooled performance and paired replay-minus-no-replay differences",
        "| Outcome | Reservoir replay mean | SD | 95% CI | Paired difference vs E2 | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Accuracy | {out['reservoir_accuracy']['mean']:.4f} | {out['reservoir_accuracy']['sample_sd']:.4f} | [{out['reservoir_accuracy']['ci95_low']:.4f}, {out['reservoir_accuracy']['ci95_high']:.4f}] | {a['mean']:.4f} | [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] |",
        f"| Macro-F1 | {out['reservoir_macro_f1']['mean']:.4f} | {out['reservoir_macro_f1']['sample_sd']:.4f} | [{out['reservoir_macro_f1']['ci95_low']:.4f}, {out['reservoir_macro_f1']['ci95_high']:.4f}] | {f['mean']:.4f} | [{f['ci95_low']:.4f}, {f['ci95_high']:.4f}] |",
        f"| Forgetting (Batches 1-9) | {forgetting['mean']:.4f} | {forgetting['sample_sd']:.4f} | [{forgetting['ci95_low']:.4f}, {forgetting['ci95_high']:.4f}] | {g['mean']:.4f} | [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}] |",
        "",
        "### 6.2 Current-batch accuracy by stream position",
        "| Batch | Reservoir current-batch accuracy | Reservoir current-batch Macro-F1 |",
        "|---:|---:|---:|",
    ]
    for row in stage_summary:
        lines.append(f"| {row['batch']} | {row['accuracy_mean']:.4f} | {row['macro_f1_mean']:.4f} |")
    lines.extend([
        "",
        "## 7. Statistical analysis",
        "Means, sample standard deviations, and two-sided 95% Student-t confidence intervals are calculated across the ten paired seeds (df=9). Replay-minus-no-replay differences are paired by seed because E3 reuses E2's exact train/test partitions, selected labelled rows, and initialization seeds. Intervals describe variability under this seed protocol and are not evidence of generalization to other datasets or deployment environments.",
        "",
        "## 8. Observations and interpretation",
        "Interpret accuracy, Macro-F1, forgetting, and current-batch performance jointly. Lower forgetting alone is not sufficient if overall predictive performance collapses. This arm is a simple random-replay control, not a learned or utility-based memory policy.",
        "",
        "## 9. Limitations — what this experiment does not establish",
        "- The dataset batches are coarse chronological groups, not equal-duration consecutive time windows.",
        "- This is six-class temporal/domain drift, not class-incremental or open-set novel-class discovery.",
        "- The scaler uses unlabeled Batch 1 training-pool features, as in E2.",
        "- A capacity-20 uniform reservoir is only one memory size and one policy; no capacity sweep or memory utility scoring is tested.",
        "- The 5-current/5-replay ratio is fixed after Batch 1 and may not be optimal.",
        "- Equal update and example-presentation counts are a compute proxy, not exact FLOPs, energy, or wall-clock parity.",
        "- No active label acquisition, uncertainty estimation, compute-tier allocation, or complete Open-World AI policy is evaluated.",
        "",
        "## 10. Conclusion",
        f"Reservoir replay versus E2 no-replay: paired accuracy difference {a['mean']:.4f} (95% CI [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]); paired Macro-F1 difference {f['mean']:.4f} (95% CI [{f['ci95_low']:.4f}, {f['ci95_high']:.4f}]); paired forgetting difference {g['mean']:.4f} (95% CI [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}]). Interpret support only after comparing these outcomes and current-batch trajectory; this does not validate the Open-World AI policy.",
        "",
        "## 11. Reproducibility record",
        "Each seed folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `memory_trace.csv`, and `reservoir_replay_model_and_memory.pt`. The top-level directory contains `summary.json`, `summary.csv`, `stage_summary.csv`, and this report. Each run records source and dataset SHA-256 hashes, environment versions, selected rows, memory history, and training exposure counts.",
        "",
        "## 12. Next experiment",
        "If reservoir replay reduces forgetting, test memory capacity and replay ratio under the same label and training-exposure budgets. If results are mixed, inspect per-batch retention and class composition in the reservoir before testing utility-based selection. Only then evaluate the project's memory-utility pipeline and resource-allocation policy.",
        "",
        "Integrity note: this report is generated only after every requested seed completes and every run passes update, presentation, capacity, and E2 row-selection consistency checks.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run E3 fixed-capacity reservoir replay against E2 no-replay"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--e2-results-dir", type=Path,
        default=project_root / "experiments" / "results" / "e2_temporal_baseline_seeds42-51",
        help="Completed E2 results directory used as paired no-replay reference",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "experiments" / "results" / "e3_fixed_reservoir_replay_seeds42-51",
    )
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    if len(args.seeds) != 10:
        parser.error("E3 paired protocol is locked to exactly 10 distinct seeds")

    e2_summary_path = args.e2_results_dir / "summary.json"
    if not e2_summary_path.is_file():
        raise FileNotFoundError(f"Completed E2 summary not found: {e2_summary_path}")
    e2_summary = json.loads(e2_summary_path.read_text(encoding="utf-8"))
    if e2_summary.get("status") != "completed":
        raise ValueError("E2 reference summary is not marked completed")
    e2_seeds = {int(row["seed"]) for row in e2_summary.get("seed_records", [])}
    if set(args.seeds) - e2_seeds:
        raise ValueError(f"E2 reference is missing seed(s): {sorted(set(args.seeds) - e2_seeds)}")
    if int(e2_summary.get("protocol", {}).get("label_budget_per_strategy", -1)) != e2.LABEL_BUDGET:
        raise ValueError("E2 label budget differs from locked E3 protocol")
    if int(e2_summary.get("protocol", {}).get("optimizer_updates_per_strategy", -1)) != EXPECTED_UPDATES:
        raise ValueError("E2 optimizer-update budget differs from locked E3 protocol")
    if int(e2_summary.get("protocol", {}).get("example_presentations_per_strategy", -1)) != EXPECTED_PRESENTATIONS:
        raise ValueError("E2 example-presentation budget differs from locked E3 protocol")

    data_paths = [args.data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in data_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s):\n" + "\n".join(map(str, missing)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. Choose a fresh output directory; existing results will not be overwritten."
        )

    print("Loading and validating all ten dataset batches...")
    batches = [e2.base.load_batch(path) for path in data_paths]
    for i, (_, labels) in enumerate(batches, start=1):
        print(f"Batch {i}: {len(labels)} rows validated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    e2_runner_path = Path(e2.__file__).resolve()
    helper_path = Path(e2.base.__file__).resolve()
    records: list[dict[str, Any]] = []
    for position, seed in enumerate(args.seeds, start=1):
        print(f"\n[{position}/{len(args.seeds)}] seed={seed}: fixed reservoir replay")
        record = run_one_seed(
            seed=seed,
            batches=batches,
            data_paths=data_paths,
            output_dir=args.output_dir,
            e2_results_dir=args.e2_results_dir,
            runner_path=runner_path,
            helper_path=helper_path,
            e2_runner_path=e2_runner_path,
        )
        records.append(record)
        print(
            f"Replay accuracy={record['reservoir_accuracy']:.6f}; "
            f"Macro-F1={record['reservoir_macro_f1']:.6f}; "
            f"forgetting={record['reservoir_mean_forgetting']:.6f}"
        )
        print(f"Saved: {record['output_dir']}")

    def vals(key: str) -> list[float]:
        return [float(row[key]) for row in records]

    outcome_keys = [
        "reservoir_accuracy", "reservoir_macro_f1", "reservoir_mean_forgetting",
        "e2_no_replay_accuracy", "e2_no_replay_macro_f1", "e2_no_replay_mean_forgetting",
        "paired_accuracy_difference_replay_minus_no_replay",
        "paired_macro_f1_difference_replay_minus_no_replay",
        "paired_forgetting_difference_replay_minus_no_replay",
        "reservoir_current_batch_accuracy_mean", "reservoir_current_batch_macro_f1_mean",
    ]
    outcomes = {key: e2.mean_sd_ci(vals(key)) for key in outcome_keys}
    stage_summary: list[dict[str, Any]] = []
    for batch_position in range(10):
        seed_metrics = [
            json.loads((args.output_dir / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
            for seed in args.seeds
        ]
        stage_rows = [row["stage_trajectory"][batch_position] for row in seed_metrics]
        stage_summary.append({
            "batch": batch_position + 1,
            "accuracy_mean": float(np.mean([r["current_batch_metrics"]["accuracy"] for r in stage_rows])),
            "macro_f1_mean": float(np.mean([r["current_batch_metrics"]["macro_f1_all_six_classes"] for r in stage_rows])),
        })

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "seeds": args.seeds,
        "protocol": {
            "label_budget_per_strategy": e2.LABEL_BUDGET,
            "memory_capacity": MEMORY_CAPACITY,
            "optimizer_updates_per_strategy": EXPECTED_UPDATES,
            "example_presentations_per_strategy": EXPECTED_PRESENTATIONS,
            "current_presentations_total": 10 * 24 + 9 * 24 * 5,
            "replay_presentations_total": 9 * 24 * 5,
            "updates_per_batch": UPDATES_PER_BATCH,
            "batch_size": e2.BATCH_SIZE,
            "replay_ratio_after_batch1": "5 current : 5 replay",
        },
        "outcomes": outcomes,
        "seed_records": records,
        "stage_summary": stage_summary,
        "reference": {
            "e2_results_dir": str(args.e2_results_dir.resolve()),
            "e2_summary_sha256": sha256_file(e2_summary_path),
        },
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "runner_script_sha256": sha256_file(runner_path),
            "e2_runner_script_sha256": sha256_file(e2_runner_path),
            "helper_script_sha256": e2.sha256_file(helper_path),
            "dataset_files": [
                {"name": path.name, "size_bytes": path.stat().st_size, "sha256": e2.sha256_file(path)}
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
    report = generate_report(output_dir=args.output_dir, seeds=args.seeds, summary=summary, stage_summary=stage_summary)
    (args.output_dir / "research_report.md").write_text(report, encoding="utf-8")

    print("\nE3 FIXED RESERVOIR REPLAY COMPLETE")
    print(f"Completed seeds: {len(records)}/{len(args.seeds)}")
    print(f"Reservoir accuracy mean: {outcomes['reservoir_accuracy']['mean']:.6f}")
    print(f"Paired accuracy difference vs E2 no-replay: {outcomes['paired_accuracy_difference_replay_minus_no_replay']['mean']:.6f}")
    print(f"Reservoir mean forgetting: {outcomes['reservoir_mean_forgetting']['mean']:.6f}")
    print(f"Paired forgetting difference vs E2 no-replay: {outcomes['paired_forgetting_difference_replay_minus_no_replay']['mean']:.6f}")
    print(f"Summary JSON: {args.output_dir / 'summary.json'}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")
    print(f"Stage summary CSV: {args.output_dir / 'stage_summary.csv'}")
    print(f"Research report: {args.output_dir / 'research_report.md'}")
    print("Each seed folder contains metrics, selected rows, memory trace, and replay checkpoint.")


if __name__ == "__main__":
    main()
