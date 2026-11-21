"""E4 memory-capacity sweep for fixed-ratio uniform reservoir replay.

Run from project root:
  python .\\experiments\\open_world_e4_memory_capacity_sweep.py

The runner reuses the validated E3 replay implementation without modifying
E2/E3 source files. It varies only the reservoir capacity (10, 20, 40, 80
labelled records). The post-warm-up minibatch composition remains 5 current +
5 replay examples; each arm uses 100 labels, 240 optimizer updates, and 2,400
example presentations per seed. E2 no-replay is the zero-memory reference;
E3 capacity-20 results are an independent replication reference.

All capacities use the same seeds, E2 train/test partitions, selected labelled
rows, scaler, model initialization, optimizer, and chronological batch order.
This is an experimental baseline, not the Open-World AI utility-based policy.
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

import open_world_e2_temporal_baseline as e2
import open_world_e3_fixed_reservoir_replay as e3

EXPERIMENT_ID = "E4-MEMORY-CAPACITY-01"
DEFAULT_SEEDS = list(range(42, 52))
DEFAULT_CAPACITIES = [10, 20, 40, 80]
EXPECTED_UPDATES = e2.EXPECTED_UPDATES_PER_STRATEGY
EXPECTED_PRESENTATIONS = e2.EXPECTED_EXAMPLE_PRESENTATIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def summarize(values: list[float]) -> dict[str, Any]:
    return e2.mean_sd_ci(values)


def load_seed_record(summary: dict[str, Any], seed: int) -> dict[str, Any]:
    matches = [r for r in summary.get("seed_records", []) if int(r["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one reference record for seed {seed}; found {len(matches)}")
    return matches[0]


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(
    *, metrics: dict[str, Any], capacity: int, seed: int, e2_record: dict[str, Any]
) -> None:
    protocol = metrics["protocol"]
    training_audit = metrics["training_audit"]
    memory_history = metrics["memory_history"]
    if metrics.get("seed") != seed:
        raise ValueError(f"Seed mismatch in capacity-{capacity} run {seed}")
    if protocol.get("memory_capacity_examples") != capacity:
        raise ValueError(f"Memory capacity mismatch in capacity-{capacity} run {seed}")
    if protocol.get("label_budget") != 100:
        raise ValueError(f"Label budget mismatch in capacity-{capacity} run {seed}")
    if protocol.get("optimizer_updates_total") != EXPECTED_UPDATES:
        raise ValueError(f"Optimizer update mismatch in capacity-{capacity} run {seed}")
    if protocol.get("example_presentations_total") != EXPECTED_PRESENTATIONS:
        raise ValueError(f"Presentation mismatch in capacity-{capacity} run {seed}")
    if protocol.get("same_e2_label_selections") is not True:
        raise ValueError(f"E2 selected-row pairing not verified in capacity-{capacity} run {seed}")
    if protocol.get("same_e2_train_test_splits") is not True:
        raise ValueError(f"E2 split pairing not verified in capacity-{capacity} run {seed}")
    if protocol.get("same_e2_model_initialization_seed") is not True:
        raise ValueError(f"E2 initialization pairing not verified in capacity-{capacity} run {seed}")
    if training_audit.get("optimizer_updates") != EXPECTED_UPDATES:
        raise ValueError(f"Training audit update count mismatch in capacity-{capacity} run {seed}")
    if training_audit.get("example_presentations") != EXPECTED_PRESENTATIONS:
        raise ValueError(f"Training audit presentation count mismatch in capacity-{capacity} run {seed}")
    if sum(training_audit.get("current_presentations_by_batch", [])) != 1320:
        raise ValueError(f"Current-example exposure mismatch in capacity-{capacity} run {seed}")
    if sum(training_audit.get("replay_presentations_by_batch", [])) != 1080:
        raise ValueError(f"Replay-example exposure mismatch in capacity-{capacity} run {seed}")
    sizes = [int(row["memory_size"]) for row in memory_history]
    if len(sizes) != 10 or max(sizes) > capacity or sizes[-1] != capacity:
        raise ValueError(f"Memory trace invalid for capacity-{capacity} run {seed}: {sizes}")
    if metrics["e2_no_replay_reference"]["accuracy"] != float(e2_record["online_accuracy"]):
        raise ValueError(f"E2 reference accuracy mismatch for capacity-{capacity} run {seed}")


def verify_capacity20_replication(
    *, metrics: dict[str, Any], e3_metrics: dict[str, Any], seed: int
) -> dict[str, float]:
    """Require the repeated capacity-20 arm to reproduce the completed E3 run."""
    checks = {
        "accuracy": (
            float(metrics["final_pooled_metrics"]["accuracy"]),
            float(e3_metrics["final_pooled_metrics"]["accuracy"]),
        ),
        "macro_f1": (
            float(metrics["final_pooled_metrics"]["macro_f1_all_six_classes"]),
            float(e3_metrics["final_pooled_metrics"]["macro_f1_all_six_classes"]),
        ),
        "mean_forgetting": (
            float(metrics["forgetting"]["mean_batches_1_to_9"]),
            float(e3_metrics["forgetting"]["mean_batches_1_to_9"]),
        ),
    }
    differences: dict[str, float] = {}
    for name, (current, reference) in checks.items():
        delta = abs(current - reference)
        differences[name] = delta
        if delta > 1e-12:
            raise ValueError(
                f"Capacity-20 replication mismatch for seed {seed}, {name}: "
                f"E4={current:.15g}, E3={reference:.15g}, abs_diff={delta:.3g}"
            )
    if metrics["memory_history"] != e3_metrics["memory_history"]:
        raise ValueError(f"Capacity-20 memory trace does not reproduce E3 for seed {seed}")
    return differences


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run E4 fixed-ratio reservoir memory-capacity sweep against E2 and E3 references"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--capacities", type=int, nargs="+", default=DEFAULT_CAPACITIES,
        help="Reservoir capacities in labelled examples; must include 20 for E3 replication",
    )
    parser.add_argument(
        "--e2-results-dir", type=Path,
        default=project_root / "experiments" / "results" / "e2_temporal_baseline_seeds42-51",
    )
    parser.add_argument(
        "--e3-results-dir", type=Path,
        default=project_root / "experiments" / "results" / "e3_fixed_reservoir_replay_seeds42-51",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "experiments" / "results" / "e4_memory_capacity_sweep_seeds42-51",
    )
    args = parser.parse_args()

    if len(args.seeds) != 10 or len(set(args.seeds)) != 10 or any(s < 0 for s in args.seeds):
        parser.error("The paired E4 protocol requires exactly 10 distinct non-negative seeds")
    if not args.capacities or len(set(args.capacities)) != len(args.capacities):
        parser.error("--capacities must contain distinct positive integers")
    if 20 not in args.capacities:
        parser.error("--capacities must include 20 so the E3 replication check can be performed")
    if any(c < 1 or c > 80 for c in args.capacities):
        parser.error("capacities must be between 1 and 80; E2 is the zero-memory control")

    e2_summary_path = args.e2_results_dir / "summary.json"
    e3_summary_path = args.e3_results_dir / "summary.json"
    if not e2_summary_path.is_file() or not e3_summary_path.is_file():
        raise FileNotFoundError("Completed E2 and E3 summary.json files are both required")
    e2_summary = json.loads(e2_summary_path.read_text(encoding="utf-8"))
    e3_summary = json.loads(e3_summary_path.read_text(encoding="utf-8"))
    if e2_summary.get("status") != "completed" or e3_summary.get("status") != "completed":
        raise ValueError("E2 and E3 references must both be marked completed")
    if int(e3_summary.get("protocol", {}).get("memory_capacity", -1)) != 20:
        raise ValueError("E3 reference is not the capacity-20 reservoir experiment")
    if int(e2_summary.get("protocol", {}).get("label_budget_per_strategy", -1)) != 100:
        raise ValueError("E2 label budget differs from E4 protocol")
    if int(e2_summary.get("protocol", {}).get("optimizer_updates_per_strategy", -1)) != EXPECTED_UPDATES:
        raise ValueError("E2 optimizer-update budget differs from E4 protocol")
    if int(e2_summary.get("protocol", {}).get("example_presentations_per_strategy", -1)) != EXPECTED_PRESENTATIONS:
        raise ValueError("E2 presentation budget differs from E4 protocol")
    e2_seed_ids = {int(r["seed"]) for r in e2_summary.get("seed_records", [])}
    e3_seed_ids = {int(r["seed"]) for r in e3_summary.get("seed_records", [])}
    if set(args.seeds) - e2_seed_ids or set(args.seeds) - e3_seed_ids:
        raise ValueError("E2 or E3 reference is missing one or more requested seeds")

    data_paths = [args.data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [p for p in data_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s):\n" + "\n".join(map(str, missing)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. "
            "Choose a fresh output directory; existing results will not be overwritten."
        )

    print("Loading and validating all ten dataset batches...")
    batches = [e2.base.load_batch(path) for path in data_paths]
    for i, (_, labels) in enumerate(batches, start=1):
        print(f"Batch {i}: {len(labels)} rows validated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    e2_runner_path = Path(e2.__file__).resolve()
    helper_path = Path(e2.base.__file__).resolve()
    e3_engine_path = Path(e3.__file__).resolve()
    e3_engine_hash = sha256_file(e3_engine_path)
    e2_seed_records = {int(r["seed"]): r for r in e2_summary["seed_records"]}
    all_records: list[dict[str, Any]] = []
    capacity_summaries: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    replication_checks: list[dict[str, Any]] = []

    original_capacity = e3.MEMORY_CAPACITY
    try:
        total_runs = len(args.capacities) * len(args.seeds)
        run_position = 0
        for capacity in args.capacities:
            e3.MEMORY_CAPACITY = int(capacity)
            capacity_root = args.output_dir / f"capacity_{capacity}"
            capacity_root.mkdir(parents=True, exist_ok=False)
            arm_records: list[dict[str, Any]] = []
            print(f"\n=== RESERVOIR CAPACITY: {capacity} EXAMPLES ===")

            for seed in args.seeds:
                run_position += 1
                print(f"\n[{run_position}/{total_runs}] capacity={capacity}; seed={seed}")
                e2_record = e2_seed_records[int(seed)]
                record = e3.run_one_seed(
                    seed=int(seed),
                    batches=batches,
                    data_paths=data_paths,
                    output_dir=capacity_root,
                    e2_results_dir=args.e2_results_dir,
                    runner_path=runner_path,
                    helper_path=helper_path,
                    e2_runner_path=e2_runner_path,
                )
                run_dir = capacity_root / f"seed_{seed}"
                metrics = read_metrics(run_dir)
                validate_run(metrics=metrics, capacity=int(capacity), seed=int(seed), e2_record=e2_record)

                # Re-label the run metadata as an E4 capacity arm and retain the
                # imported E3 engine hash as explicit provenance.
                metrics["experiment_id"] = EXPERIMENT_ID
                metrics["capacity_sweep"] = {
                    "arm_capacity_examples": int(capacity),
                    "fixed_post_warmup_replay_composition": "5 current + 5 replay",
                    "e3_engine_script_sha256": e3_engine_hash,
                }
                metrics["provenance"]["e3_engine_script_sha256"] = e3_engine_hash
                write_json(run_dir / "metrics.json", metrics)

                if capacity == 20:
                    e3_reference_dir = args.e3_results_dir / f"seed_{seed}"
                    e3_reference_metrics = read_metrics(e3_reference_dir)
                    diffs = verify_capacity20_replication(
                        metrics=metrics, e3_metrics=e3_reference_metrics, seed=int(seed)
                    )
                    replication_checks.append({"capacity": 20, "seed": int(seed), **diffs, "status": "PASS"})

                record.update({
                    "capacity": int(capacity),
                    "output_dir": str(run_dir),
                    "reservoir_accuracy": float(metrics["final_pooled_metrics"]["accuracy"]),
                    "reservoir_macro_f1": float(metrics["final_pooled_metrics"]["macro_f1_all_six_classes"]),
                    "reservoir_mean_forgetting": float(metrics["forgetting"]["mean_batches_1_to_9"]),
                    "reservoir_current_batch_accuracy_mean": float(metrics["final_current_batch_mean"]["accuracy"])
                    if "final_current_batch_mean" in metrics else float(record["reservoir_current_batch_accuracy_mean"]),
                    "optimizer_updates": int(metrics["training_audit"]["optimizer_updates"]),
                    "example_presentations": int(metrics["training_audit"]["example_presentations"]),
                    "memory_capacity": int(capacity),
                })
                arm_records.append(record)
                all_records.append(record.copy())
                print(
                    f"Accuracy={record['reservoir_accuracy']:.6f}; "
                    f"Macro-F1={record['reservoir_macro_f1']:.6f}; "
                    f"forgetting={record['reservoir_mean_forgetting']:.6f}"
                )
                print(f"Saved: {run_dir}")

            def vals(key: str) -> list[float]:
                return [float(row[key]) for row in arm_records]

            capacity_summary = {
                "capacity": int(capacity),
                "n": len(arm_records),
                "accuracy": summarize(vals("reservoir_accuracy")),
                "macro_f1": summarize(vals("reservoir_macro_f1")),
                "mean_forgetting": summarize(vals("reservoir_mean_forgetting")),
                "current_batch_accuracy_mean": summarize(vals("reservoir_current_batch_accuracy_mean")),
                "paired_accuracy_difference_vs_e2": summarize(vals("paired_accuracy_difference_replay_minus_no_replay")),
                "paired_macro_f1_difference_vs_e2": summarize(vals("paired_macro_f1_difference_replay_minus_no_replay")),
                "paired_forgetting_difference_vs_e2": summarize(vals("paired_forgetting_difference_replay_minus_no_replay")),
            }
            capacity_summaries.append(capacity_summary)

            for batch_position in range(10):
                batch_rows: list[dict[str, Any]] = []
                for seed in args.seeds:
                    m = read_metrics(capacity_root / f"seed_{seed}")
                    batch_rows.append(m["stage_trajectory"][batch_position]["current_batch_metrics"])
                stage_summaries.append({
                    "capacity": int(capacity),
                    "batch": batch_position + 1,
                    "accuracy_mean": float(np.mean([r["accuracy"] for r in batch_rows])),
                    "macro_f1_mean": float(np.mean([r["macro_f1_all_six_classes"] for r in batch_rows])),
                })

    finally:
        e3.MEMORY_CAPACITY = original_capacity

    # Compare every capacity to the repeated 20-example arm, paired by seed.
    by_capacity_seed: dict[int, dict[int, dict[str, Any]]] = {}
    for row in all_records:
        by_capacity_seed.setdefault(int(row["capacity"]), {})[int(row["seed"])] = row
    if 20 not in by_capacity_seed:
        raise ValueError("Capacity-20 arm did not complete; cannot perform E3 replication reference")
    cap20 = by_capacity_seed[20]
    paired_vs20: list[dict[str, Any]] = []
    for capacity in args.capacities:
        if capacity == 20:
            continue
        accuracy_diffs = [
            by_capacity_seed[capacity][seed]["reservoir_accuracy"] - cap20[seed]["reservoir_accuracy"]
            for seed in args.seeds
        ]
        f1_diffs = [
            by_capacity_seed[capacity][seed]["reservoir_macro_f1"] - cap20[seed]["reservoir_macro_f1"]
            for seed in args.seeds
        ]
        forgetting_diffs = [
            by_capacity_seed[capacity][seed]["reservoir_mean_forgetting"] - cap20[seed]["reservoir_mean_forgetting"]
            for seed in args.seeds
        ]
        paired_vs20.append({
            "capacity": int(capacity),
            "reference_capacity": 20,
            "accuracy_difference_vs_capacity20": summarize(accuracy_diffs),
            "macro_f1_difference_vs_capacity20": summarize(f1_diffs),
            "forgetting_difference_vs_capacity20": summarize(forgetting_diffs),
        })

    replication_pass = len(replication_checks) == len(args.seeds) and all(
        row["status"] == "PASS" for row in replication_checks
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed" if len(all_records) == len(args.capacities) * len(args.seeds) and replication_pass else "audit_failed",
        "seeds": [int(s) for s in args.seeds],
        "capacities": [int(c) for c in args.capacities],
        "dataset": {
            "name": "UCI Gas Sensor Array Drift Dataset",
            "directory": str(args.data_dir.resolve()),
            "batch_rows": [int(len(y)) for _, y in batches],
            "total_rows": int(sum(len(y) for _, y in batches)),
        },
        "protocol": {
            "label_budget_per_strategy": 100,
            "labels_per_batch": 10,
            "memory_capacity_unit": "stored labelled examples (records), not bytes",
            "memory_capacities": [int(c) for c in args.capacities],
            "post_warmup_minibatch_composition": "5 current + 5 replay",
            "batch1_warmup_composition": "10 current; no earlier memory exists",
            "optimizer_updates_per_arm_seed": EXPECTED_UPDATES,
            "example_presentations_per_arm_seed": EXPECTED_PRESENTATIONS,
            "current_presentations_per_arm_seed": 1320,
            "replay_presentations_per_arm_seed": 1080,
            "updates_per_batch": 24,
            "batch_size": 10,
            "optimizer": "Adam",
            "learning_rate": e2.LEARNING_RATE,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "scaler": "Batch 1 training-pool features only; frozen thereafter; no clipping",
            "same_e2_label_selections": True,
            "same_e2_train_test_splits": True,
            "same_e2_model_initialization_seeds": True,
            "replay_sampling": "uniform sampling with replacement from prior reservoir memory",
            "reservoir_update": "standard uniform reservoir sampling; update after training each batch",
            "e2_zero_memory_control": "E2 online_no_replay reference; no extra E4 zero-capacity arm",
            "e3_capacity20_replication": "required exact check for accuracy, Macro-F1, forgetting, and memory trace per seed",
        },
        "outcomes_by_capacity": capacity_summaries,
        "paired_comparisons_vs_capacity20": paired_vs20,
        "capacity20_replication_checks": replication_checks,
        "seed_records": all_records,
        "stage_summary": stage_summaries,
        "references": {
            "e2_summary_sha256": sha256_file(e2_summary_path),
            "e3_summary_sha256": sha256_file(e3_summary_path),
        },
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(e3.torch.__version__),
            "platform": platform.platform(),
            "runner_script_sha256": sha256_file(runner_path),
            "e3_engine_script_sha256": e3_engine_hash,
            "e2_runner_script_sha256": sha256_file(e2_runner_path),
            "helper_script_sha256": sha256_file(helper_path),
            "dataset_files": [
                {"name": p.name, "size_bytes": p.stat().st_size, "sha256": e2.sha256_file(p)}
                for p in data_paths
            ],
        },
    }

    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "capacity", "seed", "reservoir_accuracy", "reservoir_macro_f1",
            "reservoir_mean_forgetting", "paired_accuracy_difference_replay_minus_no_replay",
            "paired_macro_f1_difference_replay_minus_no_replay",
            "paired_forgetting_difference_replay_minus_no_replay",
            "optimizer_updates", "example_presentations", "memory_capacity", "output_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_records:
            writer.writerow({key: row.get(key) for key in fields})
    with (args.output_dir / "capacity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "capacity", "n", "accuracy_mean", "accuracy_sd", "accuracy_ci95_low", "accuracy_ci95_high",
            "macro_f1_mean", "macro_f1_sd", "macro_f1_ci95_low", "macro_f1_ci95_high",
            "forgetting_mean", "forgetting_sd", "forgetting_ci95_low", "forgetting_ci95_high",
        ])
        writer.writeheader()
        for row in capacity_summaries:
            writer.writerow({
                "capacity": row["capacity"], "n": row["n"],
                "accuracy_mean": row["accuracy"]["mean"], "accuracy_sd": row["accuracy"]["sample_sd"],
                "accuracy_ci95_low": row["accuracy"]["ci95_low"], "accuracy_ci95_high": row["accuracy"]["ci95_high"],
                "macro_f1_mean": row["macro_f1"]["mean"], "macro_f1_sd": row["macro_f1"]["sample_sd"],
                "macro_f1_ci95_low": row["macro_f1"]["ci95_low"], "macro_f1_ci95_high": row["macro_f1"]["ci95_high"],
                "forgetting_mean": row["mean_forgetting"]["mean"], "forgetting_sd": row["mean_forgetting"]["sample_sd"],
                "forgetting_ci95_low": row["mean_forgetting"]["ci95_low"], "forgetting_ci95_high": row["mean_forgetting"]["ci95_high"],
            })
    with (args.output_dir / "paired_comparisons_vs_capacity20.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "capacity", "reference_capacity",
            "accuracy_difference_mean", "accuracy_difference_ci95_low", "accuracy_difference_ci95_high",
            "macro_f1_difference_mean", "macro_f1_difference_ci95_low", "macro_f1_difference_ci95_high",
            "forgetting_difference_mean", "forgetting_difference_ci95_low", "forgetting_difference_ci95_high",
        ])
        writer.writeheader()
        for row in paired_vs20:
            writer.writerow({
                "capacity": row["capacity"], "reference_capacity": 20,
                "accuracy_difference_mean": row["accuracy_difference_vs_capacity20"]["mean"],
                "accuracy_difference_ci95_low": row["accuracy_difference_vs_capacity20"]["ci95_low"],
                "accuracy_difference_ci95_high": row["accuracy_difference_vs_capacity20"]["ci95_high"],
                "macro_f1_difference_mean": row["macro_f1_difference_vs_capacity20"]["mean"],
                "macro_f1_difference_ci95_low": row["macro_f1_difference_vs_capacity20"]["ci95_low"],
                "macro_f1_difference_ci95_high": row["macro_f1_difference_vs_capacity20"]["ci95_high"],
                "forgetting_difference_mean": row["forgetting_difference_vs_capacity20"]["mean"],
                "forgetting_difference_ci95_low": row["forgetting_difference_vs_capacity20"]["ci95_low"],
                "forgetting_difference_ci95_high": row["forgetting_difference_vs_capacity20"]["ci95_high"],
            })
    with (args.output_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["capacity", "batch", "accuracy_mean", "macro_f1_mean"])
        writer.writeheader()
        writer.writerows(stage_summaries)

    # Twelve-section generated research report.
    lines = [
        f"# {EXPERIMENT_ID}: Fixed-Ratio Reservoir Memory-Capacity Sweep",
        "",
        "## 1. Experiment identification",
        f"- Status: {summary['status']}; completed runs: {len(all_records)}/{len(args.capacities) * len(args.seeds)}.",
        f"- Seeds: `{args.seeds}`; capacity arms: `{args.capacities}` stored labelled records.",
        f"- Output directory: `{args.output_dir}`.",
        "",
        "## 2. Research objective",
        "Estimate how fixed reservoir capacity affects final predictive performance and forgetting when label selections, data splits, initialization, replay ratio, optimizer updates, and total example presentations are held constant.",
        "",
        "## 3. Hypotheses",
        "- H1: Increasing reservoir capacity may improve retention by broadening the set of historical labelled examples available for replay.",
        "- H2: Capacity effects may show diminishing returns or non-monotonic behaviour; monotonic improvement is not assumed.",
        "- H3: Current-batch adaptation may vary as the composition of the replay reservoir changes.",
        "",
        "## 4. Experimental setup",
        "- Dataset: UCI Gas Sensor Array Drift; 10 coarse chronological batches; 128 features; six shared classes.",
        f"- Seeds: `{args.seeds}`; capacities: `{args.capacities}` examples per reservoir.",
        "- Each arm uses 100 labels (10 per batch), 240 optimizer updates, and 2,400 example presentations per seed.",
        "- After Batch 1 warm-up, every update has five current examples and five replay examples; replay is sampled with replacement from a standard uniform reservoir.",
        "- Same E2 train/test partitions, selected labels, Batch-1-only frozen scaler, model, optimizer, and initialization seeds are reused.",
        "- E2 online-no-replay is the zero-memory reference; capacity 20 is repeated as an E3 replication check.",
        "",
        "## 5. Methodology",
        "For each seed and capacity, process batches 1-10 in order. Train with 10 current examples per update during Batch 1 because no prior memory exists. After training each batch, insert its 10 labelled examples into a uniform reservoir of the assigned capacity. From Batch 2 onward, use five examples from the current labelled batch and five samples with replacement from the prior reservoir per update. The total update and presentation budgets therefore remain constant across capacities. Evaluate all seen held-out batches after each phase and calculate peak-to-final forgetting on Batches 1-9.",
        "",
        "## 6. Results",
        "| Capacity | Accuracy mean (95% CI) | Macro-F1 mean (95% CI) | Forgetting mean (95% CI) |",
        "|---:|---:|---:|---:|",
    ]
    for row in capacity_summaries:
        a, f, g = row["accuracy"], row["macro_f1"], row["mean_forgetting"]
        lines.append(
            f"| {row['capacity']} | {a['mean']:.4f} [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] "
            f"| {f['mean']:.4f} [{f['ci95_low']:.4f}, {f['ci95_high']:.4f}] "
            f"| {g['mean']:.4f} [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}] |"
        )
    lines.extend([
        "",
        "### 6.1 Paired comparison against capacity 20",
        "| Capacity | Accuracy difference (95% CI) | Macro-F1 difference (95% CI) | Forgetting difference (95% CI) |",
        "|---:|---:|---:|---:|",
    ])
    for row in paired_vs20:
        a, f, g = row["accuracy_difference_vs_capacity20"], row["macro_f1_difference_vs_capacity20"], row["forgetting_difference_vs_capacity20"]
        lines.append(
            f"| {row['capacity']} | {a['mean']:.4f} [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] "
            f"| {f['mean']:.4f} [{f['ci95_low']:.4f}, {f['ci95_high']:.4f}] "
            f"| {g['mean']:.4f} [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}] |"
        )
    lines.extend([
        "",
        "## 7. Statistical analysis",
        "For each capacity, report means, sample standard deviations, and two-sided 95% Student-t confidence intervals across ten seeds (df=9). Capacity comparisons are paired by seed against the capacity-20 arm. These are exploratory multiple comparisons; no multiplicity adjustment is applied. Confidence intervals are conditional on this dataset and protocol, not population-wide guarantees.",
        "",
        "## 8. Observations and interpretation",
        "Interpret final pooled accuracy, Macro-F1, forgetting, and current-batch trajectory jointly. A larger record capacity does not automatically imply better results: reservoir composition, class distribution, drift, and the fixed replay ratio may interact. Memory capacity is counted in stored records, not bytes or measured RAM.",
        "",
        "## 9. Limitations — what this experiment does not establish",
        "- One dataset with coarse, non-uniform chronological batches is used.",
        "- This is temporal/domain drift over six known classes, not class-incremental or open-set novel-class discovery.",
        "- Only capacities in the requested sweep and one 5:5 replay ratio are tested.",
        "- Uniform reservoir replay is not the project's future-utility memory selection policy.",
        "- Capacity counts labelled records, not serialized bytes, RAM overhead, or energy use.",
        "- Equal update and presentation counts are a compute proxy, not exact FLOPs or wall-clock parity.",
        "- Ten seeds provide limited precision; further datasets and independent replications are required.",
        "- Multiple paired capacity comparisons are exploratory and are not multiplicity-adjusted.",
        "",
        "## 10. Conclusion",
        "The capacity effect is supported only to the extent shown by the completed paired seed results. The capacity-20 arm must reproduce E3 within the strict numerical and memory-trace checks before the sweep is considered audited. No result from this experiment alone validates the complete Open-World AI agent or its utility-based memory policy.",
        "",
        "## 11. Reproducibility record",
        "Each capacity/seed folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, `memory_trace.csv`, and `reservoir_replay_model_and_memory.pt`. Top-level artifacts are `summary.json`, `summary.csv`, `capacity_summary.csv`, `paired_comparisons_vs_capacity20.csv`, `stage_summary.csv`, and this report. Script, helper, reference-summary, and dataset SHA-256 hashes are recorded.",
        "",
        "## 12. Next experiment",
        "After identifying a useful capacity range, hold capacity fixed and test replay-ratio allocation, then compare uniform reservoir replay with the project's memory-utility ranking under identical budgets. Any utility-based policy must be tested as a distinct treatment, not inferred from this random-replay sweep.",
        "",
        f"Integrity note: capacity-20 replication checks passed for {len(replication_checks)}/{len(args.seeds)} seeds. Each capacity/seed run is accepted only after label, split, initialization, update, presentation, and memory-capacity checks pass.",
    ])
    report = "\n".join(lines) + "\n"
    (args.output_dir / "research_report.md").write_text(report, encoding="utf-8")

    print("\nE4 MEMORY-CAPACITY SWEEP COMPLETE")
    print(f"Completed runs: {len(all_records)}/{len(args.capacities) * len(args.seeds)}")
    print(f"Capacity-20 E3 replication: {'PASS' if replication_pass else 'FAIL'}")
    for row in capacity_summaries:
        print(
            f"Capacity {row['capacity']}: accuracy={row['accuracy']['mean']:.6f}; "
            f"Macro-F1={row['macro_f1']['mean']:.6f}; "
            f"forgetting={row['mean_forgetting']['mean']:.6f}"
        )
    print(f"Summary JSON: {args.output_dir / 'summary.json'}")
    print(f"Capacity summary CSV: {args.output_dir / 'capacity_summary.csv'}")
    print(f"Paired comparisons CSV: {args.output_dir / 'paired_comparisons_vs_capacity20.csv'}")
    print(f"Research report: {args.output_dir / 'research_report.md'}")


if __name__ == "__main__":
    main()
