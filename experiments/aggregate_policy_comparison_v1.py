"""Aggregate the predeclared 10-seed DQN/Random/Heuristic comparison."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

EXPERIMENT_ID = "CDQN-POLICY-COMP-01-V1"
POLICIES = ("dqn", "random_feasible", "fixed_heuristic")
SEEDS = tuple(range(42, 52))
T_CRITICAL_DF9_95 = 2.2621571627409915
PAIR_KEYS = (
    "initial_task_model_sha256",
    "split_manifest_sha256",
    "dataset_batch_sha256",
    "prediction_checkpoint_sha256",
    "v2_source_sha256",
    "comparison_runner_sha256",
)
METRICS = {
    "heldout_macro_f1": lambda d: d["heldout_test"]["pooled"]["macro_f1_fixed_six"],
    "heldout_accuracy": lambda d: d["heldout_test"]["pooled"]["accuracy"],
    "labels_acquired": lambda d: d["labels_acquired"],
    "compute_budget_consumed": lambda d: d["compute_budget_consumed"],
    "task_optimizer_updates": lambda d: d["task_optimizer_updates"],
    "memory_final_used": lambda d: d["memory_final_used"],
    "dqn_training_steps": lambda d: d["dqn_training_steps"],
    "end_to_end_run_seconds": lambda d: d["timing_seconds"]["end_to_end_run_seconds"],
}


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    if len(values) != 10:
        raise ValueError("This preregistered analysis expects exactly 10 paired seeds (42-51)")
    m = mean(values)
    sd = stdev(values)
    half = T_CRITICAL_DF9_95 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": m,
        "sample_sd": sd,
        "ci95_low": m - half,
        "ci95_high": m + half,
    }


def load_runs(results_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = {p: {} for p in POLICIES}
    for policy in POLICIES:
        for seed in SEEDS:
            path = results_root / policy / f"seed_{seed}" / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing full-run summary: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("experiment_id") != EXPERIMENT_ID:
                raise ValueError(f"Wrong experiment ID in {path}")
            if data.get("policy") != policy or data.get("seed") != seed:
                raise ValueError(f"Policy/seed identity mismatch in {path}")
            if data.get("status") != "completed_full_stream_single_seed":
                raise ValueError(f"Run is not a completed full-stream run: {path}")
            if not data.get("heldout_test"):
                raise ValueError(f"Held-out test metrics missing from {path}")
            runs[policy][seed] = data

    # Paired checks are within seed. Initial model hashes are expected to differ
    # between seeds, but must match across policy arms for the same seed.
    for seed in SEEDS:
        reference = runs["dqn"][seed]["same_seed_pairing"]
        for policy in POLICIES[1:]:
            candidate = runs[policy][seed]["same_seed_pairing"]
            for key in PAIR_KEYS:
                if candidate.get(key) != reference.get(key):
                    raise ValueError(
                        f"Pairing mismatch for seed {seed}, policy {policy}, key {key}"
                    )
    return runs


def aggregate(results_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    runs = load_runs(results_root)
    by_policy: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        by_policy[policy] = {}
        for metric, getter in METRICS.items():
            values = [float(getter(runs[policy][seed])) for seed in SEEDS]
            by_policy[policy][metric] = mean_sd_ci(values)

    paired = {}
    for comparator in ("random_feasible", "fixed_heuristic"):
        paired[f"dqn_minus_{comparator}"] = {}
        for metric, getter in METRICS.items():
            differences = [
                float(getter(runs["dqn"][seed])) - float(getter(runs[comparator][seed]))
                for seed in SEEDS
            ]
            paired[f"dqn_minus_{comparator}"][metric] = mean_sd_ci(differences)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "aggregated_30_full_runs",
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "by_policy": by_policy,
        "paired_differences": paired,
        "statistical_note": (
            "Nominal two-sided 95% paired Student-t confidence intervals, df=9. "
            "No p-values or multiplicity adjustment are calculated. Treat as exploratory."
        ),
        "interpretation_limits": [
            "One dataset and one stream protocol; no broad generalization claim.",
            "Six known classes only; no unknown-class discovery evaluation.",
            "Compute is a task-learner budget proxy; controller/evaluator compute is separately timed.",
            "Memory is measured in records, not bytes/RAM.",
            "DQN, random and heuristic arms are compared under equal initial budgets, not forced to spend equal amounts.",
        ],
    }

    destination = output_dir or (results_root / "aggregate")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    table_path = destination / "aggregate_policy_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "metric", "n", "mean", "sample_sd", "ci95_low", "ci95_high"])
        for policy in POLICIES:
            for metric, stats in by_policy[policy].items():
                writer.writerow([policy, metric, stats["n"], stats["mean"], stats["sample_sd"], stats["ci95_low"], stats["ci95_high"]])
        for contrast, metric_stats in paired.items():
            for metric, stats in metric_stats.items():
                writer.writerow([contrast, metric, stats["n"], stats["mean"], stats["sample_sd"], stats["ci95_low"], stats["ci95_high"]])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID + " aggregator")
    parser.add_argument("--results-root", default="experiments/results/policy_comp_v1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.results_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    out = Path(args.output_dir) if args.output_dir else None
    if out is not None and not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    print(json.dumps(aggregate(root, out), indent=2))
