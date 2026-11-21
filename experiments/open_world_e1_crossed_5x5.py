"""E1 crossed 5x5 factorial experiment for sample-selection and training seeds.

Run from the project root:
  python .\\experiments\\open_world_e1_crossed_5x5.py

The script reuses the validated E1 v2 data loading, scaling, training,
metrics, and checkpoint implementation. It does not modify either existing
E1 script. Each (sampling_seed, training_seed) pair is run exactly once.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import open_world_e1_randomness_decomposition as e1
import open_world_static_100_baseline_v2 as base

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
LABEL_BUDGET = 100
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EXPERIMENT_ID = "E1-CROSS-01"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def two_factor_decomposition(matrix: np.ndarray) -> dict[str, Any]:
    """Descriptive balanced two-factor sums-of-squares decomposition.

    With one run per cell, interaction is estimable as a descriptive component,
    but there is no independent within-cell error estimate for inferential
    ANOVA tests. Therefore no p-values or significance claims are produced.
    """
    if matrix.ndim != 2 or matrix.shape != (5, 5):
        raise ValueError(f"Expected a 5x5 outcome matrix, got {matrix.shape}")
    grand_mean = float(np.mean(matrix))
    sampling_means = np.mean(matrix, axis=1)
    training_means = np.mean(matrix, axis=0)
    ss_sampling = float(matrix.shape[1] * np.sum((sampling_means - grand_mean) ** 2))
    ss_training = float(matrix.shape[0] * np.sum((training_means - grand_mean) ** 2))
    interaction_residuals = (
        matrix
        - sampling_means[:, np.newaxis]
        - training_means[np.newaxis, :]
        + grand_mean
    )
    ss_interaction = float(np.sum(interaction_residuals**2))
    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    components = {
        "sampling_seed": ss_sampling,
        "training_seed": ss_training,
        "sampling_training_interaction": ss_interaction,
    }
    if not np.isclose(sum(components.values()), ss_total, rtol=1e-9, atol=1e-12):
        raise ArithmeticError("Two-factor sums of squares do not sum to total SS")
    shares = {
        name: (100.0 * value / ss_total if ss_total > 0 else 0.0)
        for name, value in components.items()
    }
    return {
        "grand_mean": grand_mean,
        "sum_of_squares": components,
        "total_sum_of_squares": ss_total,
        "share_of_total_variation_percent": shares,
        "degrees_of_freedom": {
            "sampling_seed": 4,
            "training_seed": 4,
            "interaction": 16,
            "total": 24,
            "independent_within_cell_error": 0,
        },
        "inferential_status": (
            "Descriptive only. One observation per cell gives no independent "
            "within-cell error estimate; do not calculate inferential F-tests "
            "or p-values from this design."
        ),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def generate_report(
    *,
    output_root: Path,
    sampling_seeds: list[int],
    training_seeds: list[int],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    matrix_accuracy: np.ndarray,
    matrix_macro_f1: np.ndarray,
) -> str:
    lines: list[str] = [
        f"# {EXPERIMENT_ID}: E1 Crossed Randomness Experiment (5 × 5)",
        "",
        "## 1. Experiment identification",
        f"- Experiment ID: `{EXPERIMENT_ID}`",
        "- Status: completed; all 25 cells executed successfully if this report exists",
        f"- Output directory: `{output_root}`",
        "",
        "## 2. Research objective",
        "Quantify how the initial labelled-row sample, PyTorch training randomness, and their interaction relate to future-batch performance under the locked E1 static-baseline protocol.",
        "",
        "## 3. Hypotheses",
        "- H1: Changing the sampling seed changes selected Batch 1 rows, fitted scaler, and downstream performance.",
        "- H2: Changing the training seed changes model initialization and minibatch order, and may change downstream performance while rows/scaler remain fixed.",
        "- H3: The effect of training seed may depend on which labelled sample was selected (sampling × training interaction).",
        "",
        "## 4. Experimental setup",
        f"- Dataset: UCI Gas Sensor Array Drift, batches 1–10; {summary['dataset']['total_rows']:,} validated rows.",
        f"- Sampling seeds: `{sampling_seeds}`",
        f"- Training seeds: `{training_seeds}`",
        "- Design: complete balanced 5 × 5 factorial; one run per seed pair (25 runs).",
        f"- Label budget: {LABEL_BUDGET}; uniformly sampled without replacement from Batch 1.",
        f"- Model: {summary['protocol']['model']}; CPU; Adam; learning rate {LEARNING_RATE}; {EPOCHS} epochs; batch size {BATCH_SIZE}.",
        "- Scaling: feature-wise min–max to [-1, 1], fitted only on the selected 100 training rows; no future clipping.",
        "- Future batches 2–10 are evaluation-only; no future updates, validation selection, or checkpoint selection.",
        "",
        "## 5. Methodology",
        "For each seed pair, NumPy's sampling generator selects 100 Batch 1 rows using the sampling seed. The existing E1 helper fits the scaler on these rows and invokes the fixed-schedule MLP trainer using the training seed. The final-epoch model is evaluated on unselected Batch 1 rows and batches 2–10. Pooled future accuracy and fixed-six-class macro-F1 are the primary outcomes.",
        "",
        "## 6. Results",
        "### 6.1 All 25 seed combinations",
        "| Sampling seed | Training seed | Accuracy | Macro-F1 (6 classes) | Selected-row hash |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['sampling_seed']} | {row['training_seed']} | "
            f"{fmt(row['pooled_future_accuracy'])} | "
            f"{fmt(row['pooled_future_macro_f1'])} | `{row['selected_row_hash'][:12]}…` |"
        )

    lines.extend([
        "",
        "### 6.2 Overall outcome summaries",
        "| Metric | Mean across 25 cells | Sample SD | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric_name, values in [
        ("Pooled future accuracy", matrix_accuracy),
        ("Pooled future Macro-F1", matrix_macro_f1),
    ]:
        lines.append(
            f"| {metric_name} | {fmt(float(np.mean(values)))} | "
            f"{fmt(float(np.std(values, ddof=1)))} | "
            f"{fmt(float(np.min(values)))} | {fmt(float(np.max(values)))} |"
        )

    lines.extend([
        "",
        "### 6.3 Sampling-seed marginal means",
        "| Sampling seed | Mean accuracy | Mean Macro-F1 |",
        "|---:|---:|---:|",
    ])
    for idx, seed in enumerate(sampling_seeds):
        lines.append(
            f"| {seed} | {fmt(float(np.mean(matrix_accuracy[idx, :])))} | "
            f"{fmt(float(np.mean(matrix_macro_f1[idx, :])))} |"
        )
    lines.extend([
        "",
        "### 6.4 Training-seed marginal means",
        "| Training seed | Mean accuracy | Mean Macro-F1 |",
        "|---:|---:|---:|",
    ])
    for idx, seed in enumerate(training_seeds):
        lines.append(
            f"| {seed} | {fmt(float(np.mean(matrix_accuracy[:, idx])))} | "
            f"{fmt(float(np.mean(matrix_macro_f1[:, idx])))} |"
        )

    lines.extend([
        "",
        "## 7. Statistical analysis",
        "### 7.1 Descriptive two-factor sums-of-squares decomposition",
        "| Component | Accuracy SS share | Macro-F1 SS share |",
        "|---|---:|---:|",
    ])
    for component in ["sampling_seed", "training_seed", "sampling_training_interaction"]:
        label = component.replace("_", " ").title()
        a_share = summary["factorial_decomposition"]["accuracy"]["share_of_total_variation_percent"][component]
        f_share = summary["factorial_decomposition"]["macro_f1"]["share_of_total_variation_percent"][component]
        lines.append(f"| {label} | {a_share:.2f}% | {f_share:.2f}% |")
    lines.extend([
        "",
        "This is a descriptive decomposition over the specified 5 × 5 seed grid, not a population-level variance-components estimate. There is one observation per cell, so the interaction term has no independent within-cell error estimate. No inferential F-tests, p-values, or statistical-significance claims are produced.",
        "",
        "## 8. Observations and interpretation",
        "Use the marginal means and interaction shares to describe the observed seed sensitivity in this exact grid. A larger share indicates more variation associated with that factor in this finite design; it does not prove that the factor dominates for other seed values, datasets, models, or environments.",
        "",
        "## 9. Limitations — what this experiment does not establish",
        "- This remains a static supervised baseline, not the Open-World AI agent or a continual-learning policy.",
        "- It does not test online adaptation, label acquisition decisions, memory operations, compute allocation, open-set recognition, or forgetting control.",
        "- The five seed values are a finite selected grid; results may not generalize to other seeds or datasets.",
        "- One run per cell prevents an independent estimate of within-cell experimental error and formal significance testing of interaction.",
        "- Batch chronology and class support are inherited from the dataset; pooled metrics can mask per-batch and class-specific behaviour. Inspect per-batch metrics in each run's `metrics.json`.",
        "",
        "## 10. Conclusion",
        "Conclusion is restricted to the measured 5 × 5 seed grid. The report records observed descriptive differences only; it does not claim that either source of randomness is universally causal or that the research hypothesis for Open-World AI has been validated.",
        "",
        "## 11. Reproducibility record",
        f"- Python: {summary['provenance']['python_version']}",
        f"- NumPy: {summary['provenance']['numpy_version']}",
        f"- PyTorch: {summary['provenance']['torch_version']}",
        f"- Platform: {summary['provenance']['platform']}",
        f"- Crossed-runner SHA-256: `{summary['provenance']['crossed_runner_sha256']}`",
        f"- E1 randomness helper SHA-256: `{summary['provenance']['e1_helper_sha256']}`",
        f"- Static baseline helper SHA-256: `{summary['provenance']['static_baseline_helper_sha256']}`",
        f"- Dataset directory: `{summary['dataset']['directory']}`",
        "- Each run folder contains `metrics.json`, `metrics.csv`, `selected_training_rows.csv`, and `model_and_scaler.pt`.",
        "- Combined outputs: `crossed_results.csv`, `summary.json`, and this report.",
        "",
        "## 12. Next experiment",
        "After reviewing these results, consider a replicated crossed design with additional independent seed values and/or repeated execution of selected cells to estimate run-to-run reproducibility. Then proceed to a controlled continual-learning benchmark comparison; keep that separate from this static-seed audit.",
        "",
        "---",
        "**Integrity note:** This report is generated from the saved run metrics. No result is inserted unless its corresponding run completed and produced a metrics record.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the E1 complete 5x5 crossed sampling/training seed experiment"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "experiments" / "results" / "e1_crossed_5x5_sampling42-46_training42-46",
    )
    args = parser.parse_args()

    for name, seeds in [
        ("--sampling-seeds", args.sampling_seeds),
        ("--training-seeds", args.training_seeds),
    ]:
        if len(seeds) != 5 or len(set(seeds)) != 5:
            parser.error(f"{name} must contain exactly five distinct integers")
        if any(seed < 0 for seed in seeds):
            parser.error(f"{name} values must be non-negative")

    output_root: Path = args.output_root.resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_root}\n"
                "Choose a fresh --output-root so previous results are never overwritten."
            )

    data_paths = [args.data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in data_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s):\n" + "\n".join(map(str, missing)))

    print("Loading and validating the ten dataset batches...")
    batches = [base.load_batch(path) for path in data_paths]
    total_rows = 0
    for index, (_, labels) in enumerate(batches, start=1):
        total_rows += int(len(labels))
        print(f"Batch {index}: {len(labels)} rows validated")

    torch.set_num_threads(1)
    output_root.mkdir(parents=True, exist_ok=True)
    planned = len(args.sampling_seeds) * len(args.training_seeds)
    records: list[dict[str, Any]] = []

    for i, sampling_seed in enumerate(args.sampling_seeds, start=1):
        for j, training_seed in enumerate(args.training_seeds, start=1):
            run_dir = output_root / f"sampling_{sampling_seed}_training_{training_seed}"
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
            print(
                f"\n[{len(records) + 1}/{planned}] "
                f"sampling_seed={sampling_seed}; training_seed={training_seed}"
            )
            record = e1.save_run(
                output_dir=run_dir,
                mode="crossed",
                sampling_seed=sampling_seed,
                training_seed=training_seed,
                batches=batches,
                data_paths=data_paths,
                runner_path=Path(__file__).resolve(),
            )
            # The shared E1 helper labels every non-selection run as training-only.
            # Correct that metadata explicitly for this complete crossed design.
            metrics_path = run_dir / "metrics.json"
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics["protocol"]["mode_interpretation"] = (
                "both sampling_seed and training_seed vary across a complete "
                "5x5 crossed factorial design; this run is one seed-pair cell"
            )
            metrics["protocol"]["reference_anchor"] = (
                "sampling_seed=42 and training_seed=42 should reproduce the "
                "E1 seed-42 anchor under the same helper implementation and runtime"
            )
            write_json(metrics_path, metrics)
            records.append(record)
            print(
                f"Pooled future accuracy={record['pooled_future_accuracy']:.6f}; "
                f"Macro-F1={record['pooled_future_macro_f1']:.6f}"
            )
            print(f"Saved: {run_dir}")

    expected_pairs = {
        (s, t) for s in args.sampling_seeds for t in args.training_seeds
    }
    observed_pairs = {(r["sampling_seed"], r["training_seed"]) for r in records}
    if len(records) != 25 or observed_pairs != expected_pairs:
        raise RuntimeError("Crossed run completeness check failed; expected 25 unique seed pairs")

    accuracy_matrix = np.asarray(
        [
            [
                next(r["pooled_future_accuracy"] for r in records
                     if r["sampling_seed"] == s and r["training_seed"] == t)
                for t in args.training_seeds
            ]
            for s in args.sampling_seeds
        ],
        dtype=float,
    )
    macro_f1_matrix = np.asarray(
        [
            [
                next(r["pooled_future_macro_f1"] for r in records
                     if r["sampling_seed"] == s and r["training_seed"] == t)
                for t in args.training_seeds
            ]
            for s in args.sampling_seeds
        ],
        dtype=float,
    )

    # Verify that a sampling seed selects the same rows regardless of training seed.
    row_hash_consistency: dict[str, bool] = {}
    for sampling_seed in args.sampling_seeds:
        hashes = {
            r["selected_row_hash"] for r in records
            if r["sampling_seed"] == sampling_seed
        }
        row_hash_consistency[str(sampling_seed)] = len(hashes) == 1
    if not all(row_hash_consistency.values()):
        raise RuntimeError("Selected-row hash changed across training seeds for a fixed sampling seed")

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "E1 complete crossed randomness design",
        "status": "completed",
        "design": {
            "type": "balanced two-factor crossed factorial",
            "sampling_seeds": args.sampling_seeds,
            "training_seeds": args.training_seeds,
            "rows": len(args.sampling_seeds),
            "columns": len(args.training_seeds),
            "planned_runs": planned,
            "completed_runs": len(records),
            "one_observation_per_cell": True,
        },
        "objective": "Describe sampling-seed, training-seed, and interaction effects on pooled future-batch performance under the fixed E1 static baseline protocol.",
        "protocol": {
            "label_budget": LABEL_BUDGET,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model": "128 -> 64 -> ReLU -> 6 MLP (as defined by E1 baseline helper)",
            "optimizer": "Adam",
            "device": "CPU",
            "future_batch_updates": False,
            "validation_or_checkpoint_selection": False,
            "scaling_fit": "selected 100 Batch 1 rows only",
            "primary_metrics": ["pooled future accuracy", "pooled future fixed-six-class macro-F1"],
        },
        "dataset": {
            "name": "UCI Gas Sensor Array Drift Dataset",
            "batches": 10,
            "total_rows": total_rows,
            "directory": str(args.data_dir.resolve()),
        },
        "quality_checks": {
            "all_25_seed_pairs_completed": len(records) == 25,
            "seed_pairs_unique": len(observed_pairs) == 25,
            "fixed_sampling_seed_selects_same_rows_across_training_seeds": row_hash_consistency,
        },
        "metrics_summary": {
            "pooled_future_accuracy": {
                "mean": float(np.mean(accuracy_matrix)),
                "sample_sd": float(np.std(accuracy_matrix, ddof=1)),
                "minimum": float(np.min(accuracy_matrix)),
                "maximum": float(np.max(accuracy_matrix)),
            },
            "pooled_future_macro_f1": {
                "mean": float(np.mean(macro_f1_matrix)),
                "sample_sd": float(np.std(macro_f1_matrix, ddof=1)),
                "minimum": float(np.min(macro_f1_matrix)),
                "maximum": float(np.max(macro_f1_matrix)),
            },
        },
        "factorial_decomposition": {
            "accuracy": two_factor_decomposition(accuracy_matrix),
            "macro_f1": two_factor_decomposition(macro_f1_matrix),
        },
        "sampling_seed_marginal_means": [
            {
                "sampling_seed": seed,
                "mean_accuracy": float(np.mean(accuracy_matrix[i, :])),
                "mean_macro_f1": float(np.mean(macro_f1_matrix[i, :])),
            }
            for i, seed in enumerate(args.sampling_seeds)
        ],
        "training_seed_marginal_means": [
            {
                "training_seed": seed,
                "mean_accuracy": float(np.mean(accuracy_matrix[:, j])),
                "mean_macro_f1": float(np.mean(macro_f1_matrix[:, j])),
            }
            for j, seed in enumerate(args.training_seeds)
        ],
        "runs": records,
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "crossed_runner_sha256": base.sha256_file(Path(__file__).resolve()),
            "e1_helper_sha256": base.sha256_file(Path(e1.__file__).resolve()),
            "static_baseline_helper_sha256": base.sha256_file(Path(base.__file__).resolve()),
            "dataset_files": [
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
                }
                for path in data_paths
            ],
        },
    }

    # Combined outputs.
    csv_path = output_root / "crossed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sampling_seed", "training_seed", "selected_row_hash",
            "pooled_future_samples", "pooled_future_accuracy",
            "pooled_future_macro_f1", "selected_class_counts", "output_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["selected_class_counts"] = json.dumps(record["selected_class_counts"], sort_keys=True)
            writer.writerow({key: row.get(key) for key in fields})

    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    report_text = generate_report(
        output_root=output_root,
        sampling_seeds=args.sampling_seeds,
        training_seeds=args.training_seeds,
        records=records,
        summary=summary,
        matrix_accuracy=accuracy_matrix,
        matrix_macro_f1=macro_f1_matrix,
    )
    report_path = output_root / "research_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print("\nCROSSED EXPERIMENT COMPLETE")
    print(f"Completed runs: {len(records)}/25")
    print(f"Mean pooled future accuracy: {np.mean(accuracy_matrix):.6f}")
    print(f"Mean pooled future Macro-F1: {np.mean(macro_f1_matrix):.6f}")
    print(f"Results CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Research report: {report_path}")
    print("Each run folder contains metrics.json, metrics.csv, selected_training_rows.csv, and model_and_scaler.pt.")


if __name__ == "__main__":
    main()
