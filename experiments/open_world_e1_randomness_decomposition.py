"""Separate sample-selection and training randomness for E1.

This runner deliberately reuses the validated E1 v2 implementation so that
loading, scaling, MLP architecture, optimizer, metrics, and checkpoint format
remain aligned with the ten-seed E1 baseline.

Mode A (--mode selection):
  - vary NumPy sampling seed
  - hold PyTorch training seed fixed

Mode B (--mode training):
  - hold NumPy sampling seed fixed
  - vary PyTorch training seed (initialization + minibatch order)

Default settings reproduce the E1 protocol: 100 labels, 60 epochs,
batch size 32, Adam lr=0.001, CPU, no validation/checkpoint selection,
and no future-batch updates.
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

import open_world_static_100_baseline_v2 as base

DEFAULT_SEEDS = list(range(42, 52))
LABEL_BUDGET = 100
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_run(
    *,
    output_dir: Path,
    mode: str,
    sampling_seed: int,
    training_seed: int,
    batches: list[tuple[np.ndarray, np.ndarray]],
    data_paths: list[Path],
    runner_path: Path,
) -> dict[str, Any]:
    x_batch1, y_batch1 = batches[0]
    if len(y_batch1) <= LABEL_BUDGET:
        raise ValueError(
            f"Batch 1 must contain more than {LABEL_BUDGET} rows; got {len(y_batch1)}"
        )

    # Sample selection is exclusively controlled by NumPy's sampling seed.
    sampling_rng = np.random.default_rng(sampling_seed)
    selected_indices = sampling_rng.permutation(len(y_batch1))[:LABEL_BUDGET]
    selected_mask = np.zeros(len(y_batch1), dtype=bool)
    selected_mask[selected_indices] = True
    unselected_indices = np.flatnonzero(~selected_mask)

    x_train_raw = x_batch1[selected_indices]
    y_train = y_batch1[selected_indices]
    selected_rows_sorted = np.sort(selected_indices) + 1
    selected_rows_hash = sha256_text(",".join(map(str, selected_rows_sorted.tolist())))

    class_counts = {
        class_id: int(np.sum(y_train == class_id)) for class_id in range(1, 7)
    }
    missing_classes = [class_id for class_id, count in class_counts.items() if count == 0]

    x_train, scaler_min, scaler_max = base.fit_scaler(x_train_raw)
    model, epoch_losses = base.train_fixed_schedule(
        x_train=x_train,
        y_train=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=training_seed,
    )

    constant_features = (scaler_max - scaler_min) == 0
    batch_results: dict[str, Any] = {}
    scale_results: dict[str, Any] = {}

    # The unselected Batch 1 rows are not used for training or model selection.
    x_b1_eval = base.transform_with_scaler(
        x_batch1[unselected_indices], scaler_min, scaler_max
    )
    batch_results["batch1_unselected"] = base.calculate_metrics(
        y_batch1[unselected_indices], base.predict(model, x_b1_eval)
    )
    scale_results["batch1_unselected"] = base.scaling_diagnostics(
        x_b1_eval, constant_features
    )

    pooled_y: list[np.ndarray] = []
    pooled_predictions: list[np.ndarray] = []
    for batch_number in range(2, 11):
        x_raw, y = batches[batch_number - 1]
        x_scaled = base.transform_with_scaler(x_raw, scaler_min, scaler_max)
        predictions = base.predict(model, x_scaled)
        batch_results[f"batch{batch_number}"] = base.calculate_metrics(y, predictions)
        scale_results[f"batch{batch_number}"] = base.scaling_diagnostics(
            x_scaled, constant_features
        )
        pooled_y.append(y)
        pooled_predictions.append(predictions)

    pooled = base.calculate_metrics(
        np.concatenate(pooled_y), np.concatenate(pooled_predictions)
    )

    helper_path = Path(base.__file__).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_in_selection_order = (selected_indices + 1).astype(int).tolist()

    result: dict[str, Any] = {
        "protocol": {
            "name": "E1_randomness_decomposition",
            "mode": mode,
            "mode_interpretation": (
                "sampling_seed varies; training_seed fixed"
                if mode == "selection"
                else "sampling_seed fixed; training_seed varies"
            ),
            "sampling_seed": int(sampling_seed),
            "training_seed": int(training_seed),
            "label_budget": LABEL_BUDGET,
            "sampling": "uniform random Batch 1 row sampling without replacement, independent of labels",
            "training_randomness": "PyTorch model initialization and minibatch permutation controlled by training_seed",
            "selected_batch1_row_numbers_1_based_in_sampling_order": rows_in_selection_order,
            "selected_row_numbers_sorted_sha256": selected_rows_hash,
            "selected_label_class_counts_after_sampling": class_counts,
            "training_classes_missing_from_sample": missing_classes,
            "scaling": "feature-wise min-max to [-1, 1], fit on selected 100 rows only",
            "future_value_clipping": False,
            "future_batch_updates": False,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "fixed_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_set_used": False,
            "checkpoint_selection": "none; final epoch model used",
            "metric_policy": "Undefined precision/recall/F1 are null; fixed-six macro-F1 treats undefined class F1 as zero; present-class macro-F1 averages classes with true support > 0.",
            "training_samples": int(len(y_train)),
            "optimizer_steps_per_epoch": int(np.ceil(len(y_train) / BATCH_SIZE)),
            "optimizer_steps_total": int(np.ceil(len(y_train) / BATCH_SIZE)) * EPOCHS,
            "batch1_unselected_evaluation_samples": int(len(unselected_indices)),
            "constant_features_in_training_sample_zero_based": np.flatnonzero(
                constant_features
            ).astype(int).tolist(),
            "reference_anchor": "sampling_seed=42 and training_seed=42 should reproduce E1 seed 42 metrics under the same helper implementation and runtime",
        },
        "training_history": {"epoch_mean_cross_entropy_loss": epoch_losses},
        "batch_results": batch_results,
        "pooled_future_batches_2_to_10": pooled,
        "scaling_diagnostics": scale_results,
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "runner_script_sha256": base.sha256_file(runner_path),
            "e1_helper_script_sha256": base.sha256_file(helper_path),
            "data_directory": str(data_paths[0].parent.resolve()),
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

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "evaluation", "samples", "accuracy",
            "macro_f1_all_six_classes", "macro_f1_present_classes_only",
        ])
        for name, metrics in batch_results.items():
            writer.writerow([
                name, metrics["samples"], f"{metrics['accuracy']:.10f}",
                f"{metrics['macro_f1_all_six_classes']:.10f}",
                f"{metrics['macro_f1_present_classes_only']:.10f}",
            ])
        writer.writerow([
            "pooled_future_batches_2_to_10", pooled["samples"],
            f"{pooled['accuracy']:.10f}",
            f"{pooled['macro_f1_all_six_classes']:.10f}",
            f"{pooled['macro_f1_present_classes_only']:.10f}",
        ])

    with (output_dir / "selected_training_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch1_row_number_1_based", "class_id", "class_name"])
        for index in selected_indices:
            label = int(y_batch1[index])
            writer.writerow([int(index + 1), label, base.CLASS_NAMES[label]])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
            "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
            "class_names": base.CLASS_NAMES,
            "training_row_numbers_1_based": rows_in_selection_order,
            "sampling_seed": sampling_seed,
            "training_seed": training_seed,
            "fixed_epochs": EPOCHS,
        },
        output_dir / "model_and_scaler.pt",
    )

    return {
        "mode": mode,
        "sampling_seed": sampling_seed,
        "training_seed": training_seed,
        "selected_row_hash": selected_rows_hash,
        "selected_class_counts": class_counts,
        "pooled_future_samples": pooled["samples"],
        "pooled_future_accuracy": pooled["accuracy"],
        "pooled_future_macro_f1": pooled["macro_f1_all_six_classes"],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate E1 sample-selection randomness from training randomness"
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--mode", choices=["selection", "training"], required=True,
        help="selection: vary sampling seed; training: vary training seed",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Ten varied seeds (default: 42 through 51)",
    )
    parser.add_argument(
        "--fixed-training-seed", type=int, default=42,
        help="Used in selection mode; default 42",
    )
    parser.add_argument(
        "--fixed-sampling-seed", type=int, default=42,
        help="Used in training mode; default 42",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=project_root / "experiments" / "results" / "e1_randomness_decomposition",
    )
    args = parser.parse_args()

    if len(args.seeds) == 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct seed values")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative integers")

    data_paths = [args.data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in data_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s):\n" + "\n".join(map(str, missing)))

    mode_root = args.output_root / ("selection_effect" if args.mode == "selection" else "training_effect")
    planned_dirs: list[Path] = []
    for seed in args.seeds:
        sampling_seed = seed if args.mode == "selection" else args.fixed_sampling_seed
        training_seed = args.fixed_training_seed if args.mode == "selection" else seed
        planned_dirs.append(mode_root / f"sampling_{sampling_seed}_training_{training_seed}")
    summary_json = mode_root / "summary.json"
    summary_csv = mode_root / "summary.csv"
    conflicts = [
        path for path in planned_dirs
        if path.exists() and any(path.iterdir())
    ]
    if summary_json.exists() or summary_csv.exists():
        conflicts.extend([path for path in [summary_json, summary_csv] if path.exists()])
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing experiment artifacts:\n"
            + "\n".join(map(str, conflicts))
            + "\nChoose a fresh --output-root to preserve previous results."
        )

    print("Loading and validating all 10 batches...")
    batches = [base.load_batch(path) for path in data_paths]
    for index, (_, labels) in enumerate(batches, start=1):
        print(f"Batch {index}: {len(labels)} rows validated")

    torch.set_num_threads(1)
    mode_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, seed in enumerate(args.seeds, start=1):
        sampling_seed = seed if args.mode == "selection" else args.fixed_sampling_seed
        training_seed = args.fixed_training_seed if args.mode == "selection" else seed
        run_dir = mode_root / f"sampling_{sampling_seed}_training_{training_seed}"
        print(
            f"\n[{position}/{len(args.seeds)}] mode={args.mode}; "
            f"sampling_seed={sampling_seed}; training_seed={training_seed}"
        )
        record = save_run(
            output_dir=run_dir,
            mode=args.mode,
            sampling_seed=sampling_seed,
            training_seed=training_seed,
            batches=batches,
            data_paths=data_paths,
            runner_path=Path(__file__).resolve(),
        )
        records.append(record)
        print(
            f"Pooled future accuracy={record['pooled_future_accuracy']:.6f}; "
            f"Macro-F1={record['pooled_future_macro_f1']:.6f}"
        )
        print(f"Saved: {run_dir}")

    summary = {
        "experiment": "E1 randomness decomposition",
        "mode": args.mode,
        "varied_seeds": args.seeds,
        "fixed_training_seed": args.fixed_training_seed if args.mode == "selection" else None,
        "fixed_sampling_seed": args.fixed_sampling_seed if args.mode == "training" else None,
        "runs": records,
        "interpretation": (
            "Variation primarily reflects selected-row composition and its fitted scaler, with PyTorch training randomness held fixed."
            if args.mode == "selection"
            else "Variation reflects PyTorch initialization and minibatch-order randomness, with selected rows and fitted scaler held fixed."
        ),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode", "sampling_seed", "training_seed", "selected_row_hash",
                "pooled_future_samples", "pooled_future_accuracy",
                "pooled_future_macro_f1", "selected_class_counts", "output_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            csv_record = dict(record)
            csv_record["selected_class_counts"] = json.dumps(
                record["selected_class_counts"], sort_keys=True
            )
            writer.writerow(csv_record)

    accuracies = np.asarray([row["pooled_future_accuracy"] for row in records], dtype=float)
    f1s = np.asarray([row["pooled_future_macro_f1"] for row in records], dtype=float)
    print("\nRANDOMNESS DECOMPOSITION SUMMARY")
    print(f"Mode: {args.mode}")
    print(f"Runs: {len(records)}")
    print(f"Mean accuracy: {float(np.mean(accuracies)):.6f}")
    print(f"Sample SD accuracy: {float(np.std(accuracies, ddof=1)) if len(records) > 1 else 0.0:.6f}")
    print(f"Mean Macro-F1: {float(np.mean(f1s)):.6f}")
    print(f"Sample SD Macro-F1: {float(np.std(f1s, ddof=1)) if len(records) > 1 else 0.0:.6f}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    print("Each run directory contains metrics.json, metrics.csv, selected_training_rows.csv, and model_and_scaler.pt.")


if __name__ == "__main__":
    main()
