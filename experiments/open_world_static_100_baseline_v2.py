"""100-label static baseline for the Open-World AI gas-sensor benchmark.

Protocol:
- Select exactly 100 Batch 1 rows uniformly at random, independently of labels.
- Reveal/use labels only for those selected rows during training.
- Fit feature-wise min-max scaling on the selected 100 rows only.
- Train the same 128 -> 64 -> ReLU -> 6 MLP for a fixed 60 epochs.
- Do not use validation data or checkpoint selection.
- Evaluate the unselected Batch 1 rows and Batches 2-10 after training only.
- Never update the model on evaluation batches; never clip future scaled values.

Run from the project root or pass --data-dir explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

N_FEATURES = 128
N_CLASSES = 6
CLASS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}
DEFAULT_SEED = 42
DEFAULT_LABEL_BUDGET = 100


class SmallMLP(nn.Module):
    """Same small MLP architecture used by the unconstrained static baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and strictly validate one LIBSVM-style 128-feature batch."""
    features: list[np.ndarray] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                label = int(parts[0])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid class label") from exc
            if label not in CLASS_NAMES:
                raise ValueError(f"{path.name}:{line_number}: class {label} is outside 1..6")

            row = np.zeros(N_FEATURES, dtype=np.float64)
            seen: set[int] = set()
            for token in parts[1:]:
                try:
                    index_text, value_text = token.split(":", 1)
                    feature_index = int(index_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}:{line_number}: invalid feature token {token!r}"
                    ) from exc
                if not 1 <= feature_index <= N_FEATURES:
                    raise ValueError(
                        f"{path.name}:{line_number}: feature index {feature_index} is outside 1..128"
                    )
                if feature_index in seen:
                    raise ValueError(
                        f"{path.name}:{line_number}: duplicate feature index {feature_index}"
                    )
                if not np.isfinite(value):
                    raise ValueError(
                        f"{path.name}:{line_number}: non-finite feature value"
                    )
                row[feature_index - 1] = value
                seen.add(feature_index)

            if len(seen) != N_FEATURES:
                raise ValueError(
                    f"{path.name}:{line_number}: expected 128 features, found {len(seen)}"
                )
            features.append(row)
            labels.append(label)

    if not features:
        raise ValueError(f"No data rows found in {path}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit min/max on the labelled training sample and return its scaled values."""
    minimum = x_train.min(axis=0)
    maximum = x_train.max(axis=0)
    span = maximum - minimum
    constant_features = span == 0
    safe_span = span.copy()
    safe_span[constant_features] = 1.0
    scaled = 2.0 * (x_train - minimum) / safe_span - 1.0
    scaled[:, constant_features] = 0.0
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values produced while scaling training data")
    return scaled, minimum, maximum


def transform_with_scaler(
    x: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Apply the fixed training scaler without clipping future values."""
    span = maximum - minimum
    constant_features = span == 0
    safe_span = span.copy()
    safe_span[constant_features] = 1.0
    scaled = 2.0 * (x - minimum) / safe_span - 1.0
    scaled[:, constant_features] = 0.0
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values produced during transformation")
    return scaled


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return explicit fixed-six and present-classes metrics.

    Undefined precision/recall/F1 values are represented as None, not numeric zero.
    The legacy six-class macro-F1 treats an undefined class F1 as zero so that
    historical E0 values remain reproducible; present-class macro-F1 averages
    only classes with at least one true example in this evaluation batch.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation set")

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        actual_id = int(actual)
        predicted_id = int(predicted)
        if actual_id not in CLASS_NAMES or predicted_id not in CLASS_NAMES:
            raise ValueError("Labels and predictions must be in the range 1..6")
        confusion[actual_id - 1, predicted_id - 1] += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_all_six: list[float] = []
    f1_present: list[float] = []
    present_class_ids: list[int] = []

    for class_id in range(1, N_CLASSES + 1):
        i = class_id - 1
        tp = int(confusion[i, i])
        predicted_count = int(confusion[:, i].sum())
        support = int(confusion[i, :].sum())
        fp = predicted_count - tp
        fn = support - tp

        precision = tp / predicted_count if predicted_count > 0 else None
        recall = tp / support if support > 0 else None
        f1_denominator = 2 * tp + fp + fn
        f1 = (2.0 * tp / f1_denominator) if f1_denominator > 0 else None

        # For backward compatibility, undefined class F1 contributes zero to
        # the fixed six-class macro-F1. Its undefined status is also explicit.
        f1_all_six.append(0.0 if f1 is None else f1)
        if support > 0:
            present_class_ids.append(class_id)
            f1_present.append(0.0 if f1 is None else f1)

        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        }

    macro_all_six = float(np.mean(f1_all_six))
    macro_present = float(np.mean(f1_present)) if f1_present else None
    accuracy = float(np.trace(confusion) / confusion.sum())
    return {
        "samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": macro_all_six,
        "macro_f1_all_six_classes_undefined_as_zero": macro_all_six,
        "macro_f1_all_six_classes": macro_all_six,
        "macro_f1_present_classes_only": macro_present,
        "present_class_ids": present_class_ids,
        "per_class": per_class,
        "confusion_matrix_labels_1_to_6": confusion.tolist(),
    }


def scaling_diagnostics(
    x_scaled: np.ndarray, constant_features: np.ndarray
) -> dict[str, Any]:
    """Measure values outside the training range; do not clip them."""
    variable_mask = ~constant_features.astype(bool)
    if not np.any(variable_mask):
        return {
            "nonconstant_features_checked": 0,
            "scaled_values_checked": 0,
            "outside_training_range_values": 0,
            "outside_training_range_fraction": None,
            "max_abs_scaled_value": None,
        }
    values = x_scaled[:, variable_mask]
    outside = (values < -1.0) | (values > 1.0)
    return {
        "nonconstant_features_checked": int(np.sum(variable_mask)),
        "scaled_values_checked": int(values.size),
        "outside_training_range_values": int(np.sum(outside)),
        "outside_training_range_fraction": float(np.mean(outside)),
        "max_abs_scaled_value": float(np.max(np.abs(values))),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_fixed_schedule(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[SmallMLP, list[float]]:
    """Train for every prescribed epoch; no validation or checkpoint selection."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SmallMLP().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_x = torch.as_tensor(x_train, dtype=torch.float32, device="cpu")
    train_y = torch.as_tensor(y_train - 1, dtype=torch.long, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epoch_losses: list[float] = []

    for _epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        weighted_loss = 0.0
        seen = 0
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_count = len(indices)
            weighted_loss += float(loss.detach().item()) * batch_count
            seen += batch_count
        epoch_losses.append(weighted_loss / seen)

    return model, epoch_losses


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        return (model(tensor).argmax(dim=1).cpu().numpy() + 1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 100-label static gas-sensor baseline")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--label-budget", type=int, choices=[100], default=DEFAULT_LABEL_BUDGET,
                        help="Locked at 100 for this experiment")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in the output directory")
    args = parser.parse_args()

    if args.label_budget < 1 or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("label-budget, epochs, batch-size and learning-rate must be positive")

    data_dir: Path = args.data_dir
    output_dir = args.output_dir or (project_root / "experiments" / "results" / f"static_baseline_100_seed{args.seed}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory already contains files: {output_dir}. Choose --output-dir or pass --overwrite.")
    paths = [data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing batch file(s):\n" + "\n".join(map(str, missing)))

    print("Loading and validating all 10 batches...")
    batches = [load_batch(path) for path in paths]
    for i, (_, labels) in enumerate(batches, start=1):
        print(f"Batch {i}: {len(labels)} rows validated")

    x_batch1, y_batch1 = batches[0]
    if args.label_budget >= len(y_batch1):
        raise ValueError(
            f"Label budget ({args.label_budget}) must be smaller than Batch 1 size ({len(y_batch1)})"
        )

    # The row indices are selected using only Batch 1 row count and the seed.
    # The loader has already parsed/range-checked labels for file integrity; those
    # labels do not participate in row selection. The learner receives y_train
    # only for these 100 selected rows. The evaluation harness retains labels to
    # calculate post-prediction metrics.
    sampling_rng = np.random.default_rng(args.seed)
    selected_indices = sampling_rng.permutation(len(y_batch1))[: args.label_budget]
    selected_mask = np.zeros(len(y_batch1), dtype=bool)
    selected_mask[selected_indices] = True
    unselected_indices = np.flatnonzero(~selected_mask)

    x_train_raw = x_batch1[selected_indices]
    y_train = y_batch1[selected_indices]
    x_batch1_unselected = x_batch1[unselected_indices]
    y_batch1_unselected = y_batch1[unselected_indices]

    selected_row_numbers_sorted = np.sort(selected_indices) + 1
    selected_rows_sha256 = hashlib.sha256(
        ",".join(map(str, selected_row_numbers_sorted.tolist())).encode("utf-8")
    ).hexdigest()

    selected_class_counts = {
        class_id: int(np.sum(y_train == class_id)) for class_id in range(1, 7)
    }
    missing_training_classes = [
        class_id for class_id, count in selected_class_counts.items() if count == 0
    ]
    print(f"\nSelected {len(selected_indices)} rows uniformly (seed={args.seed}).")
    print(f"Selected-label class counts: {selected_class_counts}")
    if missing_training_classes:
        print(
            "WARNING: no labelled training examples for class(es): "
            + ", ".join(map(str, missing_training_classes))
        )

    # Fit scaler exclusively on the selected 100 labelled rows.
    x_train, scaler_min, scaler_max = fit_scaler(x_train_raw)

    print(f"\nTraining MLP on CPU for exactly {args.epochs} epochs (no validation selection)...")
    model, epoch_losses = train_fixed_schedule(
        x_train=x_train,
        y_train=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    batch_results: dict[str, Any] = {}
    range_diagnostics: dict[str, Any] = {}
    training_constant_features = (scaler_max - scaler_min) == 0

    # Batch 1 remainder is held out from training and used only for post-training audit.
    x_b1_eval = transform_with_scaler(x_batch1_unselected, scaler_min, scaler_max)
    batch_results["batch1_unselected"] = calculate_metrics(
        y_batch1_unselected, predict(model, x_b1_eval)
    )
    range_diagnostics["batch1_unselected"] = scaling_diagnostics(x_b1_eval, training_constant_features)

    pooled_future_y: list[np.ndarray] = []
    pooled_future_pred: list[np.ndarray] = []
    for batch_number in range(2, 11):
        x_future_raw, y_future = batches[batch_number - 1]
        x_future = transform_with_scaler(x_future_raw, scaler_min, scaler_max)
        future_predictions = predict(model, x_future)
        batch_results[f"batch{batch_number}"] = calculate_metrics(y_future, future_predictions)
        range_diagnostics[f"batch{batch_number}"] = scaling_diagnostics(x_future, training_constant_features)
        pooled_future_y.append(y_future)
        pooled_future_pred.append(future_predictions)

    pooled_metrics = calculate_metrics(
        np.concatenate(pooled_future_y), np.concatenate(pooled_future_pred)
    )

    results: dict[str, Any] = {
        "protocol": {
            "name": "100_label_random_static_supervised_baseline",
            "device": "CPU",
            "seed": args.seed,
            "batch1_sampling": "uniform random row sampling without replacement; independent of labels",
            "label_budget": int(args.label_budget),
            "selected_batch1_row_numbers_1_based": (selected_indices + 1).astype(int).tolist(),
            "selected_row_numbers_sorted_sha256": selected_rows_sha256,
            "label_access_boundary": "The learner is trained only on the selected 100 labels; evaluation labels are retained by the audit harness for metrics.",
            "selected_label_class_counts_after_sampling": selected_class_counts,
            "training_classes_missing_from_sample": missing_training_classes,
            "scaling": "feature-wise min-max to [-1, 1], fit on selected labelled rows only",
            "future_value_clipping": False,
            "future_batch_updates": False,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "fixed_epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_set_used": False,
            "checkpoint_selection": "none; final epoch model used",
            "metric_policy": "Undefined precision/recall/F1 are null; fixed-six macro-F1 treats undefined class F1 as zero; present-class macro-F1 averages classes with true support > 0.",
            "training_samples": int(len(y_train)),
            "optimizer_steps_per_epoch": int(np.ceil(len(y_train) / args.batch_size)),
            "optimizer_steps_total": int(np.ceil(len(y_train) / args.batch_size)) * args.epochs,
            "training_sample_present_class_count": int(len([v for v in selected_class_counts.values() if v > 0])),
            "batch1_unselected_evaluation_samples": int(len(y_batch1_unselected)),
            "constant_features_in_training_sample_zero_based": np.flatnonzero(
                scaler_max - scaler_min == 0
            ).astype(int).tolist(),
        },
        "training_history": {"epoch_mean_cross_entropy_loss": epoch_losses},
        "batch_results": batch_results,
        "pooled_future_batches_2_to_10": pooled_metrics,
        "scaling_diagnostics": range_diagnostics,
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "data_directory": str(data_dir.resolve()),
            "dataset_files": [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    manifest_path = output_dir / "selected_training_rows.csv"
    model_path = output_dir / "model_and_scaler.pt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "evaluation", "samples", "accuracy",
            "macro_f1_all_six_classes", "macro_f1_present_classes_only"
        ])
        for name, metrics in batch_results.items():
            writer.writerow([
                name, metrics["samples"], f"{metrics['accuracy']:.6f}",
                f"{metrics['macro_f1_all_six_classes']:.6f}",
                f"{metrics['macro_f1_present_classes_only']:.6f}",
            ])
        writer.writerow([
            "pooled_future_batches_2_to_10", pooled_metrics["samples"],
            f"{pooled_metrics['accuracy']:.6f}",
            f"{pooled_metrics['macro_f1_all_six_classes']:.6f}",
            f"{pooled_metrics['macro_f1_present_classes_only']:.6f}",
        ])

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch1_row_number_1_based", "class_id", "class_name"])
        for index in selected_indices:
            label = int(y_batch1[index])
            writer.writerow([int(index + 1), label, CLASS_NAMES[label]])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
            "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
            "class_names": CLASS_NAMES,
            "training_row_numbers_1_based": (selected_indices + 1).astype(int).tolist(),
            "fixed_epochs": args.epochs,
            "seed": args.seed,
        },
        model_path,
    )

    print("\n100-LABEL STATIC BASELINE RESULTS (no future updates)")
    print("Model used: final epoch; no validation/checkpoint selection")
    print(f"{'Evaluation':<28} {'N':>6} {'Accuracy':>10} {'Macro-F1(6)':>13} {'Macro-F1(present)':>18}")
    print("-" * 80)
    for name, metrics in batch_results.items():
        print(
            f"{name:<28} {metrics['samples']:>6} {metrics['accuracy']:>10.4f} "
            f"{metrics['macro_f1_all_six_classes']:>13.4f} "
            f"{metrics['macro_f1_present_classes_only']:>18.4f}"
        )
    print(
        f"{'pooled_future_batches_2_to_10':<28} {pooled_metrics['samples']:>6} "
        f"{pooled_metrics['accuracy']:>10.4f} "
        f"{pooled_metrics['macro_f1_all_six_classes']:>13.4f} "
        f"{pooled_metrics['macro_f1_present_classes_only']:>18.4f}"
    )

    print("\nSaved files:")
    print(json_path)
    print(csv_path)
    print(manifest_path)
    print(model_path)
    print("\nThe learner trained on exactly 100 selected labels. Evaluation labels were used only by the audit harness after prediction.")


if __name__ == "__main__":
    main()
