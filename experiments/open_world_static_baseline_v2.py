"""Static supervised baseline for the Open-World AI gas-sensor benchmark.

Protocol:
- Stratified 80/20 split of Batch 1 (seed 42).
- Fit per-feature min-max scaling on Batch 1 training rows only.
- Map training-range values to [-1, 1]; do not clip future values.
- Train a small MLP on Batch 1 training rows.
- Select the checkpoint by Batch 1 selection-set macro-F1; this set is not an independent test.
- Evaluate Batch 1 validation and Batches 2-10 without any model updates.

Run from the project root, or pass --data-dir explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import copy
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
SEED = 42


class SmallMLP(nn.Module):
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
    """Read one LIBSVM-style batch; feature indices are expected to be 1..128."""
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
                raise ValueError(
                    f"{path.name}:{line_number}: class {label} is outside 1..{N_CLASSES}"
                )

            row = np.zeros(N_FEATURES, dtype=np.float64)
            seen: set[int] = set()
            for item in parts[1:]:
                try:
                    index_text, value_text = item.split(":", 1)
                    feature_index = int(index_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}:{line_number}: invalid feature token {item!r}"
                    ) from exc

                if not 1 <= feature_index <= N_FEATURES:
                    raise ValueError(
                        f"{path.name}:{line_number}: feature index {feature_index} is outside 1..{N_FEATURES}"
                    )
                if feature_index in seen:
                    raise ValueError(
                        f"{path.name}:{line_number}: duplicate feature index {feature_index}"
                    )
                if not np.isfinite(value):
                    raise ValueError(
                        f"{path.name}:{line_number}: non-finite value for feature {feature_index}"
                    )

                row[feature_index - 1] = value
                seen.add(feature_index)

            if len(seen) != N_FEATURES:
                raise ValueError(
                    f"{path.name}:{line_number}: expected {N_FEATURES} features, found {len(seen)}"
                )

            features.append(row)
            labels.append(label)

    if not features:
        raise ValueError(f"No data rows found in {path}")

    x = np.vstack(features)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def stratified_split(
    labels: np.ndarray, train_fraction: float = 0.80, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_id in range(1, N_CLASSES + 1):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 2:
            raise ValueError(
                f"Batch 1 needs at least two examples of each class; class {class_id} has {len(class_indices)}"
            )
        shuffled = rng.permutation(class_indices)
        split_at = int(round(len(shuffled) * train_fraction))
        split_at = min(max(split_at, 1), len(shuffled) - 1)
        train_indices.extend(shuffled[:split_at].tolist())
        validation_indices.extend(shuffled[split_at:].tolist())

    train_array = rng.permutation(np.asarray(train_indices, dtype=np.int64))
    validation_array = rng.permutation(np.asarray(validation_indices, dtype=np.int64))
    return train_array, validation_array


def fit_minmax_train_only(
    x_train: np.ndarray, x_to_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    minimum = x_train.min(axis=0)
    maximum = x_train.max(axis=0)
    span = maximum - minimum
    constant_features = span == 0
    safe_span = span.copy()
    safe_span[constant_features] = 1.0

    transformed = 2.0 * (x_to_transform - minimum) / safe_span - 1.0
    # Constant training features carry no training variation; represent them as zero.
    transformed[:, constant_features] = 0.0

    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values produced during scaling")
    return transformed, minimum, maximum, constant_features


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


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy() + 1
    return predictions.astype(np.int64)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[SmallMLP, int, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SmallMLP().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_x = torch.as_tensor(x_train, dtype=torch.float32, device="cpu")
    train_y = torch.as_tensor(y_train - 1, dtype=torch.long, device="cpu")
    rng = torch.Generator(device="cpu").manual_seed(seed)

    training_history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=rng)
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

        validation_predictions = predict(model, x_validation)
        validation_metrics = calculate_metrics(y_validation, validation_predictions)
        training_history.append({
            "epoch": epoch,
            "mean_training_cross_entropy": weighted_loss / seen,
            "selection_set_macro_f1_all_six": validation_metrics["macro_f1"],
            "selection_set_accuracy": validation_metrics["accuracy"],
        })
        current_key = (validation_metrics["macro_f1"], validation_metrics["accuracy"])
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics, training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the static gas-sensor baseline")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_root / "data" / "raw" / "gas_sensor_array_drift" / "Dataset",
        help="Directory containing batch1.dat through batch10.dat",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in the output directory")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")

    data_dir: Path = args.data_dir
    output_dir = args.output_dir or (default_root / "experiments" / "results" / f"static_baseline_unconstrained_seed{args.seed}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory already contains files: {output_dir}. Choose --output-dir or pass --overwrite.")
    batch_paths = [data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in batch_paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing batch file(s):\n{missing_text}")

    print("Loading and validating all 10 batches...")
    batches = [load_batch(path) for path in batch_paths]
    for index, (_, labels) in enumerate(batches, start=1):
        print(f"Batch {index}: {len(labels)} rows validated")

    x_batch1, y_batch1 = batches[0]
    train_indices, validation_indices = stratified_split(y_batch1, seed=args.seed)
    x_train_raw = x_batch1[train_indices]
    y_train = y_batch1[train_indices]
    x_validation_raw = x_batch1[validation_indices]
    y_validation = y_batch1[validation_indices]

    # Fit scaling parameters exclusively on the Batch 1 training subset.
    x_train, scaler_min, scaler_max, constant_features = fit_minmax_train_only(
        x_train_raw, x_train_raw
    )
    x_validation, _, _, _ = fit_minmax_train_only(x_train_raw, x_validation_raw)

    print("\nTraining static MLP on CPU...")
    model, best_epoch, validation_metrics, training_history = train_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    steps_per_epoch = int(np.ceil(len(y_train) / args.batch_size))
    results: dict[str, Any] = {
        "protocol": {
            "name": "unconstrained_static_supervised_reference",
            "device": "CPU",
            "seed": args.seed,
            "batch1_split": "stratified 80/20",
            "scaling": "feature-wise min-max to [-1, 1], fit on Batch 1 training subset only",
            "future_value_clipping": False,
            "future_batch_updates": False,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "checkpoint_selection": "highest Batch 1 selection-set macro-F1; accuracy breaks ties",
            "metric_policy": "Undefined precision/recall/F1 are null; fixed-six macro-F1 treats undefined class F1 as zero; present-class macro-F1 averages classes with true support > 0.",
            "batch1_selection_set_is_independent_test": False,
            "best_epoch": best_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "optimizer_steps_executed_all_epochs": steps_per_epoch * args.epochs,
            "optimizer_steps_represented_by_selected_checkpoint": steps_per_epoch * best_epoch,
            "train_samples": int(len(y_train)),
            "selection_set_samples": int(len(y_validation)),
            "constant_features_in_training_split_zero_based": np.flatnonzero(constant_features).astype(int).tolist(),
            "comparison_note": "This is an unconstrained 355-label reference, not a label-budget-matched control for E1.",
        },
        "training_history": training_history,
        "provenance": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "data_directory": str(data_dir.resolve()),
            "dataset_files": [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in batch_paths],
        },
        "batch_results": {},
        "scaling_diagnostics": {},
    }

    results["batch_results"]["batch1_selection_set"] = validation_metrics
    pooled_future_y: list[np.ndarray] = []
    pooled_future_predictions: list[np.ndarray] = []
    for batch_number in range(2, 11):
        x_future_raw, y_future = batches[batch_number - 1]
        x_future, _, _, future_constants = fit_minmax_train_only(x_train_raw, x_future_raw)
        predictions = predict(model, x_future)
        results["batch_results"][f"batch{batch_number}"] = calculate_metrics(y_future, predictions)
        results["scaling_diagnostics"][f"batch{batch_number}"] = scaling_diagnostics(x_future, future_constants)
        pooled_future_y.append(y_future)
        pooled_future_predictions.append(predictions)

    pooled_future_metrics = calculate_metrics(
        np.concatenate(pooled_future_y), np.concatenate(pooled_future_predictions)
    )
    results["pooled_future_batches_2_to_10"] = pooled_future_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    model_path = output_dir / "model_and_scaler.pt"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "samples", "accuracy", "macro_f1_all_six_undefined_as_zero", "macro_f1_present_classes_only"])
        for evaluation, metrics in results["batch_results"].items():
            writer.writerow([
                evaluation,
                metrics["samples"],
                f"{metrics['accuracy']:.6f}",
                f"{metrics['macro_f1']:.6f}",
                "" if metrics["macro_f1_present_classes_only"] is None else f"{metrics['macro_f1_present_classes_only']:.6f}",
            ])
        writer.writerow([
            "pooled_future_batches_2_to_10",
            pooled_future_metrics["samples"],
            f"{pooled_future_metrics['accuracy']:.6f}",
            f"{pooled_future_metrics['macro_f1']:.6f}",
            f"{pooled_future_metrics['macro_f1_present_classes_only']:.6f}",
        ])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
            "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        model_path,
    )

    print("\nSTATIC BASELINE RESULTS (no updates after Batch 1)")
    print(f"Best epoch selected on Batch 1 selection set (not independent test): {best_epoch}")
    print(f"{'Evaluation':<24} {'N':>6} {'Accuracy':>10} {'Macro-F1(6)':>13} {'Macro-F1(present)':>18}")
    print("-" * 76)
    for evaluation, metrics in results["batch_results"].items():
        print(
            f"{evaluation:<24} {metrics['samples']:>6} "
            f"{metrics['accuracy']:>10.4f} {metrics['macro_f1']:>13.4f} "
            f"{metrics['macro_f1_present_classes_only']:>18.4f}"
        )

    print(
        f"{'pooled_future_batches_2_to_10':<24} {pooled_future_metrics['samples']:>6} "
        f"{pooled_future_metrics['accuracy']:>10.4f} {pooled_future_metrics['macro_f1']:>13.4f} "
        f"{pooled_future_metrics['macro_f1_present_classes_only']:>18.4f}"
    )

    print("\nSaved files:")
    print(report_path)
    print(csv_path)
    print(model_path)
    print("\nNote: Batch 1 selection-set metrics are checkpoint-selection scores, not an independent test estimate.")
    print("Batches 2-10 were evaluated only; their labels were not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
