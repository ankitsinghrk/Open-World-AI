"""Static supervised baseline for the Open-World AI gas-sensor benchmark.

Protocol:
- Stratified 80/20 split of Batch 1 (seed 42).
- Fit per-feature min-max scaling on Batch 1 training rows only.
- Map training-range values to [-1, 1]; do not clip future values.
- Train a small MLP on Batch 1 training rows.
- Select the checkpoint by Batch 1 validation macro-F1.
- Evaluate Batch 1 validation and Batches 2-10 without any model updates.

Run from the project root, or pass --data-dir explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import copy
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
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual) - 1, int(predicted) - 1] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id in range(1, N_CLASSES + 1):
        index = class_id - 1
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {
        "samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix_labels_1_to_6": confusion.tolist(),
    }


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
) -> tuple[SmallMLP, int, dict[str, Any]]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = SmallMLP().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_x = torch.as_tensor(x_train, dtype=torch.float32, device="cpu")
    train_y = torch.as_tensor(y_train - 1, dtype=torch.long, device="cpu")
    rng = torch.Generator(device="cpu").manual_seed(SEED)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), generator=rng)
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x[indices]
            batch_y = train_y[indices]

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

        validation_predictions = predict(model, x_validation)
        validation_metrics = calculate_metrics(y_validation, validation_predictions)
        current_key = (validation_metrics["macro_f1"], validation_metrics["accuracy"])
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model, best_epoch, best_metrics


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
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")

    data_dir: Path = args.data_dir
    batch_paths = [data_dir / f"batch{i}.dat" for i in range(1, 11)]
    missing = [path for path in batch_paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing batch file(s):\n{missing_text}")

    print("Loading and validating all 10 batches...")
    batches = [load_batch(path) for path in batch_paths]
    for index, (_, labels) in enumerate(batches, start=1):
        counts = {class_id: int(np.sum(labels == class_id)) for class_id in range(1, 7)}
        print(f"Batch {index}: {len(labels)} rows | class counts: {counts}")

    x_batch1, y_batch1 = batches[0]
    train_indices, validation_indices = stratified_split(y_batch1)
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
    model, best_epoch, validation_metrics = train_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    results: dict[str, Any] = {
        "protocol": {
            "name": "static_supervised_baseline",
            "device": "CPU",
            "seed": SEED,
            "batch1_split": "stratified 80/20",
            "scaling": "feature-wise min-max to [-1, 1], fit on Batch 1 training subset only",
            "future_value_clipping": False,
            "future_batch_updates": False,
            "model": "128 -> 64 -> ReLU -> 6 MLP",
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "checkpoint_selection": "highest Batch 1 validation macro-F1; accuracy breaks ties",
            "best_epoch": best_epoch,
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "constant_features_in_training_split": np.flatnonzero(constant_features).astype(int).tolist(),
        },
        "batch_results": {},
    }

    results["batch_results"]["batch1_validation"] = validation_metrics
    for batch_number in range(2, 11):
        x_future_raw, y_future = batches[batch_number - 1]
        x_future, _, _, _ = fit_minmax_train_only(x_train_raw, x_future_raw)
        predictions = predict(model, x_future)
        results["batch_results"][f"batch{batch_number}"] = calculate_metrics(
            y_future, predictions
        )

    output_dir = default_root / "experiments" / "results" / "static_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    model_path = output_dir / "model_and_scaler.pt"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "samples", "accuracy", "macro_f1"])
        for evaluation, metrics in results["batch_results"].items():
            writer.writerow([
                evaluation,
                metrics["samples"],
                f"{metrics['accuracy']:.6f}",
                f"{metrics['macro_f1']:.6f}",
            ])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_min": torch.as_tensor(scaler_min, dtype=torch.float64),
            "scaler_max": torch.as_tensor(scaler_max, dtype=torch.float64),
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "seed": SEED,
        },
        model_path,
    )

    print("\nSTATIC BASELINE RESULTS (no updates after Batch 1)")
    print(f"Best epoch selected on Batch 1 validation: {best_epoch}")
    print(f"{'Evaluation':<20} {'N':>6} {'Accuracy':>12} {'Macro-F1':>12}")
    print("-" * 52)
    for evaluation, metrics in results["batch_results"].items():
        print(
            f"{evaluation:<20} {metrics['samples']:>6} "
            f"{metrics['accuracy']:>11.4f} {metrics['macro_f1']:>12.4f}"
        )

    print("\nSaved files:")
    print(report_path)
    print(csv_path)
    print(model_path)
    print("\nNote: Batches 2-10 were evaluated only; their labels were not used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
