from pathlib import Path
import numpy as np
import torch

from experiments.open_world_static_100_baseline_v2 import (
    load_batch, calculate_metrics, CLASS_NAMES
)
from src.prediction_provider import PredictionProvider

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/gas_sensor_array_drift/Dataset"
CHECKPOINT = ROOT / "experiments/results/static_baseline_100_seed42/model_and_scaler.pt"

metadata = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
training_rows = metadata["training_row_numbers_1_based"]

if hasattr(training_rows, "tolist"):
    training_rows = training_rows.tolist()

training_rows = [int(row) for row in training_rows]

x1, y1 = load_batch(DATA / "batch1.dat")

if len(training_rows) != 100:
    raise ValueError("Checkpoint must contain exactly 100 training rows.")

if len(set(training_rows)) != 100:
    raise ValueError("Duplicate training row numbers found.")

if min(training_rows) < 1 or max(training_rows) > len(y1):
    raise ValueError("Invalid training row numbers.")

train_mask = np.zeros(len(y1), dtype=bool)
train_mask[np.array(training_rows) - 1] = True

provider = PredictionProvider(str(CHECKPOINT))

all_true = []
all_pred = []
batch_results = []

def predict_rows(features):
    return np.asarray([
        provider.predict(row.tolist())["prediction"]
        for row in features
    ], dtype=np.int64)

for number in range(1, 11):
    x, y = load_batch(DATA / f"batch{number}.dat")

    if number == 1:
        x = x[~train_mask]
        y = y[~train_mask]

    predictions = predict_rows(x)
    metrics = calculate_metrics(y, predictions)

    batch_results.append((number, metrics))
    if number >= 2:
        all_true.extend(y.tolist())
        all_pred.extend(predictions.tolist())

overall = calculate_metrics(
    np.asarray(all_true),
    np.asarray(all_pred)
)

print("\nBATCH-WISE ACCURACY")
for number, metrics in batch_results:
    print(
        f"Batch {number}: "
        f"samples={metrics['samples']}, "
        f"accuracy={metrics['accuracy']:.4f}"
    )

print("\nOVERALL EVALUATION")
print("Samples:", overall["samples"])
print("Accuracy:", round(overall["accuracy"], 6))
print("Macro F1:", round(overall["macro_f1_all_six_classes"], 6))

print("\nCLASS-WISE RESULTS")
for class_id, result in overall["per_class"].items():
    print(
        class_id,
        CLASS_NAMES[int(class_id)],
        "recall=", result["recall"],
        "support=", result["support"],
        "predicted=", result["predicted_count"]
    )

print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
for row in overall["confusion_matrix_labels_1_to_6"]:
    print(row)

print("\nTRAINING ROWS EXCLUDED:", len(training_rows))
print("MODEL UPDATED DURING EVALUATION: NO")
