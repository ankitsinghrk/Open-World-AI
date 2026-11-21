import numpy as np
import torch
import torch.nn as nn

from src.uncertainty import UncertaintyEstimator


class SmallMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.network(x)


class PredictionProvider:

    def __init__(self, checkpoint_path):
        self.device = torch.device("cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        required = {
            "model_state_dict",
            "scaler_min",
            "scaler_max",
            "class_names"
        }

        if not required.issubset(checkpoint):
            raise ValueError("Invalid checkpoint")

        self.class_names = checkpoint["class_names"]

        if set(self.class_names.keys()) != set(range(1, 7)):
            raise ValueError("Invalid class mapping")

        self.minimum = checkpoint["scaler_min"].numpy().copy()
        self.maximum = checkpoint["scaler_max"].numpy().copy()

        if self.minimum.shape != (128,) or self.maximum.shape != (128,):
            raise ValueError("Invalid scaler dimensions")

        if not np.isfinite(self.minimum).all():
            raise ValueError("Invalid scaler minimum")

        if not np.isfinite(self.maximum).all():
            raise ValueError("Invalid scaler maximum")

        if np.any(self.maximum < self.minimum):
            raise ValueError("Invalid scaler range")

        self.model = SmallMLP().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.uncertainty_estimator = UncertaintyEstimator()

    def predict(self, raw_observation):

        features = np.asarray(raw_observation, dtype=np.float64)

        if features.shape != (128,):
            raise ValueError("Expected exactly 128 features")

        if not np.isfinite(features).all():
            raise ValueError("Features must be finite")

        span = self.maximum - self.minimum
        safe_span = np.where(span == 0, 1.0, span)

        scaled = 2.0 * (features - self.minimum) / safe_span - 1.0
        scaled[span == 0] = 0.0

        if not np.isfinite(scaled).all():
            raise ValueError("Scaled features must be finite")

        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits.double(), dim=1)[0].tolist()

        prediction = int(np.argmax(probabilities)) + 1
        uncertainty = self.uncertainty_estimator.calculate(probabilities)

        return {
            "probabilities": probabilities,
            "prediction": prediction,
            "uncertainty": uncertainty,
        }
