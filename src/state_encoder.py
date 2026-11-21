
import math


class StateEncoder:
    def __init__(self, feature_dim, history_size=5):
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")

        if not isinstance(history_size, int) or isinstance(history_size, bool) or history_size < 0:
            raise ValueError("history_size must be a non-negative integer")

        self.feature_dim = feature_dim
        self.history_size = history_size

        # Features + prediction + uncertainty + novelty
        # + memory, label and compute remaining + history
        self.state_dim = feature_dim + 6 + history_size

    def encode(self, state):
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        features = state.get("features")
        if not isinstance(features, (list, tuple)):
            raise ValueError("features must be a list or tuple")

        if len(features) != self.feature_dim:
            raise ValueError("Incorrect feature dimension")

        prediction = state.get("prediction")
        uncertainty = state.get("uncertainty")
        novelty = state.get("novelty")

        memory_remaining = state.get("memory_remaining")
        label_remaining = state.get("label_remaining")
        compute_remaining = state.get("compute_remaining")

        history = state.get("history")

        values = list(features) + [
            prediction,
            uncertainty,
            novelty,
            memory_remaining,
            label_remaining,
            compute_remaining
        ]

        if not isinstance(history, (list, tuple)):
            raise ValueError("history must be a list or tuple")

        history_values = list(history[-self.history_size:]) if self.history_size else []

        history_values = history_values + [0.0] * (
            self.history_size - len(history_values)
        )

        values.extend(history_values)

        if len(values) != self.state_dim:
            raise ValueError("Incorrect encoded state dimension")

        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError("All state values must be finite numbers")

        return [float(value) for value in values]
