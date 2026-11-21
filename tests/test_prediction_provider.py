import os
import tempfile
import unittest

import numpy as np
import torch

from src.prediction_provider import PredictionProvider, SmallMLP


class TestPredictionProvider(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(
            self.temp_dir.name, "test_checkpoint.pt"
        )

        model = SmallMLP()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_min": torch.zeros(128, dtype=torch.float64),
            "scaler_max": torch.ones(128, dtype=torch.float64),
            "class_names": {
                1: "Ethanol",
                2: "Ethylene",
                3: "Ammonia",
                4: "Acetaldehyde",
                5: "Acetone",
                6: "Toluene",
            },
        }

        torch.save(checkpoint, self.checkpoint_path)
        self.provider = PredictionProvider(self.checkpoint_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_checkpoint_loads(self):
        self.assertEqual(len(self.provider.class_names), 6)

    def test_returns_six_probabilities(self):
        result = self.provider.predict(np.full(128, 0.5))
        self.assertEqual(len(result["probabilities"]), 6)

    def test_probabilities_sum_to_one(self):
        result = self.provider.predict(np.full(128, 0.5))
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=9)

    def test_prediction_is_valid_class_id(self):
        result = self.provider.predict(np.full(128, 0.5))
        self.assertIn(result["prediction"], range(1, 7))

    def test_uncertainty_is_normalized(self):
        result = self.provider.predict(np.full(128, 0.5))
        self.assertGreaterEqual(result["uncertainty"], 0.0)
        self.assertLessEqual(result["uncertainty"], 1.0)

    def test_input_is_unchanged(self):
        observation = np.full(128, 0.5)
        original = observation.copy()

        self.provider.predict(observation)

        np.testing.assert_array_equal(observation, original)

    def test_rejects_wrong_feature_count(self):
        with self.assertRaises(ValueError):
            self.provider.predict([0.5] * 127)

    def test_rejects_non_finite_features(self):
        observation = np.full(128, 0.5)
        observation[0] = np.nan

        with self.assertRaises(ValueError):
            self.provider.predict(observation)


if __name__ == "__main__":
    unittest.main()
