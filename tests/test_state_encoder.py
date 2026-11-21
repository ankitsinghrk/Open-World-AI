
import unittest

from src.state_encoder import StateEncoder


class TestStateEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = StateEncoder(feature_dim=3, history_size=2)

    def valid_state(self):
        return {
            "features": [0.1, 0.2, 0.3],
            "prediction": 1,
            "uncertainty": 0.4,
            "novelty": 0.5,
            "memory_remaining": 0.6,
            "label_remaining": 0.7,
            "compute_remaining": 0.8,
            "history": [0.9, 1.0]
        }

    def test_state_dimension(self):
        self.assertEqual(self.encoder.state_dim, 11)

    def test_valid_state_encoding(self):
        encoded = self.encoder.encode(self.valid_state())

        self.assertEqual(len(encoded), 11)
        self.assertTrue(all(isinstance(x, float) for x in encoded))

    def test_missing_features(self):
        state = self.valid_state()
        state["features"] = None

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_wrong_feature_dimension(self):
        state = self.valid_state()
        state["features"] = [0.1, 0.2]

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_missing_history(self):
        state = self.valid_state()
        state["history"] = None

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_short_history_padding(self):
        state = self.valid_state()
        state["history"] = [0.5]

        encoded = self.encoder.encode(state)

        self.assertEqual(encoded[-2:], [0.5, 0.0])

    def test_non_numeric_value(self):
        state = self.valid_state()
        state["uncertainty"] = "invalid"

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_nan_value(self):
        state = self.valid_state()
        state["novelty"] = float("nan")

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_infinite_value(self):
        state = self.valid_state()
        state["features"][0] = float("inf")

        with self.assertRaises(ValueError):
            self.encoder.encode(state)

    def test_boolean_value(self):
        state = self.valid_state()
        state["prediction"] = True

        with self.assertRaises(ValueError):
            self.encoder.encode(state)


if __name__ == "__main__":
    unittest.main()
