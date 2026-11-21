
import unittest
from unittest.mock import Mock

from src.environment.environment import Environment


class TestEnvironmentPrediction(unittest.TestCase):

    def setUp(self):
        self.env = Environment.__new__(Environment)
        self.env.prediction_provider = Mock()

        self.env.feature_extractor = Mock()
        self.env.feature_extractor.extract.return_value = [1.0] * 128

    def test_predict_observation(self):
        expected = {
            "prediction": 4,
            "uncertainty": 0.5,
            "probabilities": [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]
        }

        self.env.prediction_provider.predict.return_value = expected

        result = self.env.predict_observation([1.0] * 128)

        self.assertEqual(result, expected)

    def test_build_state_receives_prediction(self):
        self.env.prediction_provider.predict.return_value = {
            "prediction": 3,
            "uncertainty": 0.4
        }

        self.env.build_agent_state = Mock(return_value="encoded_state")

        result = self.env.build_agent_state_with_prediction(
            observation=[1.0] * 128,
            novelty=0.2,
            resources={"memory": 10},
            history=[]
        )

        self.assertEqual(result, "encoded_state")

        self.env.prediction_provider.predict.assert_called_once_with(
            [1.0] * 128
        )

        self.env.build_agent_state.assert_called_once_with(
            observation=[1.0] * 128,
            prediction=3,
            uncertainty=0.4,
            novelty=0.2,
            resources={"memory": 10},
            history=[],
            extracted_features=[1.0] * 128
        )

    def test_missing_prediction_provider(self):
        self.env.prediction_provider = None

        with self.assertRaisesRegex(
            ValueError, "PredictionProvider is not connected"
        ):
            self.env.predict_observation([1.0] * 128)


if __name__ == "__main__":
    unittest.main()