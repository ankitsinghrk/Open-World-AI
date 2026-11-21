import unittest
from unittest.mock import Mock

from src.prediction_provider import PredictionProvider
from src.environment.environment import Environment


class TestRealPredictionStateIntegration(unittest.TestCase):

    def test_real_checkpoint_to_encoded_state(self):
        provider = PredictionProvider(
            "experiments/results/static_baseline_100_seed42/model_and_scaler.pt"
        )

        env = Environment.__new__(Environment)
        env.prediction_provider = provider
        env.feature_extractor = Mock()
        env.feature_extractor.extract.return_value = [0.1] * 128
        env.memory_manager = Mock()
        env.memory_manager.get_memory_status.return_value = {
            "memory_used": 2,
            "memory_capacity": 10
        }
        env.state_builder = Mock()
        env.state_builder.build_state.return_value = {"test": "state"}
        env.state_encoder = Mock()
        env.state_encoder.encode.return_value = "encoded_state"

        observation = ((provider.minimum + provider.maximum) / 2).tolist()

        result = env.build_agent_state_with_prediction(
            observation=observation,
            novelty=0.2,
            resources={"memory": 10},
            history=[]
        )

        self.assertEqual(result, "encoded_state")

        state_args = env.state_builder.build_state.call_args.kwargs
        self.assertIn(state_args["prediction"], range(1, 7))
        self.assertGreaterEqual(state_args["uncertainty"], 0.0)
        self.assertLessEqual(state_args["uncertainty"], 1.0)
        self.assertEqual(state_args["novelty"], 0.2)

        env.state_encoder.encode.assert_called_once_with({"test": "state"})


if __name__ == "__main__":
    unittest.main()
