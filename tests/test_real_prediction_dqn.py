
import unittest
import torch

from config import INITIAL_LABEL_BUDGET, INITIAL_COMPUTE_BUDGET
from src.action_manager import ActionManager
from src.dqn_agent import DQNNetworks
from src.environment.environment import Environment
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager
from src.prediction_provider import PredictionProvider
from src.resource_manager import ResourceManager
from src.reward_calculator import RewardCalculator
from src.state_builder import StateBuilder
from src.state_encoder import StateEncoder


class TestRealPredictionDQN(unittest.TestCase):

    def test_real_checkpoint_to_dqn(self):

        checkpoint_path = (
            "experiments/results/static_baseline_100_seed42/"
            "model_and_scaler.pt"
        )

        provider = PredictionProvider(checkpoint_path)

        feature_extractor = FeatureExtractor(
            encoder=lambda observation: observation["features"]
        )

        state_builder = StateBuilder()

        state_encoder = StateEncoder(
            feature_dim=128,
            history_size=5
        )

        memory_manager = MemoryManager(capacity=10)

        resource_manager = ResourceManager(
            label_budget=INITIAL_LABEL_BUDGET,
            compute_budget=INITIAL_COMPUTE_BUDGET
        )

        environment = Environment(
            feature_extractor=feature_extractor,
            state_builder=state_builder,
            state_encoder=state_encoder,
            memory_manager=memory_manager,
            action_manager=ActionManager(),
            resource_manager=resource_manager,
            reward_calculator=RewardCalculator(
                forgetting_weight=0.5,
                label_cost_weight=0.1,
                compute_cost_weight=0.1
            ),
            prediction_provider=provider
        )

        observation = {
            "features": [0.5] * 128
        }

        state = environment.build_agent_state_with_prediction(
            observation=observation,
            novelty=0.2,
            resources=environment.get_current_resources(),
            history=[]
        )

        self.assertEqual(len(state), state_encoder.state_dim)

        self.assertTrue(torch.isfinite(
            torch.tensor(state, dtype=torch.float32)
        ).all())

        agent = DQNNetworks(
            state_dim=state_encoder.state_dim,
            action_dim=24,
            hidden_dim=32
        )

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = agent.online_network(state_tensor)

        self.assertEqual(tuple(q_values.shape), (1, 24))
        self.assertTrue(torch.isfinite(q_values).all())


if __name__ == "__main__":
    unittest.main()
