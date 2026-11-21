
import unittest
import torch

from src.dqn_agent import QNetwork, DQNNetworks


class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.state_dim = 8
        self.agent = DQNNetworks(
            state_dim=self.state_dim,
            action_dim=24,
            hidden_dim=32
        )

        self.state = torch.randn(self.state_dim)
        self.mask = torch.tensor(
            [True, True, False] + [False] * 21,
            dtype=torch.bool
        )

    def test_q_network_output_shape(self):
        network = QNetwork(self.state_dim)
        output = network(self.state)

        self.assertEqual(output.shape, (24,))

    def test_invalid_action_dimension(self):
        with self.assertRaises(ValueError):
            QNetwork(self.state_dim, action_dim=10)

    def test_mask_q_values(self):
        q_values = torch.tensor([1.0, 5.0, 10.0])
        mask = torch.tensor([True, True, False])

        result = QNetwork.mask_q_values(q_values, mask)

        self.assertEqual(result[0].item(), 1.0)
        self.assertEqual(result[1].item(), 5.0)
        self.assertTrue(torch.isinf(result[2]))

    def test_empty_action_mask(self):
        q_values = torch.randn(24)
        mask = torch.zeros(24, dtype=torch.bool)

        with self.assertRaises(ValueError):
            QNetwork.mask_q_values(q_values, mask)

    def test_select_action_returns_valid_action(self):
        action = self.agent.select_action(
            self.state,
            self.mask,
            epsilon=1.0
        )

        self.assertIn(action, [0, 1])

    def test_select_action_exploration(self):
        for _ in range(20):
            action = self.agent.select_action(
                self.state,
                self.mask,
                epsilon=1.0
            )

            self.assertIn(action, [0, 1])

    def test_select_action_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            self.agent.select_action(
                self.state,
                self.mask,
                epsilon=1.5
            )

    def test_target_network_update(self):
        with torch.no_grad():
            for parameter in self.agent.online_network.parameters():
                parameter.fill_(0.5)

        self.agent.update_target_network()

        for online, target in zip(
            self.agent.online_network.parameters(),
            self.agent.target_network.parameters()
        ):
            self.assertTrue(torch.equal(online, target))

    def test_training_step(self):
        batch = []

        for _ in range(4):
            batch.append((
                torch.randn(self.state_dim).tolist(),
                0,
                0.5,
                torch.randn(self.state_dim).tolist(),
                False,
                [True] + [False] * 23
            ))

        loss = self.agent.train_step(batch)

        self.assertIsInstance(loss, float)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertGreaterEqual(loss, 0)

    def test_empty_training_batch(self):
        with self.assertRaises(ValueError):
            self.agent.train_step([])


if __name__ == "__main__":
    unittest.main()
