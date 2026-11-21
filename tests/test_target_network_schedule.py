
import unittest

from src.dqn_trainer import DQNTrainer


class FakeAgent:

    def __init__(self):
        self.target_updates = 0

    def train_step(self, batch, gamma=0.99):
        return 0.5

    def update_target_network(self):
        self.target_updates += 1


class FakeReplayBuffer:

    def __len__(self):
        return 10

    def sample(self, batch_size):
        return [None] * batch_size


class TestTargetNetworkSchedule(unittest.TestCase):

    def setUp(self):
        self.agent = FakeAgent()
        self.buffer = FakeReplayBuffer()

    def test_target_updates_at_configured_interval(self):

        trainer = DQNTrainer(
            self.agent,
            self.buffer,
            batch_size=2,
            target_update_frequency=2
        )

        trainer.train()
        self.assertEqual(self.agent.target_updates, 0)

        trainer.train()
        self.assertEqual(self.agent.target_updates, 1)

    def test_no_update_before_interval(self):

        trainer = DQNTrainer(
            self.agent,
            self.buffer,
            batch_size=2,
            target_update_frequency=3
        )

        trainer.train()
        trainer.train()

        self.assertEqual(self.agent.target_updates, 0)

    def test_update_every_step_when_frequency_is_one(self):

        trainer = DQNTrainer(
            self.agent,
            self.buffer,
            batch_size=2,
            target_update_frequency=1
        )

        trainer.train()
        trainer.train()

        self.assertEqual(self.agent.target_updates, 2)

    def test_invalid_frequency(self):

        with self.assertRaises(ValueError):
            DQNTrainer(
                self.agent,
                self.buffer,
                target_update_frequency=0
            )

    def test_training_steps_count(self):

        trainer = DQNTrainer(
            self.agent,
            self.buffer,
            batch_size=2,
            target_update_frequency=5
        )

        trainer.train()
        trainer.train()

        self.assertEqual(trainer.training_steps, 2)


if __name__ == "__main__":
    unittest.main()
