
import math


class DQNTrainer:

    def __init__(
        self,
        agent,
        replay_buffer,
        batch_size=32,
        gamma=0.99,
        target_update_frequency=100
    ):

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("Batch size must be a positive integer")

        if (
            not isinstance(gamma, (int, float))
            or isinstance(gamma, bool)
            or not math.isfinite(gamma)
            or not 0 <= gamma <= 1
        ):
            raise ValueError("Gamma must be between 0 and 1")

        if (
            not isinstance(target_update_frequency, int)
            or isinstance(target_update_frequency, bool)
            or target_update_frequency <= 0
        ):
            raise ValueError(
                "Target update frequency must be a positive integer"
            )

        if not callable(getattr(agent, "train_step", None)):
            raise ValueError("Agent must provide train_step")

        if not callable(getattr(agent, "update_target_network", None)):
            raise ValueError("Agent must provide update_target_network")

        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.training_steps = 0

    def train(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        loss = self.agent.train_step(
            batch,
            gamma=self.gamma
        )

        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise ValueError(
                "Training loss must be finite and non-negative"
            )

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.agent.update_target_network()

        return loss
