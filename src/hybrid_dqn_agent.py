
"""Independent DQN agents for the two-stage Hybrid V1 architecture."""

import random
from collections import deque

import torch
import torch.nn as nn


class HybridQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=128):

        super().__init__()

        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("Invalid state dimension")

        if action_dim not in (4, 6):
            raise ValueError("Hybrid action dimension must be 4 or 6")

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class HybridDQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        replay_capacity=5000,
        batch_size=32,
        target_update_frequency=100,
    ):

        if action_dim not in (4, 6):
            raise ValueError("Action dimension must be 4 or 6")

        if not 0 < gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")

        if replay_capacity <= 0 or batch_size <= 0:
            raise ValueError("Invalid replay settings")

        if target_update_frequency <= 0:
            raise ValueError("Invalid target update frequency")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.online_network = HybridQNetwork(
            state_dim,
            action_dim,
        )

        self.target_network = HybridQNetwork(
            state_dim,
            action_dim,
        )

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.replay_buffer = deque(maxlen=replay_capacity)
        self.training_steps = 0

    def select_action(self, state, feasible_mask, epsilon=0.1):

        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        )

        mask_tensor = torch.as_tensor(feasible_mask)

        if state_tensor.shape != (self.state_dim,):
            raise ValueError("Invalid state dimension")

        if not torch.isfinite(state_tensor).all():
            raise ValueError("State must contain finite values")

        if mask_tensor.shape != (self.action_dim,):
            raise ValueError("Invalid feasible mask dimension")

        if mask_tensor.dtype != torch.bool:
            raise ValueError("Feasible mask must be Boolean")

        feasible_indices = torch.where(mask_tensor)[0].tolist()

        if not feasible_indices:
            raise ValueError("No feasible actions available")

        if random.random() < epsilon:
            return random.choice(feasible_indices)

        with torch.no_grad():
            q_values = self.online_network(state_tensor)

            q_values = q_values.masked_fill(
                ~mask_tensor,
                float("-inf"),
            )

            return int(torch.argmax(q_values).item())

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_mask,
    ):

        if type(action) is not int or not 0 <= action < self.action_dim:
            raise ValueError("Invalid action index")

        if type(done) is not bool:
            raise ValueError("Done must be Boolean")

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).clone()

        next_state_tensor = torch.as_tensor(
            next_state,
            dtype=torch.float32,
        ).clone()

        mask_tensor = torch.as_tensor(next_mask).clone()

        if state_tensor.shape != (self.state_dim,):
            raise ValueError("Invalid state dimension")

        if next_state_tensor.shape != (self.state_dim,):
            raise ValueError("Invalid next-state dimension")

        if not torch.isfinite(state_tensor).all():
            raise ValueError("Invalid state values")

        if not torch.isfinite(next_state_tensor).all():
            raise ValueError("Invalid next-state values")

        if mask_tensor.shape != (self.action_dim,):
            raise ValueError("Invalid next mask dimension")

        if mask_tensor.dtype != torch.bool:
            raise ValueError("Next mask must be Boolean")

        if not done and not torch.any(mask_tensor):
            raise ValueError("Non-terminal state needs a feasible action")

        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("Invalid reward")

        if not torch.isfinite(torch.tensor(float(reward))):
            raise ValueError("Reward must be finite")

        self.replay_buffer.append((
            state_tensor,
            action,
            float(reward),
            next_state_tensor,
            done,
            mask_tensor,
        ))

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(
            self.replay_buffer,
            self.batch_size,
        )

        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor(
            [item[1] for item in batch],
            dtype=torch.long,
        )
        rewards = torch.tensor(
            [item[2] for item in batch],
            dtype=torch.float32,
        )
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor(
            [item[4] for item in batch],
            dtype=torch.bool,
        )
        next_masks = torch.stack([item[5] for item in batch])

        current_q = self.online_network(states)

        selected_q = current_q.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target_network(next_states)

            next_q = next_q.masked_fill(
                ~next_masks,
                float("-inf"),
            )

            max_next_q = next_q.max(dim=1).values

            max_next_q = torch.where(
                dones,
                torch.zeros_like(max_next_q),
                max_next_q,
            )

            targets = rewards + self.gamma * (~dones).float() * max_next_q

        loss = nn.functional.smooth_l1_loss(
            selected_q,
            targets,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return float(loss.item())
