import torch
import copy
import random
import math
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim=24, hidden_dim=128):
        super().__init__()

        if state_dim <= 0:
            raise ValueError("State dimension must be positive")

        if action_dim != 24:
            raise ValueError("Action dimension must be 24")

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

    @staticmethod
    def mask_q_values(q_values, feasible_mask):
        if q_values.shape != feasible_mask.shape:
            raise ValueError("Q-values and mask must have the same shape")

        if feasible_mask.dtype != torch.bool:
            raise ValueError("Feasible mask must contain Boolean values")

        if not torch.any(feasible_mask):
            raise ValueError("At least one action must be feasible")

        masked_q_values = q_values.clone()
        masked_q_values[~feasible_mask] = float("-inf")

        return masked_q_values



class DQNNetworks:

    def __init__(self, state_dim, action_dim=24, hidden_dim=128):

        self.online_network = QNetwork(
            state_dim, action_dim, hidden_dim
        )

        self.target_network = copy.deepcopy(self.online_network)
        self.optimizer = torch.optim.Adam(
    self.online_network.parameters(),
    lr=0.001
)

        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )


    def select_action(self, state, feasible_mask, epsilon=0.1):

        # Convert state into a PyTorch tensor
        state = torch.as_tensor(state, dtype=torch.float32)

        # Convert feasible mask into a PyTorch tensor
        feasible_mask = torch.as_tensor(feasible_mask)

        if feasible_mask.shape != (24,):
            raise ValueError("Feasible mask must have 24 elements")

        if feasible_mask.dtype != torch.bool:
            raise ValueError("Feasible mask must contain Boolean values")

        if not torch.any(feasible_mask):
            raise ValueError("At least one action must be feasible")

        if state.ndim != 1:
            raise ValueError("State must be a one-dimensional vector")

        if state.shape[0] != self.online_network.network[0].in_features:
            raise ValueError("State dimension does not match network input")

        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
            raise ValueError("Invalid epsilon value")

        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        feasible_actions = torch.where(feasible_mask)[0].tolist()

        # Exploration
        if random.random() < epsilon:
            return random.choice(feasible_actions)

        # Exploitation
        with torch.no_grad():

            q_values = self.online_network(state)

            masked_q_values = QNetwork.mask_q_values(
                q_values,
                feasible_mask
            )

            action = torch.argmax(masked_q_values).item()

        return action





    def train_step(self, batch, gamma=0.99):

        # Validate discount factor
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not math.isfinite(gamma)
            or not 0 <= gamma <= 1
        ):
            raise ValueError("Gamma must be between 0 and 1")

        if not isinstance(batch, (list, tuple)) or len(batch) == 0:
            raise ValueError("Training batch cannot be empty")

        expected_state_dim = (
            self.online_network.network[0].in_features
        )

        states_list = []
        next_states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        masks_list = []

        # Validate every experience
        for experience in batch:

            if not isinstance(experience, (list, tuple)) or len(experience) != 6:
                raise ValueError(
                    "Each experience must contain 6 elements"
                )

            state, action, reward, next_state, done, next_mask = experience

            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < 24
            ):
                raise ValueError("Action must be between 0 and 23")

            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise ValueError("Reward must be a finite number")

            if type(done) is not bool:
                raise ValueError("Done must be Boolean")

            try:
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32
                )
                next_state_tensor = torch.as_tensor(
                    next_state, dtype=torch.float32
                )
                mask_tensor = torch.as_tensor(next_mask)
            except (TypeError, ValueError, RuntimeError):
                raise ValueError("Invalid experience data")

            if state_tensor.shape != (expected_state_dim,):
                raise ValueError("Invalid state dimension")

            if next_state_tensor.shape != (expected_state_dim,):
                raise ValueError("Invalid next-state dimension")

            if (
                not torch.isfinite(state_tensor).all()
                or not torch.isfinite(next_state_tensor).all()
            ):
                raise ValueError("States must contain finite values")

            if mask_tensor.shape != (24,):
                raise ValueError("Next-state mask must have 24 elements")

            if mask_tensor.dtype != torch.bool:
                raise ValueError("Next-state mask must be Boolean")

            if not done and not torch.any(mask_tensor):
                raise ValueError(
                    "Non-terminal state must have a feasible next action"
                )

            states_list.append(state_tensor)
            next_states_list.append(next_state_tensor)
            actions_list.append(action)
            rewards_list.append(float(reward))
            dones_list.append(done)
            masks_list.append(mask_tensor)

        # Convert validated experiences into tensors
        states = torch.stack(states_list)
        next_states = torch.stack(next_states_list)
        actions = torch.tensor(actions_list, dtype=torch.long)
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        dones = torch.tensor(dones_list, dtype=torch.bool)
        next_masks = torch.stack(masks_list)

        # Calculate current Q-values
        current_q_values = self.online_network(states)

        selected_q_values = current_q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Calculate target Q-values
        with torch.no_grad():

            next_q_values = self.target_network(next_states)

            masked_next_q_values = next_q_values.masked_fill(
                ~next_masks, float("-inf")
            )

            max_next_q_values = masked_next_q_values.max(
                dim=1
            ).values

            # Terminal states do not bootstrap
            max_next_q_values = torch.where(
                dones,
                torch.zeros_like(max_next_q_values),
                max_next_q_values
            )

            target_q_values = (
                rewards
                + gamma * (~dones).float() * max_next_q_values
            )

        # Calculate loss
        loss = nn.functional.smooth_l1_loss(
            selected_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()
