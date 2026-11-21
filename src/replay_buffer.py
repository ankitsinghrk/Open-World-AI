

import random
import copy
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):

        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise ValueError("Capacity must be a positive integer")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_feasible_mask):

        experience = (
            state,
            action,
            reward,
            next_state,
            done,
            next_feasible_mask
        )

        # Protect stored experience from external modifications.
        self.buffer.append(copy.deepcopy(experience))

    def sample(self, batch_size):

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("Batch size must be a positive integer")

        if batch_size > len(self.buffer):
            raise ValueError("Not enough experiences in buffer")

        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)
