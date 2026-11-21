
import unittest

from src.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = ReplayBuffer(capacity=3)

    def add_experience(self, number):
        self.buffer.add(
            state=[number, 0.1],
            action=number,
            reward=number * 0.5,
            next_state=[number + 1, 0.2],
            done=False,
            next_feasible_mask=[True, False]
        )

    def test_initial_buffer_is_empty(self):
        self.assertEqual(len(self.buffer), 0)

    def test_add_experience(self):
        self.add_experience(1)
        self.assertEqual(len(self.buffer), 1)

    def test_buffer_respects_capacity(self):
        for i in range(5):
            self.add_experience(i)

        self.assertEqual(len(self.buffer), 3)

    def test_oldest_experience_is_removed(self):
        for i in range(4):
            self.add_experience(i)

        stored_actions = [
            experience[1] for experience in self.buffer.buffer
        ]

        self.assertEqual(stored_actions, [1, 2, 3])

    def test_sample_returns_requested_number(self):
        for i in range(3):
            self.add_experience(i)

        samples = self.buffer.sample(2)

        self.assertEqual(len(samples), 2)

    def test_sample_contains_only_stored_experiences(self):
        for i in range(3):
            self.add_experience(i)

        samples = self.buffer.sample(2)

        stored_experiences = list(self.buffer.buffer)

        for experience in samples:
            self.assertIn(experience, stored_experiences)

    def test_rejects_zero_capacity(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0)

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=-1)

    def test_rejects_zero_batch_size(self):
        with self.assertRaises(ValueError):
            self.buffer.sample(0)

    def test_rejects_negative_batch_size(self):
        with self.assertRaises(ValueError):
            self.buffer.sample(-1)

    def test_rejects_batch_larger_than_buffer(self):
        with self.assertRaises(ValueError):
            self.buffer.sample(1)



    def test_rejects_boolean_capacity(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=True)

    def test_rejects_float_capacity(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=1.5)

    def test_rejects_boolean_batch_size(self):
        self.add_experience(1)

        with self.assertRaises(ValueError):
            self.buffer.sample(True)

    def test_rejects_float_batch_size(self):
       self.add_experience(1)

       with self.assertRaises(ValueError):
        self.buffer.sample(1.5)

    def test_stored_experience_is_protected_from_external_changes(self):
        state = [1.0, 2.0]
        next_state = [3.0, 4.0]
        mask = [True, False]

        self.buffer.add(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False,
            next_feasible_mask=mask
        )

        # Change the original lists after storing.
        state[0] = 99.0
        next_state[0] = 99.0
        mask[0] = False

        stored = self.buffer.buffer[0]

        self.assertEqual(stored[0], [1.0, 2.0])
        self.assertEqual(stored[3], [3.0, 4.0])
        self.assertEqual(stored[5], [True, False])

    def test_sample_does_not_modify_buffer(self):
        for i in range(3):
            self.add_experience(i)

        original_buffer = list(self.buffer.buffer)

        self.buffer.sample(2)

        self.assertEqual(list(self.buffer.buffer), original_buffer)



if __name__ == "__main__":
    unittest.main()
