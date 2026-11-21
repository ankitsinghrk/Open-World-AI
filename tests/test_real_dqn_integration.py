
import unittest
import torch

from config import INITIAL_LABEL_BUDGET, INITIAL_COMPUTE_BUDGET

from src.action_manager import ActionManager
from src.dqn_agent import DQNNetworks
from src.dqn_trainer import DQNTrainer
from src.dqn_training_loop import DQNTrainingLoop
from src.environment.environment import Environment
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager
from src.replay_buffer import ReplayBuffer
from src.resource_manager import ResourceManager
from src.reward_calculator import RewardCalculator
from src.state_builder import StateBuilder
from src.state_encoder import StateEncoder


class TestRealDQNIntegration(unittest.TestCase):

    def setUp(self):

        # State construction components
        self.feature_extractor = FeatureExtractor(
            encoder=lambda observation: observation["features"]
        )

        self.state_builder = StateBuilder()

        self.state_encoder = StateEncoder(
            feature_dim=4,
            history_size=5
        )

        # Resource and memory components
        self.action_manager = ActionManager()

        self.memory_manager = MemoryManager(capacity=10)

        self.resource_manager = ResourceManager(
            label_budget=INITIAL_LABEL_BUDGET,
            compute_budget=INITIAL_COMPUTE_BUDGET
        )

        # Reward component
        self.reward_calculator = RewardCalculator(
            forgetting_weight=0.5,
            label_cost_weight=0.1,
            compute_cost_weight=0.1
        )

        # Actual Environment
        self.environment = Environment(
            feature_extractor=self.feature_extractor,
            state_builder=self.state_builder,
            state_encoder=self.state_encoder,
            memory_manager=self.memory_manager,
            action_manager=self.action_manager,
            resource_manager=self.resource_manager,
            reward_calculator=self.reward_calculator
        )

        # Actual DQN network
        self.agent = DQNNetworks(
            state_dim=self.state_encoder.state_dim,
            action_dim=24,
            hidden_dim=32
        )

        # Make action selection deterministic for this test.
        # Network index 4 corresponds to action ID 5:
        # label=0, memory=STORE, compute tier=0.
        with torch.no_grad():
            output_layer = self.agent.online_network.network[-1]
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            output_layer.bias[4] = 10.0

        self.agent.update_target_network()

        # Actual Replay Buffer and Trainer
        self.replay_buffer = ReplayBuffer(capacity=10)

        self.trainer = DQNTrainer(
            agent=self.agent,
            replay_buffer=self.replay_buffer,
            batch_size=1,
            gamma=0.99,
            target_update_frequency=1
        )

        # Actual Training Loop
        self.training_loop = DQNTrainingLoop(
            agent=self.agent,
            environment=self.environment,
            replay_buffer=self.replay_buffer,
            trainer=self.trainer
        )

    def test_real_training_pipeline(self):

        # 1. Build initial state
        resources_before = self.environment.get_current_resources()

        state = self.environment.build_agent_state(
            observation={
                "features": [0.1, 0.2, 0.3, 0.4]
            },
            prediction=0.5,
            uncertainty=0.4,
            novelty=0.3,
            resources=resources_before,
            history=[]
        )

        self.assertEqual(
            len(state),
            self.state_encoder.state_dim
        )

        # 2. Build feasible action mask
        feasible_mask = self.environment.build_action_mask()

        self.assertEqual(len(feasible_mask), 24)
        self.assertTrue(feasible_mask[4])

        # 3. Select action using actual DQN
        selected = self.training_loop.select_action(
            state,
            feasible_mask,
            epsilon=0.0
        )

        self.assertEqual(selected["action_index"], 4)
        self.assertEqual(selected["action"]["id"], 5)

        # 4. Execute action in actual Environment
        execution = self.training_loop.execute_action(
            selected,
            item={
                "sample_id": "sample_001",
                "features": [0.1, 0.2, 0.3, 0.4]
            }
        )

        self.assertEqual(execution["action_id"], 5)
        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            1
        )

        # 5. Calculate actual resource consumption
        resources_after = self.environment.get_current_resources()

        labels_used = (
            resources_before["labels"] - resources_after["labels"]
        )

        compute_used = (
            resources_before["compute"] - resources_after["compute"]
        )

        self.assertEqual(labels_used, 0)
        self.assertEqual(compute_used, 0)

        # 6. Build next state and next feasible mask
        next_state = self.environment.build_agent_state(
            observation={
                "features": [0.2, 0.3, 0.4, 0.5]
            },
            prediction=0.6,
            uncertainty=0.35,
            novelty=0.25,
            resources=resources_after,
            history=[0.1]
        )

        next_feasible_mask = self.environment.build_action_mask()

        self.assertTrue(any(next_feasible_mask))

        # 7. Prepare reward inputs
        reward_inputs = {
            "performance_before": 0.50,
            "performance_after": 0.60,
            "labels_used": labels_used,
            "compute_used": compute_used,
            "initial_label_budget": INITIAL_LABEL_BUDGET,
            "initial_compute_budget": INITIAL_COMPUTE_BUDGET
        }

        # 8. Store transition and train actual DQN
        self.training_loop.record_transition(
            state=state,
            action_index=selected["action_index"],
            reward_inputs=reward_inputs,
            next_state=next_state,
            done=False,
            next_feasible_mask=next_feasible_mask
        )

        # 9. Verify replay storage and actual training
        self.assertEqual(len(self.replay_buffer), 1)
        self.assertEqual(self.trainer.training_steps, 1)

        experience = self.replay_buffer.sample(1)[0]

        self.assertEqual(experience[1], 4)
        self.assertFalse(experience[4])
        self.assertEqual(len(experience[0]), self.state_encoder.state_dim)
        self.assertTrue(any(experience[5]))

        # 10. Verify target network synchronization
        for online_parameter, target_parameter in zip(
            self.agent.online_network.parameters(),
            self.agent.target_network.parameters()
        ):
            self.assertTrue(
                torch.equal(online_parameter, target_parameter)
            )


    def test_multiple_real_training_steps(self):

        # Target network should update after every 2 training steps.
        self.trainer.target_update_frequency = 2

        # Build initial state
        resources = self.environment.get_current_resources()

        state = self.environment.build_agent_state(
            observation={"features": [0.1, 0.2, 0.3, 0.4]},
            prediction=0.5,
            uncertainty=0.4,
            novelty=0.3,
            resources=resources,
            history=[]
        )


        initial_target_parameters = [
            parameter.detach().clone()
            for parameter in self.agent.target_network.parameters()
        ]

        synchronized_target_parameters = None

        # Run 3 real environment and training steps
        for step in range(3):

            # Keep action selection deterministic.
            with torch.no_grad():
                output_layer = self.agent.online_network.network[-1]
                output_layer.weight.zero_()
                output_layer.bias.zero_()
                output_layer.bias[4] = 10.0

            # Select action
            feasible_mask = self.environment.build_action_mask()

            selected = self.training_loop.select_action(
                state,
                feasible_mask,
                epsilon=0.0
            )

            self.assertEqual(selected["action_index"], 4)

            # Execute action
            resources_before = self.environment.get_current_resources()

            execution = self.training_loop.execute_action(
                selected,
                item={
                    "sample_id": f"sample_{step}",
                    "features": [0.1, 0.2, 0.3, 0.4]
                }
            )

            self.assertEqual(execution["action_id"], 5)

            resources_after = self.environment.get_current_resources()

            labels_used = (
                resources_before["labels"] - resources_after["labels"]
            )

            compute_used = (
                resources_before["compute"] - resources_after["compute"]
            )

            self.assertEqual(labels_used, 0)
            self.assertEqual(compute_used, 0)

            # Build next state
            next_state = self.environment.build_agent_state(
                observation={
                    "features": [
                        0.2 + step * 0.01,
                        0.3,
                        0.4,
                        0.5
                    ]
                },
                prediction=0.6,
                uncertainty=0.35,
                novelty=0.25,
                resources=resources_after,
                history=[0.1 * (step + 1)]
            )

            next_mask = self.environment.build_action_mask()

            # Calculate reward and train
            reward_inputs = {
                "performance_before": 0.50 + step * 0.01,
                "performance_after": 0.60 + step * 0.01,
                "labels_used": labels_used,
                "compute_used": compute_used,
                "initial_label_budget": INITIAL_LABEL_BUDGET,
                "initial_compute_budget": INITIAL_COMPUTE_BUDGET
            }

            self.training_loop.record_transition(
                state=state,
                action_index=selected["action_index"],
                reward_inputs=reward_inputs,
                next_state=next_state,
                done=False,
                next_feasible_mask=next_mask
            )

            state = next_state

            # Verify target network update schedule
            # Step 1: Target network should remain unchanged.
            if step == 0:
                for target, initial in zip(
                    self.agent.target_network.parameters(),
                    initial_target_parameters
                ):
                    self.assertTrue(torch.equal(target, initial))

            # Step 2: Target network should synchronize.
            if step == 1:
                synchronized_target_parameters = [
                    parameter.detach().clone()
                    for parameter in self.agent.target_network.parameters()
                ]

                for online, target in zip(
                    self.agent.online_network.parameters(),
                    self.agent.target_network.parameters()
                ):
                    self.assertTrue(torch.equal(online, target))

            # Step 3: Target network should not synchronize again.
            if step == 2:
                for target, synchronized in zip(
                    self.agent.target_network.parameters(),
                    synchronized_target_parameters
                ):
                  self.assertTrue(torch.equal(target, synchronized))

        # Final verification
        self.assertEqual(len(self.replay_buffer), 3)
        self.assertEqual(self.trainer.training_steps, 3)

        self.assertEqual(
            self.memory_manager.get_memory_status()["used"],
            3
        )

        self.assertEqual(
            self.environment.get_current_resources(),
            {
                "labels": INITIAL_LABEL_BUDGET,
                "compute": INITIAL_COMPUTE_BUDGET
            }
        )

        # Third training step should not synchronize the target network.


    def test_step_uses_v1_reward(self):
        resources = self.environment.get_current_resources()
        state = self.environment.build_agent_state(
            observation={"features": [0.1, 0.2, 0.3, 0.4]},
            prediction=0.5,
            uncertainty=0.4,
            novelty=0.3,
            resources=resources,
            history=[],
        )
        feasible_mask = self.environment.build_action_mask()

        self.training_loop.execute_action = lambda selected, **kwargs: {
            "action_id": selected["action"]["id"]
        }

        result = self.training_loop.step(
            state=state,
            feasible_mask=feasible_mask,
            next_state=state,
            next_feasible_mask=feasible_mask,
            utility_before=0.40,
            utility_after=0.55,
            done=False,
            epsilon=0.0,
        )

        self.assertEqual(result["action_index"], 4)
        # STORE costs 1 memory-operation unit. Its weight is 0 in this fixture,
        # and tier 0 compute costs 0, so the reward is the utility change.
        self.assertAlmostEqual(result["reward"], 0.15)
        self.assertEqual(len(self.replay_buffer), 1)
        self.assertEqual(self.trainer.training_steps, 1)



    def test_v1_reward_with_real_memory_cost_and_dqn_training(self):
        # 1. Configure a non-zero memory cost weight.
        self.environment.reward_calculator = RewardCalculator(
            forgetting_weight=0.5,
            label_cost_weight=0.1,
            compute_cost_weight=0.1,
            memory_cost_weight=0.5,
        )

        # 2. Build the initial state and feasible mask.
        resources_before = self.environment.get_current_resources()

        state = self.environment.build_agent_state(
            observation={"features": [0.1, 0.2, 0.3, 0.4]},
            prediction=0.5,
            uncertainty=0.4,
            novelty=0.3,
            resources=resources_before,
            history=[],
        )

        feasible_mask = self.environment.build_action_mask()

        # 3. Select an action using the actual DQN.
        selected = self.training_loop.select_action(
            state,
            feasible_mask,
            epsilon=0.0,
        )

        self.assertEqual(selected["action_index"], 4)
        self.assertEqual(selected["action"]["id"], 5)

        # 4. Execute the actual STORE operation.
        execution = self.training_loop.execute_action(
            selected,
            item={
                "sample_id": "v1_memory_test",
                "features": [0.1, 0.2, 0.3, 0.4],
            },
        )

        self.assertEqual(execution["action_id"], 5)
        self.assertEqual(self.memory_manager.get_memory_status()["used"], 1)

        # 5. Build the next state AFTER action execution.
        resources_after = self.environment.get_current_resources()

        next_state = self.environment.build_agent_state(
            observation={"features": [0.2, 0.3, 0.4, 0.5]},
            prediction=0.6,
            uncertainty=0.35,
            novelty=0.25,
            resources=resources_after,
            history=[0.1],
        )

        next_mask = self.environment.build_action_mask()

        # 6. Calculate the actual V1 reward.
        reward_result = self.environment.calculate_v1_reward(
            utility_before=0.40,
            utility_after=0.55,
            action=selected["action"],
        )

        # STORE cost = 1, memory weight = 0.5.
        # Utility change = 0.15.
        # Expected reward = 0.15 - 0.5 = -0.35.
        self.assertAlmostEqual(reward_result["operation_cost"], 0.5)
        self.assertAlmostEqual(reward_result["reward"], -0.35)

        # 7. Store the actual transition and train the DQN.
        self.environment.store_dqn_experience(
            replay_buffer=self.replay_buffer,
            state=state,
            action_index=selected["action_index"],
            reward_result=reward_result,
            next_state=next_state,
            done=False,
            next_feasible_mask=next_mask,
        )

        loss = self.trainer.train()

        # 8. Verify the complete pipeline.
        self.assertEqual(len(self.replay_buffer), 1)
        self.assertEqual(self.trainer.training_steps, 1)

        experience = self.replay_buffer.sample(1)[0]

        self.assertEqual(experience[1], 4)
        self.assertAlmostEqual(experience[2], -0.35)
        self.assertFalse(experience[4])
        self.assertTrue(any(experience[5]))
        self.assertIsNotNone(loss)



if __name__ == "__main__":
    unittest.main()
