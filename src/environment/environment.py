import copy
from config import COMPUTE_COSTS, LABEL_COST, MEMORY_OPERATION_COSTS
import math

class Environment:


    def __init__(
        self,
        feature_extractor,
        state_builder,
        state_encoder,
        memory_manager,
        action_manager,
        resource_manager=None,
        memory_utility_pipeline=None,
        reward_calculator=None,
        prediction_provider=None

    ):
        self.feature_extractor = feature_extractor
        self.state_builder = state_builder
        self.state_encoder = state_encoder
        self.memory_manager = memory_manager
        self.action_manager = action_manager
        self.resource_manager = resource_manager
        self.memory_utility_pipeline = memory_utility_pipeline
        self.reward_calculator = reward_calculator
        self.prediction_provider = prediction_provider


    def get_current_resources(self):
        if self.resource_manager is None:
            raise ValueError("ResourceManager is not connected")

        return self.resource_manager.get_resources()

    def acquire_label(self, sample_id, label_provider):

        if sample_id is None or label_provider is None:
            raise ValueError("Sample ID and label provider are required")

        if hasattr(label_provider, "get_label"):
            label = label_provider.get_label(sample_id)
        elif callable(label_provider):
            label = label_provider(sample_id)
        else:
            raise ValueError("Invalid label provider")

        self.resource_manager.consume_label()

        return label

    def predict_observation(self, observation):

        if self.prediction_provider is None:
            raise ValueError("PredictionProvider is not connected")

        return self.prediction_provider.predict(observation)



    def build_agent_state(
        self,
        observation,
        prediction,
        uncertainty,
        novelty,
        resources,
        history,
        extracted_features=None,
    ):
        if extracted_features is None:
            features = self.feature_extractor.extract(observation)
        else:
            features = extracted_features

        memory_status = self.memory_manager.get_memory_status()

        structured_state = self.state_builder.build_state(
            features=features,
            prediction=prediction,
            uncertainty=uncertainty,
            novelty=novelty,
            memory_status=memory_status,
            resources=resources,
            history=history,
        )

        return self.state_encoder.encode(structured_state)

    def build_agent_state_with_prediction(
    self, observation, novelty, resources, history
    ):
        features = self.feature_extractor.extract(observation)

        result = self.predict_observation(features)

        return self.build_agent_state(
            observation=observation,
            prediction=result["prediction"],
            uncertainty=result["uncertainty"],
            novelty=novelty,
            resources=resources,
            history=history,
            extracted_features=features,
        )

    def build_action_mask(self):

        if self.resource_manager is None:
            raise ValueError("ResourceManager is not connected")

        resources = self.resource_manager.get_resources()
        memory_status = self.memory_manager.get_memory_status()

        feasible_actions = self.action_manager.get_feasible_actions(
            label_budget=resources["labels"],
            compute_budget=resources["compute"],
            memory_used=memory_status["used"],
            memory_capacity=memory_status["capacity"],
            compute_costs=COMPUTE_COSTS,
            label_cost=LABEL_COST,
        )

        return self.action_manager.get_feasible_mask(feasible_actions)

    def consume_compute(self, compute_tier, compute_costs):
        if self.resource_manager is None:
            raise ValueError("ResourceManager is not connected")

        if not isinstance(compute_costs, dict):
            raise ValueError("Compute costs must be a dictionary")

        if compute_tier not in compute_costs:
            raise ValueError("Invalid compute tier")

        cost = compute_costs[compute_tier]

        return self.resource_manager.consume_compute(cost)

    def evaluate_memory_candidate(
        self,
        candidate_record,
        memory_records,
        validation_data,
        evaluation_data,
        future_tasks,
        current_time
    ):

        if self.memory_utility_pipeline is None:
            raise ValueError(
                "MemoryUtilityPipeline is not connected"
            )

        return self.memory_utility_pipeline.score_candidate(
            candidate_record=candidate_record,
            memory_records=memory_records,
            validation_data=validation_data,
            evaluation_data=evaluation_data,
            future_tasks=future_tasks,
            current_time=current_time
        )

    def decode_agent_action(self, network_index, feasible_mask):

        # Validate and convert network index into action ID
        action_id = self.action_manager.get_action_id(network_index)

        # Validate the feasible action mask
        if not isinstance(feasible_mask, (list, tuple)):
            raise ValueError("Feasible mask must be a list or tuple")

        if len(feasible_mask) != 24:
            raise ValueError("Feasible mask must contain 24 elements")

        if any(type(value) is not bool for value in feasible_mask):
            raise ValueError("Feasible mask must contain Boolean values")

        # Reject actions that are not feasible
        if not feasible_mask[network_index]:
            raise ValueError("Selected action is not feasible")

        # Retrieve the corresponding action
        return self.action_manager.get_action(action_id)

    def calculate_transition_reward(
        self,
        performance_before,
        performance_after,
        labels_used,
        compute_used,
        initial_label_budget,
        initial_compute_budget,
        previous_task_score_before=None,
        previous_task_score_after=None
    ):

        if self.reward_calculator is None:
            raise ValueError("RewardCalculator is not connected")

        return self.reward_calculator.calculate(
            performance_before=performance_before,
            performance_after=performance_after,
            labels_used=labels_used,
            compute_used=compute_used,
            initial_label_budget=initial_label_budget,
            initial_compute_budget=initial_compute_budget,
            previous_task_score_before=previous_task_score_before,
            previous_task_score_after=previous_task_score_after
        )

    def calculate_v1_reward(self, utility_before, utility_after, action):
        """Rt = (U_{t+1} - U_t) - (λL cL ℓ + λM cM(μ) + λC CC(c))."""
        if self.reward_calculator is None:
            raise ValueError("RewardCalculator is not connected")

        if not isinstance(action, dict):
            raise ValueError("Action must be a dictionary")

        expected_action = self.action_manager.get_action(action.get("id"))
        if action != expected_action:
            raise ValueError("Invalid action details")

        compute_tier = action["compute"]
        memory_operation = action["memory"]

        return self.reward_calculator.calculate_v1(
            utility_before=utility_before,
            utility_after=utility_after,
            label_action=action["label"],
            memory_operation_cost=MEMORY_OPERATION_COSTS[memory_operation],
            compute_cost=COMPUTE_COSTS[compute_tier],
            label_cost=LABEL_COST,
        )



    def store_dqn_experience(
        self,
        replay_buffer,
        state,
        action_index,
        reward_result,
        next_state,
        done,
        next_feasible_mask
    ):

        if replay_buffer is None:
            raise ValueError("ReplayBuffer is not connected")

        if (
            not isinstance(action_index, int)
            or isinstance(action_index, bool)
            or not 0 <= action_index < 24
        ):
            raise ValueError("Action index must be between 0 and 23")

        if not isinstance(reward_result, dict):
            raise ValueError("Reward result must be a dictionary")

        if "reward" not in reward_result:
            raise ValueError("Reward result must contain reward")

        reward = reward_result["reward"]

        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
        ):
            raise ValueError("Reward must be a finite number")

        if type(done) is not bool:
            raise ValueError("Done must be Boolean")

        if not isinstance(next_feasible_mask, (list, tuple)):
            raise ValueError("Next feasible mask must be a list or tuple")

        if len(next_feasible_mask) != 24:
            raise ValueError("Next feasible mask must contain 24 elements")

        if any(type(value) is not bool for value in next_feasible_mask):
            raise ValueError("Mask values must be Boolean")

        if not done and not any(next_feasible_mask):
            raise ValueError(
                "Non-terminal state must have a feasible next action"
            )

        if len(state) != len(next_state):
            raise ValueError("State dimensions must match")

        replay_buffer.add(
            state=copy.deepcopy(state),
            action=action_index,
            reward=float(reward),
            next_state=copy.deepcopy(next_state),
            done=done,
            next_feasible_mask=list(next_feasible_mask)
        )

        return len(replay_buffer)



    def execute_agent_action(
        self,
        action,
        item=None,
        index=None,
        sample_id=None,
        label_provider=None
    ):
        if not isinstance(action, dict):
            raise ValueError("Action must be a dictionary")

        # Validate action against ActionManager
        expected_action = self.action_manager.get_action(
            action.get("id")
        )

        if action != expected_action:
            raise ValueError("Invalid action details")

        if self.resource_manager is None:
            raise ValueError("ResourceManager is not connected")

        label_choice = action["label"]
        operation = action["memory"]
        compute_tier = action["compute"]

        if type(label_choice) is not int or label_choice not in (0, 1):
            raise ValueError("Invalid label choice")

        if (
            type(compute_tier) is not int
            or compute_tier not in COMPUTE_COSTS
        ):
            raise ValueError("Invalid compute tier")

        resources = self.resource_manager.get_resources()
        memory_status = self.memory_manager.get_memory_status()
        memory_used = memory_status["used"]
        memory_capacity = memory_status["capacity"]

        feasible_actions = self.action_manager.get_feasible_actions(
            label_budget=resources["labels"],
            compute_budget=resources["compute"],
            memory_used=memory_used,
            memory_capacity=memory_capacity,
            compute_costs=COMPUTE_COSTS,
            label_cost=LABEL_COST,
        )
        if action["id"] not in {candidate["id"] for candidate in feasible_actions}:
            raise ValueError("Selected action is not feasible")

        if operation in ("STORE", "REPLACE") and item is None:
            raise ValueError("Item is required")

        # REPLACE uses a fixed victim rule. The policy does not choose it.
        if operation == "REPLACE" and index is None:
            index = 0

        if operation == "REPLACE" and (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < memory_used
        ):
            raise ValueError("Invalid memory index")

        # Prepare item without modifying original data
        prepared_item = copy.deepcopy(item) if item is not None else None

        # Validate label requirements before resource consumption
        if label_choice == 1:
            if sample_id is None or label_provider is None:
                raise ValueError(
                    "Sample ID and label provider are required"
                )

            if operation in ("STORE", "REPLACE"):
                if not isinstance(prepared_item, dict):
                    raise ValueError("Item must be a dictionary")

        acquired_label = None

        # Save resource budgets for rollback
        label_budget_before = self.resource_manager.label_budget
        compute_budget_before = self.resource_manager.compute_budget

        try:
            # Step 1: Acquire label if selected
            if label_choice == 1:
                acquired_label = self.acquire_label(
                    sample_id,
                    label_provider
                )

                if operation in ("STORE", "REPLACE"):
                    prepared_item["label"] = acquired_label

            # Step 2: Consume compute before changing memory
            remaining_compute = self.consume_compute(
                compute_tier,
                COMPUTE_COSTS
            )

            # DISCARD drops the current item and leaves Mt unchanged.
            if operation == "DISCARD":
                memory_result = None
            else:
                memory_result = self.execute_memory_operation(
                    operation=operation,
                    item=prepared_item,
                    index=index
                )

        except Exception:
            # Restore any resources consumed by this failed action
            labels_consumed = (
                label_budget_before - self.resource_manager.label_budget
            )

            compute_consumed = (
                compute_budget_before - self.resource_manager.compute_budget
            )

            if compute_consumed > 0:
                self.resource_manager.restore_compute(compute_consumed)

            for _ in range(labels_consumed):
                self.resource_manager.restore_label()

            raise

        return {
            "action_id": action["id"],
            "memory_operation": operation,
            "memory_result": memory_result,
            "acquired_label": acquired_label,
            "remaining_compute": remaining_compute
        }

    def execute_memory_operation(
        self,
        operation,
        item=None,
        index=None
    ):
        if not isinstance(operation, str):
            raise ValueError("Operation must be a string")

        if operation == "STORE":
            if item is None:
                raise ValueError("Item is required for STORE")

            return self.memory_manager.store(item)

        elif operation == "DISCARD":
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("Valid index is required for DISCARD")

            return self.memory_manager.discard(index)

        elif operation == "REPLACE":
            if item is None:
                raise ValueError("Item is required for REPLACE")

            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("Valid index is required for REPLACE")

            return self.memory_manager.replace(index, item)

        else:
            raise ValueError("Invalid memory operation")
