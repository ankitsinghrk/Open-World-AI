
"""Coordinate DQN action selection, execution and learning."""

class DQNTrainingLoop:

    def __init__(
        self,
        agent,
        environment,
        replay_buffer,
        trainer
    ):

        components = {
            "Agent": (agent, "select_action"),
            "Environment": (environment, "execute_agent_action"),
            "Replay buffer": (replay_buffer, "__len__"),
            "Trainer": (trainer, "train")
        }

        for name, (component, method) in components.items():
            if not callable(getattr(component, method, None)):
                raise ValueError(
                    f"{name} must provide {method}"
                )

        if getattr(trainer, "agent", None) is not agent:
            raise ValueError("Trainer must use the same DQN agent")

        if getattr(trainer, "replay_buffer", None) is not replay_buffer:
            raise ValueError("Trainer must use the same replay buffer")

        self.agent = agent
        self.environment = environment
        self.replay_buffer = replay_buffer
        self.trainer = trainer

    def select_action(self, state, feasible_mask, epsilon=0.1):

        action_index = self.agent.select_action(
            state,
            feasible_mask,
            epsilon
        )

        action = self.environment.decode_agent_action(
            action_index,
            feasible_mask
        )

        return {
            "action_index": action_index,
            "action": action
        }

    def execute_action(self, selected_action, **action_kwargs):

        if not isinstance(selected_action, dict):
            raise ValueError("Selected action must be a dictionary")

        if set(selected_action) != {"action_index", "action"}:
            raise ValueError("Invalid selected action structure")

        action_index = selected_action["action_index"]

        if (
            not isinstance(action_index, int)
            or isinstance(action_index, bool)
            or not 0 <= action_index < 24
        ):
            raise ValueError("Invalid action index")

        action = selected_action["action"]

        if not isinstance(action, dict):
            raise ValueError("Action must be a dictionary")

        if action.get("id") != action_index + 1:
            raise ValueError("Action ID does not match network index")

        return self.environment.execute_agent_action(
            action=action,
            **action_kwargs
        )

    def record_transition(
        self,
        state,
        action_index,
        reward_inputs,
        next_state,
        done,
        next_feasible_mask
    ):

        if not isinstance(reward_inputs, dict):
            raise ValueError("Reward inputs must be a dictionary")

        reward_result = self.environment.calculate_transition_reward(
            **reward_inputs
        )

        self.environment.store_dqn_experience(
            replay_buffer=self.replay_buffer,
            state=state,
            action_index=action_index,
            reward_result=reward_result,
            next_state=next_state,
            done=done,
            next_feasible_mask=next_feasible_mask
        )

        loss = self.trainer.train()

        return {
            "reward": reward_result["reward"],
            "replay_buffer_size": len(self.replay_buffer),
            "loss": loss,
            "training_steps": self.trainer.training_steps
        }

    def step(
        self,
        state,
        feasible_mask,
        next_state,
        next_feasible_mask,
        utility_before,
        utility_after,
        done=False,
        epsilon=0.1,
        **action_kwargs,
    ):
        """One causal decision: feasible At, then Rt = ΔUt − Ct, then a DQN update."""
        selected = self.select_action(state, feasible_mask, epsilon)
        execution = self.execute_action(selected, **action_kwargs)
        reward_result = self.environment.calculate_v1_reward(
            utility_before=utility_before,
            utility_after=utility_after,
            action=selected["action"],
        )
        self.environment.store_dqn_experience(
            replay_buffer=self.replay_buffer,
            state=state,
            action_index=selected["action_index"],
            reward_result=reward_result,
            next_state=next_state,
            done=done,
            next_feasible_mask=next_feasible_mask,
        )
        loss = self.trainer.train()

        return {
            "action_index": selected["action_index"],
            "action": selected["action"],
            "execution": execution,
            "reward": reward_result["reward"],
            "reward_result": reward_result,
            "loss": loss,
            "replay_buffer_size": len(self.replay_buffer),
            "training_steps": self.trainer.training_steps,
        }

    def step_with_observations(
        self,
        observe_fn,
        validation_utility_fn,
        apply_action_fn,
        epsilon=0.1,
    ):
        """
        Causal interaction:
        observe -> select -> execute -> apply learning effect
        -> re-observe -> evaluate -> store -> train DQN.

        observe_fn() must return:
            observation, novelty, resources, history
            Optional: action_kwargs, done

        validation_utility_fn(context) must return a scalar in [0, 1].
        apply_action_fn(action, execution, context) must apply the
        selected action's actual effect to the continual learner.
        """

        for name, callback in (
            ("observe_fn", observe_fn),
            ("validation_utility_fn", validation_utility_fn),
            ("apply_action_fn", apply_action_fn),
        ):
            if not callable(callback):
                raise ValueError(f"{name} must be callable")

        # 1. Observe the current environment.
        context = observe_fn()

        if not isinstance(context, dict):
            raise ValueError("Observation context must be a dictionary")

        required = {"observation", "novelty", "resources", "history"}
        if not required.issubset(context):
            raise ValueError("Observation context is missing required fields")

        # 2. Build the current state using a fresh prediction.
        state = self.environment.build_agent_state_with_prediction(
            observation=context["observation"],
            novelty=context["novelty"],
            resources=context["resources"],
            history=context["history"],
        )

        feasible_mask = self.environment.build_action_mask()

        if not any(feasible_mask):
            raise ValueError("No feasible action is available")

        # 3. Measure validation utility before the action.
        utility_before = validation_utility_fn(context)

        # 4. Select and execute the action.
        selected = self.select_action(
            state,
            feasible_mask,
            epsilon,
        )

        action_kwargs = context.get("action_kwargs", {})
        if not isinstance(action_kwargs, dict):
            raise ValueError("action_kwargs must be a dictionary")

        execution = self.execute_action(
            selected,
            **action_kwargs,
        )

        # 5. Apply the actual continual-learning effect.
        apply_action_fn(
            selected["action"],
            execution,
            context,
        )

        # 6. Re-observe AFTER action execution and learning.
        next_context = observe_fn()

        if not isinstance(next_context, dict):
            raise ValueError("Next observation context must be a dictionary")

        if not required.issubset(next_context):
            raise ValueError("Next observation context is incomplete")

        next_state = self.environment.build_agent_state_with_prediction(
            observation=next_context["observation"],
            novelty=next_context["novelty"],
            resources=next_context["resources"],
            history=next_context["history"],
        )

        next_feasible_mask = self.environment.build_action_mask()

        done = next_context.get("done", False)
        if type(done) is not bool:
            raise ValueError("done must be Boolean")

        # 7. Measure utility after the action using the same
        # held-out validation protocol.
        utility_after = validation_utility_fn(next_context)

        # 8. Calculate reward, store transition, and train the DQN.
        reward_result = self.environment.calculate_v1_reward(
            utility_before=utility_before,
            utility_after=utility_after,
            action=selected["action"],
        )

        self.environment.store_dqn_experience(
            replay_buffer=self.replay_buffer,
            state=state,
            action_index=selected["action_index"],
            reward_result=reward_result,
            next_state=next_state,
            done=done,
            next_feasible_mask=next_feasible_mask,
        )

        loss = self.trainer.train()

        return {
            "action_index": selected["action_index"],
            "action": selected["action"],
            "execution": execution,
            "utility_before": utility_before,
            "utility_after": utility_after,
            "reward": reward_result["reward"],
            "reward_result": reward_result,
            "loss": loss,
            "replay_buffer_size": len(self.replay_buffer),
            "training_steps": self.trainer.training_steps,
            "done": done,
        }
