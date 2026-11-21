
"""Two-stage Hybrid V1 training controller.

Sample decisions: label + memory operation (6 actions).
Batch decisions: compute tier (4 actions).

DISCARD means not retaining the incoming candidate. It does not
delete an existing memory record.
"""

import math
import numpy as np


class HybridTrainingController:

    def __init__(
        self,
        sample_agent,
        compute_agent,
        action_manager,
        memory_manager,
        resource_manager,
        batch_trainer,
        label_provider,
        sample_state_builder,
        compute_state_builder,
        sample_reward_fn,
        validation_utility_fn,
        compute_reward_fn,
        compute_costs,
        compute_updates,
        replacement_selector=None,
        batch_size=10,
        label_cost=1,
        base_seed=42,
    ):

        if sample_agent.action_dim != 6:
            raise ValueError("Sample agent must have 6 actions")

        if compute_agent.action_dim != 4:
            raise ValueError("Compute agent must have 4 actions")

        if not callable(label_provider):
            raise ValueError("label_provider must be callable")

        if not callable(sample_state_builder):
            raise ValueError("sample_state_builder must be callable")

        if not callable(compute_state_builder):
            raise ValueError("compute_state_builder must be callable")

        if not callable(sample_reward_fn):
            raise ValueError("sample_reward_fn must be callable")

        if not callable(validation_utility_fn):
            raise ValueError("validation_utility_fn must be callable")

        if not callable(compute_reward_fn):
            raise ValueError("compute_reward_fn must be callable")

        if replacement_selector is not None and not callable(
            replacement_selector
        ):
            raise ValueError("replacement_selector must be callable")

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Invalid batch size")

        if type(label_cost) is not int or label_cost <= 0:
            raise ValueError("Invalid label cost")

        if not isinstance(compute_costs, dict):
            raise ValueError("compute_costs must be a dictionary")

        if not isinstance(compute_updates, dict):
            raise ValueError("compute_updates must be a dictionary")

        expected_tiers = {0, 1, 2, 3}

        if set(compute_costs) != expected_tiers:
            raise ValueError("Compute costs must define tiers 0-3")

        if set(compute_updates) != expected_tiers:
            raise ValueError("Compute updates must define tiers 0-3")

        for tier in range(4):

            cost = compute_costs[tier]
            updates = compute_updates[tier]

            if type(cost) is not int or cost < 0:
                raise ValueError("Invalid compute cost")

            if type(updates) is not int or updates < 0:
                raise ValueError("Invalid optimizer update count")

            if tier == 0 and (cost != 0 or updates != 0):
                raise ValueError("Tier 0 must have zero cost and updates")

            if tier > 0 and (cost <= 0 or updates <= 0):
                raise ValueError("Positive tiers need positive values")

            if tier > 0:
                if cost <= compute_costs[tier - 1]:
                    raise ValueError("Compute costs must increase")

                if updates <= compute_updates[tier - 1]:
                    raise ValueError("Optimizer updates must increase")

        memory_capacity = getattr(memory_manager, "capacity", None)
        action_capacity = getattr(action_manager, "memory_capacity", None)

        if memory_capacity != action_capacity:
            raise ValueError("Action manager and memory capacity differ")

        if getattr(action_manager, "label_cost", label_cost) != label_cost:
            raise ValueError("Label cost differs from action manager")

        self.sample_agent = sample_agent
        self.compute_agent = compute_agent
        self.action_manager = action_manager
        self.memory_manager = memory_manager
        self.resource_manager = resource_manager
        self.batch_trainer = batch_trainer

        self.label_provider = label_provider
        self.sample_state_builder = sample_state_builder
        self.compute_state_builder = compute_state_builder

        self.sample_reward_fn = sample_reward_fn
        self.validation_utility_fn = validation_utility_fn
        self.compute_reward_fn = compute_reward_fn

        self.compute_costs = dict(compute_costs)
        self.compute_updates = dict(compute_updates)

        self.replacement_selector = replacement_selector
        self.batch_size = batch_size
        self.label_cost = label_cost
        self.base_seed = base_seed

        self.input_dim = batch_trainer.input_dim
        self.num_classes = batch_trainer.num_classes
        self.memory_capacity = memory_capacity

        self.batch_count = 0
        self.total_samples = 0
        self.closed = False

        self.pending_sample = None
        self.pending_compute = None

    # ---------------------------------------------------------
    # Resource and memory helpers
    # ---------------------------------------------------------

    def _get_resources(self):

        resources = self.resource_manager.get_resources()

        if not isinstance(resources, dict):
            raise ValueError("ResourceManager must return a dictionary")

        # Support the existing ResourceManager interface.
        if "labels" in resources:
            label_budget = resources["labels"]
        else:
            label_budget = resources.get("label_budget")

        if "compute" in resources:
            compute_budget = resources["compute"]
        else:
            compute_budget = resources.get("compute_budget")

        normalized = {
            "label_budget": label_budget,
            "compute_budget": compute_budget,
        }

        for key, value in normalized.items():

            if type(value) is not int or value < 0:
                raise ValueError("Invalid resource value: " + key)

        return normalized


    def _get_memory_records(self):

        records = self.memory_manager.get_memory_records()

        if not isinstance(records, list):
            raise ValueError("MemoryManager must return a list")

        if len(records) > self.memory_capacity:
            raise ValueError("Memory capacity exceeded")

        return records

    @staticmethod
    def _validate_reward(value):

        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.number)
        ):
            raise ValueError("Reward must be numeric")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError("Reward must be finite")

        return value

    def _make_sample_mask(self, resources, records):

        used = len(records)

        mask = np.asarray(
            self.action_manager.get_sample_feasible_mask(
                resources["label_budget"],
                used,
            ),
            dtype=bool,
        )

        if mask.shape != (6,):
            raise ValueError("Sample action mask must contain 6 values")

        # Action indices:
        # 0: no label + DISCARD
        # 1: no label + STORE
        # 2: no label + REPLACE
        # 3: label + DISCARD
        # 4: label + STORE
        # 5: label + REPLACE

        # Unlabelled items cannot enter labelled memory.
        mask[1] = False
        mask[2] = False

        # Replacement is only enabled when memory is full.
        if used < self.memory_capacity:
            mask[5] = False

        # Never replace without an explicit victim selector.
        if self.replacement_selector is None:
            mask[5] = False

        if not mask.any():
            raise RuntimeError("No feasible sample action")

        return mask

    def _make_compute_mask(self, resources, has_training_data):

        mask = np.asarray(
            self.action_manager.get_compute_feasible_mask(
                resources["compute_budget"],
                self.compute_costs,
            ),
            dtype=bool,
        )

        if mask.shape != (4,):
            raise ValueError("Compute action mask must contain 4 values")

        # Without training data, only tier 0 is allowed.
        if not has_training_data:
            mask[1:] = False

        if not mask.any():
            raise RuntimeError("No feasible compute action")

        return mask

    # ---------------------------------------------------------
    # State helpers
    # ---------------------------------------------------------

    @staticmethod
    def _validate_state(state, expected_dim):

        state = np.asarray(state, dtype=np.float32)

        if state.shape != (expected_dim,):
            raise ValueError("State dimension mismatch")

        if not np.isfinite(state).all():
            raise ValueError("State contains invalid values")

        return state

    def _build_sample_state(self, sample, features, batch_index,
                            sample_index):

        records = self._get_memory_records()
        resources = self._get_resources()

        mask = self._make_sample_mask(resources, records)

        context = {
            "sample": dict(sample),
            "features": features.copy(),
            "memory_records": records,
            "memory_used": len(records),
            "resources": resources,
            "sample_index": sample_index,
            "batch_index": batch_index,
            "feasible_mask": mask.copy(),
        }

        state = self.sample_state_builder(context)

        state = self._validate_state(
            state,
            self.sample_agent.state_dim,
        )

        return state, mask, context

    def _build_compute_state(self, context, has_training_data):

        resources = self._get_resources()
        mask = self._make_compute_mask(
            resources,
            has_training_data,
        )

        context["resources"] = resources
        context["memory_records"] = self._get_memory_records()
        context["memory_used"] = len(context["memory_records"])
        context["feasible_mask"] = mask.copy()

        state = self.compute_state_builder(context)

        state = self._validate_state(
            state,
            self.compute_agent.state_dim,
        )

        return state, mask

    # ---------------------------------------------------------
    # Delayed DQN transitions
    # ---------------------------------------------------------

    def _finish_sample_transition(self, next_state=None, next_mask=None,
                                  done=False):

        if self.pending_sample is None:
            return None

        if done:
            next_state = np.zeros(
                self.sample_agent.state_dim,
                dtype=np.float32,
            )

            next_mask = np.zeros(6, dtype=bool)

        pending = self.pending_sample

        self.sample_agent.remember(
            pending["state"],
            pending["action"],
            pending["reward"],
            next_state,
            done,
            next_mask,
        )

        loss = self.sample_agent.train_step()

        self.pending_sample = None

        return loss

    def _finish_compute_transition(self, next_state=None, next_mask=None,
                                   done=False):

        if self.pending_compute is None:
            return None

        if done:
            next_state = np.zeros(
                self.compute_agent.state_dim,
                dtype=np.float32,
            )

            next_mask = np.zeros(4, dtype=bool)

        pending = self.pending_compute

        self.compute_agent.remember(
            pending["state"],
            pending["action"],
            pending["reward"],
            next_state,
            done,
            next_mask,
        )

        loss = self.compute_agent.train_step()

        self.pending_compute = None

        return loss

    # ---------------------------------------------------------
    # Sample-level decisions
    # ---------------------------------------------------------

    def _process_sample(self, sample, batch_index, local_index,
                        epsilon):

        if not isinstance(sample, dict) or "features" not in sample:
            raise ValueError("Each sample needs a features field")

        hidden_fields = {
            "label",
            "target",
            "y",
            "true_label",
            "ground_truth",
        }

        if any(key.lower() in hidden_fields for key in sample):
            raise ValueError(
                "Incoming samples must not expose ground-truth labels"
            )

        features = np.asarray(
            sample["features"],
            dtype=np.float32,
        )

        if features.shape != (self.input_dim,):
            raise ValueError("Invalid sample feature dimensions")

        if not np.isfinite(features).all():
            raise ValueError("Sample contains invalid features")

        sample_index = self.total_samples + 1

        state, mask, state_context = self._build_sample_state(
            sample,
            features,
            batch_index,
            sample_index,
        )

        # Complete the previous sample's transition.
        previous_loss = self._finish_sample_transition(
            state,
            mask,
            done=False,
        )

        action_index = self.sample_agent.select_action(
            state,
            mask,
            epsilon,
        )

        # DQN index 0-5 -> manager ID 1-6.
        action = self.action_manager.get_sample_action(
            action_index + 1
        )

        resources_before = self._get_resources()
        memory_before = self._get_memory_records()

        acquired_label = None
        memory_result = None
        candidate_item = None

        if action["label"] == 1:

            # The label is requested only after the policy selects it.
            acquired_label = self.label_provider(dict(sample))

            if (
                isinstance(acquired_label, bool)
                or not isinstance(acquired_label, (int, np.integer))
                or not 1 <= int(acquired_label) <= self.num_classes
            ):
                raise ValueError("Label provider returned an invalid label")

            acquired_label = int(acquired_label)

            # ResourceManager consumes one label per call.
            for _ in range(self.label_cost):
                self.resource_manager.consume_label()

            candidate_item = {
                "features": features.tolist(),
                "label": acquired_label,
                "source_batch": batch_index,
                "source_row": sample.get("id", sample_index),
            }

            operation = action["memory"]

            if operation == "STORE":

                memory_result = self.memory_manager.store(
                    candidate_item
                )

            elif operation == "REPLACE":

                if self.replacement_selector is None:
                    raise RuntimeError("No replacement selector configured")

                records = self._get_memory_records()

                if len(records) != self.memory_capacity:
                    raise RuntimeError("REPLACE requires full memory")

                victim_index = self.replacement_selector(
                    records,
                    candidate_item,
                )

                if (
                    type(victim_index) is not int
                    or not 0 <= victim_index < len(records)
                ):
                    raise ValueError("Replacement selector returned bad index")

                memory_result = self.memory_manager.replace(
                    victim_index,
                    candidate_item,
                )

            elif operation == "DISCARD":
                # Do not retain this incoming candidate.
                # Do NOT delete an existing memory record.
                memory_result = None

            else:
                raise ValueError("Unknown memory operation")

        else:
            if action["memory"] != "DISCARD":
                raise RuntimeError("Unlabelled memory operation is forbidden")

        resources_after = self._get_resources()
        memory_after = self._get_memory_records()

        reward_context = {
            "sample": dict(sample),
            "features": features.copy(),
            "action": dict(action),
            "label": acquired_label,
            "candidate_item": candidate_item,
            "memory_result": memory_result,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "batch_index": batch_index,
            "sample_index": sample_index,
        }

        reward = self._validate_reward(
            self.sample_reward_fn(reward_context)
        )

        self.pending_sample = {
            "state": state.copy(),
            "action": action_index,
            "reward": reward,
        }

        self.total_samples += 1

        return {
            "action": action,
            "label": acquired_label,
            "memory_result": memory_result,
            "reward": reward,
            "previous_dqn_loss": previous_loss,
        }

    # ---------------------------------------------------------
    # Batch-level compute decision and actual training
    # ---------------------------------------------------------

    def _train_batch(self, current_features, current_labels,
                     replay_snapshot, batch_index, epsilon):

        current_count = len(current_labels)

        current_x = np.asarray(
            current_features,
            dtype=np.float32,
        ).reshape(current_count, self.input_dim)

        current_y = np.asarray(
            current_labels,
            dtype=np.int64,
        )

        has_training_data = (
            current_count > 0 or len(replay_snapshot) > 0
        )

        context = {
            "batch_index": batch_index,
            "batch_size": len(current_features) if current_count else 0,
            "current_features": current_x.copy(),
            "current_labels": current_y.copy(),
            "current_label_count": current_count,
            "replay_snapshot": replay_snapshot,
        }

        # Independent validation utility: evaluate before training.
        utility_before = self._validate_reward(
            self.validation_utility_fn(
                self.batch_trainer.model,
                context,
            )
        )

        context["utility_before"] = utility_before

        state, mask = self._build_compute_state(
            context,
            has_training_data,
        )

        # Complete the previous batch's compute transition.
        previous_loss = self._finish_compute_transition(
            state,
            mask,
            done=False,
        )

        action_index = self.compute_agent.select_action(
            state,
            mask,
            epsilon,
        )

        # DQN index 0-3 -> manager ID 1-4.
        action = self.action_manager.get_compute_action(
            action_index + 1
        )

        tier = action["compute"]
        cost = self.compute_costs[tier]
        updates = self.compute_updates[tier]

        resources_before = self._get_resources()

        if cost:
            self.resource_manager.consume_compute(cost)

        try:

            training_result = self.batch_trainer.train_batch(
                current_features=current_x,
                current_labels=current_y,
                replay_records=replay_snapshot,
                optimizer_updates=updates,
                stage=batch_index,
                seed=self.base_seed + batch_index - 1,
            )

        except Exception:

            # Restore compute budget if training itself fails.
            if cost:
                self.resource_manager.restore_compute(cost)

            raise

        # Evaluate after training using the same independent probe.
        utility_after = self._validate_reward(
            self.validation_utility_fn(
                self.batch_trainer.model,
                context,
            )
        )

        resources_after = self._get_resources()

        reward_context = {
            "batch_index": batch_index,
            "compute_action": dict(action),
            "compute_cost": cost,
            "optimizer_updates": updates,
            "utility_before": utility_before,
            "utility_after": utility_after,
            "utility_delta": utility_after - utility_before,
            "training_result": training_result,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "current_label_count": current_count,
            "replay_count": len(replay_snapshot),
        }

        compute_reward = self._validate_reward(
            self.compute_reward_fn(reward_context)
        )

        self.pending_compute = {
            "state": state.copy(),
            "action": action_index,
            "reward": compute_reward,
        }

        return {
            "action": action,
            "reward": compute_reward,
            "utility_before": utility_before,
            "utility_after": utility_after,
            "training_result": training_result,
            "previous_dqn_loss": previous_loss,
        }

    # ---------------------------------------------------------
    # Public batch interface
    # ---------------------------------------------------------

    def process_batch(
        self,
        samples,
        epsilon_sample=0.1,
        epsilon_compute=0.1,
        final=False,
    ):

        if self.closed:
            raise RuntimeError("Controller is already closed")

        if not 0 <= epsilon_sample <= 1:
            raise ValueError("Invalid sample epsilon")

        if not 0 <= epsilon_compute <= 1:
            raise ValueError("Invalid compute epsilon")

        samples = list(samples)

        if not 1 <= len(samples) <= self.batch_size:
            raise ValueError("Invalid batch length")

        if len(samples) < self.batch_size and not final:
            raise ValueError(
                "A partial batch must be marked final=True"
            )

        batch_index = self.batch_count + 1

        # Snapshot memory BEFORE processing this batch.
        # Newly retained samples become replay candidates next batch.
        replay_snapshot = self._get_memory_records()

        current_features = []
        current_labels = []
        sample_results = []
        sample_losses = []

        for local_index, sample in enumerate(samples):

            result = self._process_sample(
                sample,
                batch_index,
                local_index,
                epsilon_sample,
            )

            sample_results.append(result)

            if result["previous_dqn_loss"] is not None:
                sample_losses.append(result["previous_dqn_loss"])

            if result["label"] is not None:

                current_features.append(
                    np.asarray(sample["features"], dtype=np.float32)
                )

                current_labels.append(result["label"])

        compute_result = self._train_batch(
            current_features,
            current_labels,
            replay_snapshot,
            batch_index,
            epsilon_compute,
        )

        self.batch_count += 1

        if compute_result["previous_dqn_loss"] is not None:
            compute_loss = compute_result["previous_dqn_loss"]
        else:
            compute_loss = None

        output = {
            "batch_index": batch_index,
            "samples_processed": len(samples),
            "labels_acquired": len(current_labels),
            "replay_records_used": len(replay_snapshot),
            "sample_results": sample_results,
            "sample_dqn_losses": sample_losses,
            "compute_result": compute_result,
            "compute_dqn_loss": compute_loss,
            "resources": self._get_resources(),
            "memory_used": len(self._get_memory_records()),
        }

        if final:
            output["terminal_losses"] = self.end_stream()

        return output

    def end_stream(self):

        if self.closed:
            raise RuntimeError("Controller is already closed")

        sample_loss = self._finish_sample_transition(done=True)
        compute_loss = self._finish_compute_transition(done=True)

        self.closed = True

        return {
            "sample_dqn_loss": sample_loss,
            "compute_dqn_loss": compute_loss,
        }
