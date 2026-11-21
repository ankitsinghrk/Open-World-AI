
import copy
from unittest import result
from unittest import result

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.hybrid_action_manager import HybridActionManager
from src.hybrid_dqn_agent import HybridDQNAgent
from src.hybrid_batch_trainer import HybridBatchTrainer
from src.hybrid_training_controller import HybridTrainingController
from src.memory_manager import MemoryManager
from src.resource_manager import ResourceManager


def set_preferred_action(agent, action_index):
    """Make the network deterministically prefer one action."""

    with torch.no_grad():
        final_layer = agent.online_network.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(-10.0)
        final_layer.bias[action_index] = 10.0


def make_controller(label_budget=10, compute_budget=20):

    torch.manual_seed(42)
    np.random.seed(42)

    memory = MemoryManager(capacity=4)

    resources = ResourceManager(
        label_budget=label_budget,
        compute_budget=compute_budget,
    )

    actions = HybridActionManager(
        memory_capacity=4,
        label_cost=1,
    )

    sample_agent = HybridDQNAgent(
        state_dim=5,
        action_dim=6,
        batch_size=1,
        replay_capacity=100,
    )

    compute_agent = HybridDQNAgent(
        state_dim=4,
        action_dim=4,
        batch_size=1,
        replay_capacity=100,
    )

    # Action index 4 = acquire label + STORE.
    set_preferred_action(sample_agent, 4)

    # Action index 1 = compute tier 1.
    set_preferred_action(compute_agent, 1)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )

    trainer = HybridBatchTrainer(
        model=model,
        input_dim=4,
        num_classes=3,
        current_per_update=2,
        replay_per_update=2,
        warmup_current_per_update=2,
    )

    # External oracle: labels are not included in incoming samples.
    oracle = {
    1: 1,
    2: 2,
    3: 3,
    4: 1,
    5: 2,
}

    def label_provider(sample):
        return oracle[sample["id"]]

    def sample_state_builder(context):
        resources_now = context["resources"]

        return [
            context["memory_used"],
            resources_now["label_budget"],
            resources_now["compute_budget"],
            float(context["features"].mean()),
            context["sample_index"],
        ]

    def compute_state_builder(context):
        resources_now = context["resources"]

        return [
            context["current_label_count"],
            len(context["replay_snapshot"]),
            resources_now["label_budget"],
            resources_now["compute_budget"],
        ]

    def sample_reward(context):
        if context["label"] is not None:
            return 1.0
        return 0.0

    # Separate synthetic validation probe.
    validation_x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.1, 0.3],
        ],
        dtype=torch.float32,
    )

    validation_y = torch.tensor([0, 1, 2])

    def validation_utility(model, context):
        was_training = model.training
        model.eval()

        with torch.no_grad():
            predictions = model(validation_x).argmax(dim=1)
            accuracy = (predictions == validation_y).float().mean()

        model.train(was_training)

        return float(accuracy.item())

    def compute_reward(context):
        return (
            context["utility_delta"]
            - 0.01 * context["compute_cost"]
        )

    controller = HybridTrainingController(
        sample_agent=sample_agent,
        compute_agent=compute_agent,
        action_manager=actions,
        memory_manager=memory,
        resource_manager=resources,
        batch_trainer=trainer,
        label_provider=label_provider,
        sample_state_builder=sample_state_builder,
        compute_state_builder=compute_state_builder,
        sample_reward_fn=sample_reward,
        validation_utility_fn=validation_utility,
        compute_reward_fn=compute_reward,
        compute_costs={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        },
        compute_updates={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        },
        batch_size=2,
        label_cost=1,
        base_seed=42,
    )

    return controller, model, sample_agent, compute_agent, resources, memory


def make_samples():

    return [
        {
            "id": 1,
            "features": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "id": 2,
            "features": [0.2, 0.3, 0.4, 0.5],
        },
        {
            "id": 3,
            "features": [0.3, 0.4, 0.5, 0.6],
        },
        {
            "id": 4,
            "features": [0.4, 0.5, 0.6, 0.7],
        },
    ]


def test_two_stage_controller_runs_two_batches():

    (
        controller,
        model,
        sample_agent,
        compute_agent,
        resources,
        memory,
    ) = make_controller()

    initial_model = copy.deepcopy(model.state_dict())

    samples = make_samples()

    first_result = controller.process_batch(
        samples[:2],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
    )

    assert first_result["labels_acquired"] == 2
    assert first_result["memory_used"] == 2
    assert first_result["replay_records_used"] == 0
    assert first_result["compute_result"]["action"]["compute"] == 1
    assert first_result["compute_result"]["training_result"]["completed_optimizer_updates"] == 1

    second_result = controller.process_batch(
        samples[2:],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
        final=True,
    )

    assert second_result["labels_acquired"] == 2
    assert second_result["memory_used"] == 4

    # Only the previous batch's records were used for replay.
    assert second_result["replay_records_used"] == 2

    assert second_result["compute_result"]["action"]["compute"] == 1
    assert second_result["compute_result"]["training_result"]["completed_optimizer_updates"] == 1

    # Four labels consumed, two compute operations consumed.
    remaining = resources.get_resources()

    assert remaining["labels"] == 6
    assert remaining["compute"] == 18

    # Both policies received and trained on their transitions.
    assert len(sample_agent.replay_buffer) == 4
    assert len(compute_agent.replay_buffer) == 2

    assert sample_agent.training_steps == 4
    assert compute_agent.training_steps == 2

    assert controller.closed is True

    # The actual task model must have changed through training.
    changed = any(
        not torch.equal(initial_model[name], parameter)
        for name, parameter in model.state_dict().items()
    )

    assert changed


def test_controller_rejects_visible_ground_truth_label():

    controller, *_ = make_controller()

    samples = [
        {
            "id": 1,
            "features": [0.1, 0.2, 0.3, 0.4],
            "label": 1,
        }
    ]

    with pytest.raises(ValueError, match="ground-truth"):
        controller.process_batch(
            samples,
            epsilon_sample=0.0,
            epsilon_compute=0.0,
            final=True,
        )


def test_zero_label_budget_prevents_label_acquisition():

    controller, _, _, _, resources, _ = make_controller(
        label_budget=0
    )

    result = controller.process_batch(
        make_samples()[:2],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
        final=True,
    )

    assert result["labels_acquired"] == 0
    assert result["memory_used"] == 0

    training = result["compute_result"]["training_result"]
    assert training["completed_optimizer_updates"] == 0

    assert resources.get_resources()["labels"] == 0


def test_zero_compute_budget_prevents_training():

    controller, _, _, _, resources, _ = make_controller(
        compute_budget=0
    )

    result = controller.process_batch(
        make_samples()[:2],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
        final=True,
    )

    assert result["labels_acquired"] == 2
    assert result["compute_result"]["action"]["compute"] == 0

    training = result["compute_result"]["training_result"]
    assert training["completed_optimizer_updates"] == 0

    assert resources.get_resources()["compute"] == 0


def test_replay_only_training_when_no_current_labels():

    controller, _, sample_agent, compute_agent, _, _ = make_controller()

    controller.process_batch(
        make_samples()[:2],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
    )

    set_preferred_action(sample_agent, 0)
    set_preferred_action(compute_agent, 1)

    result = controller.process_batch(
        make_samples()[2:],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
        final=True,
    )

    assert result["labels_acquired"] == 0
    assert result["replay_records_used"] == 2

    training = result["compute_result"]["training_result"]

    assert training["completed_optimizer_updates"] == 1
    assert training["current_presentations"] == 0
    assert training["replay_presentations"] == 4


def test_full_memory_does_not_exceed_capacity():

    controller, _, sample_agent, compute_agent, resources, memory = make_controller(
    label_budget=10
)

    with torch.no_grad():
        final_layer = sample_agent.online_network.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(-20.0)
        final_layer.bias[3] = 5.0
        final_layer.bias[4] = 10.0
    set_preferred_action(compute_agent, 1)

    controller.process_batch(
        make_samples()[:2],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
    )

    controller.process_batch(
        make_samples()[2:],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
    )

    assert len(memory.get_memory_records()) == 4

    set_preferred_action(sample_agent, 3)

    result = controller.process_batch(
        [{"id": 5, "features": [0.5, 0.6, 0.7, 0.8]}],
        epsilon_sample=0.0,
        epsilon_compute=0.0,
        final=True,
    )

    assert result["labels_acquired"] == 1
    assert result["memory_used"] == 4
    assert len(memory.get_memory_records()) == 4
    assert resources.get_resources()["labels"] == 5
