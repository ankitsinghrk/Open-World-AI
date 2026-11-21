
import numpy as np
import pytest
import torch

from src.hybrid_batch_trainer import HybridBatchTrainer


def make_trainer():
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 6),
    )
    return HybridBatchTrainer(model)


def make_data(count):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(count, 128)).astype(np.float32)
    y = np.arange(count, dtype=np.int64) % 6 + 1
    return x, y


def make_memory(count=2):
    x, y = make_data(count)

    return [
        {
            "id": f"mem_{i + 1:06d}",
            "item": {
                "features": x[i],
                "label": int(y[i]),
            },
        }
        for i in range(count)
    ]


def test_mixed_current_and_replay_training():
    trainer = make_trainer()
    x, y = make_data(6)

    result = trainer.train_batch(
        x,
        y,
        make_memory(2),
        optimizer_updates=2,
        stage=2,
        seed=42,
    )

    assert result["completed_optimizer_updates"] == 2
    assert result["current_presentations"] == 10
    assert result["replay_presentations"] == 10
    assert result["total_presentations"] == 20
    assert np.isfinite(result["mean_loss"])


def test_current_only_training():
    trainer = make_trainer()
    x, y = make_data(4)

    result = trainer.train_batch(
        x,
        y,
        [],
        optimizer_updates=2,
        stage=1,
        seed=42,
    )

    assert result["completed_optimizer_updates"] == 2
    assert result["current_presentations"] == 8
    assert result["replay_presentations"] == 0


def test_replay_only_training():
    trainer = make_trainer()
    x = np.empty((0, 128), dtype=np.float32)
    y = np.empty((0,), dtype=np.int64)

    result = trainer.train_batch(
        x,
        y,
        make_memory(2),
        optimizer_updates=1,
        stage=3,
        seed=42,
    )

    assert result["completed_optimizer_updates"] == 1
    assert result["current_presentations"] == 0
    assert result["replay_presentations"] == 10


def test_no_training_data_is_handled():
    trainer = make_trainer()
    x = np.empty((0, 128), dtype=np.float32)
    y = np.empty((0,), dtype=np.int64)

    result = trainer.train_batch(
        x,
        y,
        [],
        optimizer_updates=3,
        stage=1,
        seed=42,
    )

    assert result["completed_optimizer_updates"] == 0
    assert result["skipped_reason"] == "no_training_data"


def test_zero_updates_are_handled():
    trainer = make_trainer()
    x, y = make_data(4)

    result = trainer.train_batch(
        x,
        y,
        [],
        optimizer_updates=0,
        stage=1,
        seed=42,
    )

    assert result["completed_optimizer_updates"] == 0
    assert result["skipped_reason"] == "zero_optimizer_updates"


def test_invalid_labels_are_rejected():
    trainer = make_trainer()
    x, y = make_data(3)
    y[0] = 7

    with pytest.raises(ValueError):
        trainer.train_batch(
            x,
            y,
            [],
            optimizer_updates=1,
            stage=1,
            seed=42,
        )
