
"""Flexible batch trainer for Hybrid Open-World AI V1.

This module is separate from the historical E3 baseline.
It trains the supplied task model using current labelled samples
and an optional snapshot of previous memory records.

Compute-tier-to-update mapping is intentionally not defined here.
The caller supplies optimizer_updates explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class HybridBatchTrainer:
    """Train a task model using current samples and previous replay memory."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 0.001,
        device: str = "cpu",
        input_dim: int = 128,
        num_classes: int = 6,
        current_per_update: int = 5,
        replay_per_update: int = 5,
        warmup_current_per_update: int = 10,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module")

        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")

        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 1:
            raise ValueError("num_classes must be an integer greater than 1")

        for name, value in (
            ("current_per_update", current_per_update),
            ("replay_per_update", replay_per_update),
            ("warmup_current_per_update", warmup_current_per_update),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise ValueError("learning_rate must be numeric")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.model = model.to(self.device)
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.current_per_update = current_per_update
        self.replay_per_update = replay_per_update
        self.warmup_current_per_update = warmup_current_per_update

        if optimizer is not None and not isinstance(
            optimizer, torch.optim.Optimizer
        ):
            raise TypeError("optimizer must be a PyTorch optimizer")

        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
        )

        self.loss_function = nn.CrossEntropyLoss()

    def _validate_current_data(
        self,
        features: Any,
        labels: Any,
    ) -> tuple[np.ndarray, np.ndarray]:

        if features is None:
            x = np.empty((0, self.input_dim), dtype=np.float32)
        else:
            x = np.asarray(features, dtype=np.float32)

            if x.size == 0:
                x = np.empty((0, self.input_dim), dtype=np.float32)

        if labels is None:
            y = np.empty((0,), dtype=np.int64)
        else:
            raw_y = np.asarray(labels)

            if raw_y.size == 0:
                y = np.empty((0,), dtype=np.int64)
            else:
                if not np.issubdtype(raw_y.dtype, np.number):
                    raise ValueError("Labels must be numeric")

                if not np.isfinite(raw_y).all():
                    raise ValueError("Labels contain non-finite values")

                if not np.equal(raw_y, np.floor(raw_y)).all():
                    raise ValueError("Labels must be integers")

                y = raw_y.astype(np.int64)

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Current features must have shape (N, {self.input_dim})"
            )

        if y.ndim != 1 or len(x) != len(y):
            raise ValueError("Current feature and label counts do not match")

        if not np.isfinite(x).all():
            raise ValueError("Current features contain non-finite values")

        if len(y) > 0:
            if np.any(y < 1) or np.any(y > self.num_classes):
                raise ValueError(
                    f"Labels must be between 1 and {self.num_classes}"
                )

        return x, y

    def _validate_replay_records(
        self,
        replay_records: list[dict[str, Any]] | None,
    ) -> tuple[np.ndarray, np.ndarray]:

        if replay_records is None:
            replay_records = []

        replay_features = []
        replay_labels = []

        for index, record in enumerate(replay_records):

            # MemoryManager.get_memory_records() returns
            # {"id": ..., "item": {...}} records.
            if isinstance(record, dict) and "item" in record:
                record = record["item"]

            if isinstance(record, dict):
                if "features" not in record or "label" not in record:
                    raise ValueError(
                        f"Replay record {index} must contain "
                        "'features' and 'label'"
                    )

                features = record["features"]
                label = record["label"]

            elif isinstance(record, (tuple, list)) and len(record) >= 2:
                # Also accepts E3-style (features, label, ...).
                features = record[0]
                label = record[1]

            else:
                raise ValueError(f"Invalid replay record at index {index}")

            row = np.asarray(features, dtype=np.float32)

            if row.shape != (self.input_dim,):
                raise ValueError(
                    f"Replay record {index} must contain "
                    f"{self.input_dim} features"
                )

            if not np.isfinite(row).all():
                raise ValueError(
                    f"Replay record {index} has non-finite features"
                )

            if (
                not isinstance(label, (int, np.integer))
                or isinstance(label, (bool, np.bool_))
                or not 1 <= int(label) <= self.num_classes
            ):
                raise ValueError(f"Invalid label in replay record {index}")

            replay_features.append(row)
            replay_labels.append(int(label))

        if not replay_features:
            return (
                np.empty((0, self.input_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        return (
            np.stack(replay_features).astype(np.float32),
            np.asarray(replay_labels, dtype=np.int64),
        )

    def train_batch(
        self,
        current_features: Any,
        current_labels: Any,
        replay_records: list[dict[str, Any]] | None,
        *,
        optimizer_updates: int,
        stage: int,
        seed: int,
    ) -> dict[str, Any]:
        """Train for the requested number of updates.

        Current and replay data are never mixed from the same incoming
        batch's newly acquired records: replay_records must be the
        snapshot taken at the beginning of this batch.
        """

        if (
            not isinstance(optimizer_updates, int)
            or isinstance(optimizer_updates, bool)
            or optimizer_updates < 0
        ):
            raise ValueError("optimizer_updates must be a non-negative integer")

        if not isinstance(stage, int) or isinstance(stage, bool) or stage < 1:
            raise ValueError("stage must be a positive integer")

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        current_x, current_y = self._validate_current_data(
            current_features, current_labels
        )

        replay_x, replay_y = self._validate_replay_records(replay_records)

        result = {
            "stage": stage,
            "requested_optimizer_updates": optimizer_updates,
            "completed_optimizer_updates": 0,
            "current_sample_count": len(current_x),
            "replay_memory_count": len(replay_x),
            "current_presentations": 0,
            "replay_presentations": 0,
            "total_presentations": 0,
            "losses": [],
            "mean_loss": None,
            "skipped_reason": None,
        }

        if optimizer_updates == 0:
            result["skipped_reason"] = "zero_optimizer_updates"
            return result

        if len(current_x) == 0 and len(replay_x) == 0:
            result["skipped_reason"] = "no_training_data"
            return result

        rng = np.random.default_rng(
            seed * 100_003 + stage * 1_009 + 7_919
        )

        self.model.train()

        for _ in range(optimizer_updates):

            selected_current_x = np.empty(
                (0, self.input_dim), dtype=np.float32
            )
            selected_current_y = np.empty((0,), dtype=np.int64)

            selected_replay_x = np.empty(
                (0, self.input_dim), dtype=np.float32
            )
            selected_replay_y = np.empty((0,), dtype=np.int64)

            if len(current_x) > 0 and len(replay_x) > 0:

                # Mixed training: up to 5 current + 5 replay.
                current_count = min(
                    self.current_per_update, len(current_x)
                )

                current_indices = rng.choice(
                    len(current_x),
                    size=current_count,
                    replace=False,
                )

                selected_current_x = current_x[current_indices]
                selected_current_y = current_y[current_indices]

                # Replay is sampled with replacement, as in E3.
                replay_indices = rng.choice(
                    len(replay_x),
                    size=self.replay_per_update,
                    replace=True,
                )

                selected_replay_x = replay_x[replay_indices]
                selected_replay_y = replay_y[replay_indices]

            elif len(current_x) > 0:

                # No previous memory: use available current samples.
                current_count = min(
                    self.warmup_current_per_update, len(current_x)
                )

                current_indices = rng.choice(
                    len(current_x),
                    size=current_count,
                    replace=False,
                )

                selected_current_x = current_x[current_indices]
                selected_current_y = current_y[current_indices]

            else:

                # Replay-only training is allowed when no current
                # labelled samples were acquired.
                replay_count = (
                    self.current_per_update + self.replay_per_update
                )

                replay_indices = rng.choice(
                    len(replay_x),
                    size=replay_count,
                    replace=True,
                )

                selected_replay_x = replay_x[replay_indices]
                selected_replay_y = replay_y[replay_indices]

            batch_x_np = np.concatenate(
                [selected_current_x, selected_replay_x], axis=0
            )

            batch_y_np = np.concatenate(
                [selected_current_y, selected_replay_y], axis=0
            )

            if len(batch_y_np) == 0:
                raise RuntimeError("Training batch unexpectedly became empty")

            # Shuffle current and replay examples together.
            order = rng.permutation(len(batch_y_np))
            batch_x_np = batch_x_np[order]
            batch_y_np = batch_y_np[order]

            batch_x = torch.as_tensor(
                batch_x_np,
                dtype=torch.float32,
                device=self.device,
            )

            # Dataset labels are 1..6; CrossEntropy expects 0..5.
            batch_y = torch.as_tensor(
                batch_y_np - 1,
                dtype=torch.long,
                device=self.device,
            )

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(batch_x)

            if logits.shape != (len(batch_y_np), self.num_classes):
                raise ValueError(
                    "Model output must have shape "
                    f"(batch_size, {self.num_classes})"
                )

            loss = self.loss_function(logits, batch_y)

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Training loss became non-finite")

            loss.backward()

            for parameter in self.model.parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Model gradient became non-finite"
                    )

            self.optimizer.step()

            current_count = len(selected_current_y)
            replay_count = len(selected_replay_y)

            result["losses"].append(float(loss.detach().cpu().item()))
            result["completed_optimizer_updates"] += 1
            result["current_presentations"] += current_count
            result["replay_presentations"] += replay_count
            result["total_presentations"] += current_count + replay_count

        result["mean_loss"] = float(np.mean(result["losses"]))

        return result
