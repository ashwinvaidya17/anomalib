# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning callback for sending progress to the frontend via the Plugin API."""

from __future__ import annotations

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import RunningStage

from communication.ipc import IPCConnection

logger = logging.getLogger(__name__)


class GetiInspectProgressCallback(Callback):
    """Callback for displaying training/validation/testing progress in the Geti Inspect UI.

    This callback sends progress events through a multiprocessing queue that the
    main process polls and broadcasts via WebSocket to connected frontend clients.

    Args:
        event_queue: Multiprocessing queue for sending events to main process

    Example:
        trainer = Trainer(callbacks=[GetiInspectProgressCallback(event_queue=queue)])
    """

    def __init__(self, ipc_pipe: IPCConnection) -> None:
        """Initialize the callback with an event queue for IPC.

        Args:
            ipc_pipe: IPC pipe for sending events to main process
        """
        self.ipc_pipe = ipc_pipe

    def _check_cancel_training(self, trainer: Trainer) -> None:
        """Check if training should be canceled."""
        message = self.ipc_pipe.read()
        should_cancel = bool(message and message.event == "cancel_training")
        if should_cancel:
            trainer.should_stop = True

    def _send_progress(self, progress: float, stage: RunningStage) -> None:
        """Send progress update to frontend via event queue.

        Puts a generic event message into the multiprocessing queue which will
        be picked up by the main process and broadcast via WebSocket.

        Args:
            progress: Progress value between 0.0 and 1.0
            stage: The current training stage
        """
        # Convert progress to percentage (0-100)
        progress_percent = int(progress * 100)

        try:
            # Send generic event message: {"event": "...", "data": {...}}
            self.ipc_pipe.broadcast(
                event="progress_update",
                data={
                    "status": "Running",
                    "stage": stage.name,
                    "progress": progress_percent,
                },
            )
            logger.info("Sent progress: %s - %d%%", stage.name, progress_percent)
        except Exception as e:
            logger.warning("Failed to send progress to event queue: %s", e)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.ipc_pipe.broadcast(
            event="progress_update",
            data={
                "status": "Running",
                "stage": stage.name,
                "progress": 0,
            },
        )

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: RunningStage) -> None:
        self.ipc_pipe.broadcast(
            event="progress_update",
            data={
                "status": "Completed",
                "stage": stage.name,
                "progress": 100,
            },
        )

    # Training callbacks
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training starts."""
        self._send_progress(0.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called when a training batch starts."""
        self._check_cancel_training(trainer)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch ends."""
        self._check_cancel_training(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a training epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    # Validation callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation starts."""
        # self._send_progress(0.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a validation batch starts."""
        self._check_cancel_training(trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a validation batch ends."""
        self._check_cancel_training(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a validation epoch ends."""
        # progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        # self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation ends."""
        # self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    # Test callbacks
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing starts."""
        self._send_progress(0.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a test batch starts."""
        self._check_cancel_training(trainer)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a test batch ends."""
        self._check_cancel_training(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a test epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing ends."""
        self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    # Predict callbacks
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction starts."""
        self._send_progress(0.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a prediction batch starts."""
        self._check_cancel_training(trainer)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a prediction epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a prediction batch ends."""
        self._check_cancel_training(trainer)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction ends."""
        self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)
