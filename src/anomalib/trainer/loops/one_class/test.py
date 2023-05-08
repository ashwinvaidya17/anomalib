"""Test loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from anomalib import trainer
from anomalib.trainer.utils import VisualizationStage


class AnomalibTestEpochLoop(EvaluationEpochLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: trainer.AnomalibTrainer

    def _evaluation_step_end(self, *args, **kwargs: Any) -> STEP_OUTPUT | None:
        """Computes prediction scores, bounding boxes, and applies thresholding before normalization."""
        outputs = super()._evaluation_step_end(*args, **kwargs)
        if outputs is not None:
            self.trainer.post_processor.apply_predictions(outputs)
            self.trainer.post_processor.apply_thresholding(outputs)
            if self.trainer.normalizer:
                self.trainer.normalizer.normalize(outputs)
        return outputs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Track batch outputs for epoch end.

        If this is not overridden, the outputs are not collected if the model does not have a ``test_epoch_end``
        method. Lightning ends up deleting epoch outputs if this is false. This ensures that we have the outputs
        when computing the metrics on the test set.
        """
        return True


class AnomalibTestLoop(EvaluationLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: trainer.AnomalibTrainer
        self.epoch_loop = AnomalibTestEpochLoop()

    def on_run_start(self, *args, **kwargs) -> None:
        """Can be used to call setup."""
        self.trainer.thresholder.initialize()
        self.trainer.metrics_manager.initialize()
        # Reset the image and pixel thresholds to 0.5 at start of the run.
        self.trainer.metrics_manager.set_threshold()
        return super().on_run_start(*args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``test_epoch_end``.

        Args:
            outputs (List[EPOCH_OUTPUT])
        """
        super()._evaluation_epoch_end(outputs)

        output_or_outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )
        self.trainer.metrics_manager.compute(output_or_outputs)
        self.trainer.metrics_manager.log(self.trainer, "test_epoch_end")
        self.trainer.visualization_manager.visualize_images(output_or_outputs, VisualizationStage.TEST)
        self.trainer.visualization_manager.visualize_metrics(
            VisualizationStage.TEST,
        )