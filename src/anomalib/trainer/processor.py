"""Callback that attaches necessary pre/post-processing to the model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lightning import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models import AnomalyModule
from anomalib.utils.metrics.thresholding import BaseAnomalyThreshold, F1AdaptiveThreshold

from .handlers import BaseNormalizer, MetricsManager, MinMaxNormalizer, PostProcessor, Thresholder


class _ProcessorCallback(Callback):
    def __init__(
        self,
        image_threshold: BaseAnomalyThreshold = F1AdaptiveThreshold(),
        pixel_threshold: BaseAnomalyThreshold | None = None,
        normalizer: BaseNormalizer = MinMaxNormalizer()
        # visualizers: list[VisualizationHandler]| VisualizationHandler
    ):
        self.thresholder = Thresholder(image_threshold, pixel_threshold)
        self.post_processor = PostProcessor()
        self.metrics_manager = MetricsManager()
        self.normalizer = normalizer

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str) -> None:
        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = self.normalizer.metric.cpu()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self.post_processor.process(trainer, pl_module, outputs)
            self.normalizer.update(trainer, pl_module, outputs)
            self._outputs_to_cpu(outputs)
            self.thresholder.update(trainer, pl_module, outputs)
            self.metrics_manager.update(pl_module, outputs)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        self.thresholder.reset(trainer, pl_module)
        self.metrics_manager.reset(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        self.thresholder.compute(trainer, pl_module)
        self.metrics_manager.set_threshold(
            pl_module, (pl_module.image_threshold.value.item(), pl_module.pixel_threshold.value.item())
        )
        self.normalizer.compute(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        self.metrics_manager.set_threshold(pl_module, 0.5)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        self.metrics_manager.reset(pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self.post_processor.process(trainer, pl_module, outputs)
            self.normalizer.normalize(pl_module, outputs)
            self._outputs_to_cpu(outputs)
            self.metrics_manager.update(pl_module, outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self._outputs_to_cpu(outputs)
            self.post_processor.process(trainer, pl_module, outputs)
            self.normalizer.normalize(pl_module, outputs)

    def _outputs_to_cpu(self, output):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [self._outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output
