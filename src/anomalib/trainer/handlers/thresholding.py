"""Callback to handle thresholding."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models import AnomalyModule
from anomalib.utils.metrics.thresholding import BaseAnomalyThreshold, F1AdaptiveThreshold


class Thresholder:
    """Handles thresholding."""

    def __init__(
        self,
        image_threshold: BaseAnomalyThreshold = F1AdaptiveThreshold(),
        pixel_threshold: BaseAnomalyThreshold | None = None,
    ):
        """Initializes the callback.

        Args:
            thresholder (BaseThresholding): Thresholding class to use.
        """
        self.image_threshold = image_threshold
        if pixel_threshold is None:
            self.pixel_threshold = image_threshold.clone()
        else:
            self.pixel_threshold = pixel_threshold

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str) -> None:
        pl_module.image_threshold = self.image_threshold
        pl_module.pixel_threshold = self.pixel_threshold

    def update(self, trainer: Trainer, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        pl_module.image_threshold.cpu()
        pl_module.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
            pl_module.pixel_threshold.cpu()
            pl_module.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())

    def reset(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.reset()
        pl_module.pixel_threshold.reset()

    def compute(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.compute()
        if pl_module.pixel_threshold._update_called:
            pl_module.pixel_threshold.compute()
        else:
            pl_module.pixel_threshold.value = pl_module.image_threshold.value
