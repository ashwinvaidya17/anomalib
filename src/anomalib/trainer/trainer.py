"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from lightning import Callback
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig

from anomalib.data import TaskType
from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.utils.metrics.thresholding import BaseAnomalyThreshold, F1AdaptiveThreshold

from .callbacks import MetricsManagerCallback, PostProcessorCallback, ThresholdingCallback, get_normalization_callback

log = logging.getLogger(__name__)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        callbacks: Add a callback or list of callbacks.
    """

    def __init__(
        self,
        callbacks: list[Callback] = [],
        normalizer: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.MIN_MAX,
        threshold: BaseAnomalyThreshold
        | tuple[BaseAnomalyThreshold, BaseAnomalyThreshold]
        | DictConfig
        | ListConfig
        | str = F1AdaptiveThreshold(),
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.normalizer = normalizer
        self.threshold = threshold
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        super().__init__(callbacks=self._setup_callbacks(callbacks), **kwargs)

        self.lightning_module: AnomalyModule

    def _setup_callbacks(self, callbacks: list[Callback]) -> list[Callback]:
        """Setup callbacks for the trainer."""
        # Note: this needs to be changed when normalization is part of the trainer
        _callbacks: list[Callback] = [PostProcessorCallback()]

        normalization_callback = get_normalization_callback(self.normalizer)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)

        _callbacks.append(ThresholdingCallback(self.threshold))
        _callbacks.append(MetricsManagerCallback(self.task, self.image_metric_names, self.pixel_metric_names))
        return _callbacks + callbacks
