"""Implements custom trainer for Anomalib."""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from pytorch_lightning import Trainer

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.trainer.loops import AnomalibFitLoop, AnomalibPredictionLoop, AnomalibTestLoop, AnomalibValidationLoop
from anomalib.trainer.utils import MetricsManager, Normalizer, PostProcessor, Thresholder

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.


    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        threshold_method (ThresholdMethod): Thresholding method for normalizer.
        normalization_method (NormalizationMethod): Normalization method
        manual_image_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
        manual_pixel_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
    """

    def __init__(
        self,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        manual_image_threshold: Optional[float] = None,
        manual_pixel_threshold: Optional[float] = None,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        task_type: TaskType = TaskType.SEGMENTATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = AnomalibFitLoop(
            min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None)
        )
        self.validate_loop = AnomalibValidationLoop()
        self.test_loop = AnomalibTestLoop()
        self.predict_loop = AnomalibPredictionLoop()

        self.task_type = task_type

        self.thresholder = Thresholder(
            trainer=self,
            threshold_method=threshold_method,
            manual_image_threshold=manual_image_threshold,
            manual_pixel_threshold=manual_pixel_threshold,
        )

        self.post_processor = PostProcessor(trainer=self)
        self.normalizer = Normalizer(normalization_method=normalization_method)
        self.metrics_manager = MetricsManager(trainer=self, image_metrics=image_metrics, pixel_metrics=pixel_metrics)
