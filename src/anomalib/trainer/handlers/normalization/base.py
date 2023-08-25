from abc import ABC, abstractmethod

from lightning import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

from anomalib.models import AnomalyModule


class BaseNormalizer(ABC):
    def __init__(self, metric: Metric):
        super().__init__()
        self.metric = metric

    @abstractmethod
    def update(self, trainer: Trainer, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        """Update internal metric state."""
        pass

    @abstractmethod
    def normalize(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        """Normalizes the input data inplace."""
        pass

    def compute(self, pl_module: AnomalyModule) -> None:
        """Compute the metric."""
        pl_module.normalization_metrics.compute()
