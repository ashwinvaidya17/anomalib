from pathlib import Path
from typing import Any, Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tests.legacy.helpers.dataset import get_dataset_path
from tests.legacy.helpers.metrics import get_metrics
from torch import nn

from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks import ImageVisualizerCallback


class _DummyAnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (100, 100)
        self.sigma = 4


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anomaly_map_generator = _DummyAnomalyMapGenerator()


class DummyModule(AnomalyModule):
    """A dummy model which calls visualizer callback on fake images and masks."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        self.model = _DummyModel()
        self.task = "segmentation"
        self.mode = "full"

        self.image_metrics, self.pixel_metrics = get_metrics(hparams)
        self.image_metrics.set_threshold(hparams.model.threshold.image_default)
        self.pixel_metrics.set_threshold(hparams.model.threshold.pixel_default)

    def test_step(self, batch, _):
        """Only used to trigger on_test_epoch_end."""
        self.log(name="loss", value=0.0, prog_bar=True)
        outputs = dict(
            image_path=[Path(get_dataset_path("bottle")) / "broken_large/000.png"],
            image=torch.rand((1, 3, 100, 100)).to(batch.device),
            mask=torch.zeros((1, 100, 100)).to(batch.device),
            anomaly_maps=torch.ones((1, 100, 100)).to(batch.device),
            label=torch.Tensor([0]).to(batch.device),
            pred_labels=torch.Tensor([0]).to(batch.device),
            pred_masks=torch.zeros((1, 100, 100)).to(batch.device),
        )
        return outputs

    def configure_optimizers(self):
        return None

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {}