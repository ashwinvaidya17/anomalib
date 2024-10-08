"""Visual Anomaly Model for Zero/Few-Shot Anomaly Classification."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.models import AnomalyModule

from .backends import Backend, ChatGPT, Huggingface, Ollama
from .utils import Prompt, VLMModel

logger = logging.getLogger(__name__)


class VlmAd(AnomalyModule):
    """Visual anomaly model."""

    def __init__(
        self,
        model: VLMModel | str = VLMModel.LLAMA_OLLAMA,
        api_key: str | None = None,
        k_shot: int = 3,
    ) -> None:
        super().__init__()
        self.k_shot = k_shot
        model = VLMModel(model)
        self.vlm_backend: Backend = self._setup_vlm(model, api_key)

    @staticmethod
    def _setup_vlm(model: VLMModel, api_key: str | None) -> Backend:
        if model == VLMModel.LLAMA_OLLAMA:
            return Ollama(model_name=model.value)
        if model == VLMModel.GPT_4O_MINI:
            return ChatGPT(api_key=api_key, model_name=model.value)
        if model in {VLMModel.VICUNA_7B_HF, VLMModel.VICUNA_13B_HF, VLMModel.MISTRAL_7B_HF}:
            return Huggingface(model_name=model.value)

        msg = f"Unsupported VLM model: {model}"
        raise ValueError(msg)

    def _setup(self) -> None:
        if self.k_shot:
            logger.info("Collecting reference images from training dataset.")
            dataloader = self.trainer.datamodule.train_dataloader()
            self.collect_reference_images(dataloader)

    def collect_reference_images(self, dataloader: DataLoader) -> None:
        """Collect reference images for few-shot inference."""
        count = 0
        for batch in dataloader:
            for img_path in batch["image_path"]:
                self.vlm_backend.add_reference_images(img_path)
                count += 1
                if count == self.k_shot:
                    return

    @property
    def prompt(self) -> Prompt:
        """Get the prompt."""
        return Prompt(
            predict=(
                "You are given an image. It is either normal or anomalous."
                " First say 'YES' if the image is anomalous, or 'NO' if it is normal.\n"
                "Then give the reason for your decision.\n"
                "For example, 'YES: The image has a crack on the wall.'"
            ),
            few_shot=(
                "These are a few examples of normal picture without any anomalies."
                " You have to use these to determine if the image I provide in the next"
                " chat is normal or anomalous."
            ),
        )

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Validation step."""
        del args, kwargs  # These variables are not used.
        responses = [(self.vlm_backend.predict(img_path, self.prompt)) for img_path in batch["image_path"]]
        batch["explanation"] = responses
        batch["pred_scores"] = torch.tensor([1.0 if r.startswith("Y") else 0.0 for r in responses], device=self.device)
        return batch

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model."""
        return LearningType.ZERO_SHOT if self.k_shot == 0 else LearningType.FEW_SHOT

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Doesn't need training."""
        return {}

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> None:
        """This modes does not require any transforms."""
        if image_size is not None:
            logger.warning("Ignoring image_size argument as each backend has its own transforms.")