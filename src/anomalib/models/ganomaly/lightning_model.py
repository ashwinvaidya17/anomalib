"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule
from anomalib.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss

from .torch_model import GanomalyModel

logger = logging.getLogger(__name__)


class Ganomaly(AnomalyModule):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        batch_size (int): Batch size.
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(
        self,
        batch_size: int,
        input_size: tuple[int, int],
        n_features: int,
        latent_vec_size: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()

        self.model: GanomalyModel = GanomalyModel(
            input_size=input_size,
            num_input_channels=3,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )

        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.automatic_optimization = False

        # TODO: LR should be part of optimizer in config.yaml! Since ganomaly has custom
        #   optimizer this is to be addressed later.
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def _reset_min_max(self) -> None:
        """Resets min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return [optimizer_d, optimizer_g]

    def training_step(
        self,
        batch: dict[str, str | Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:  # pylint: disable=arguments-differ
        """Training step.

        Args:
            batch (dict[str, str | Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.
        d_opt, g_opt = self.optimizers()

        # forward pass
        padded, fake, latent_i, latent_o = self.model(batch["image"])
        pred_real, _ = self.model.discriminator(padded)

        # generator update
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        g_opt.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        g_opt.step()

        # discrimator update
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)

        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        self.log_dict(
            {"generator_loss": g_loss.item(), "discriminator_loss": d_loss.item()},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"generator_loss": g_loss, "discriminator_loss": d_loss}

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        del args, kwargs  # Unused arguments.

        batch["pred_scores"] = self.model(batch["image"])
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Normalize outputs based on min/max values."""
        # logger.info("Normalizing validation outputs based on min/max values.")
        outputs["pred_scores"] = self._normalize(outputs["pred_scores"])
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step."""
        del args, kwargs  # Unused arguments.

        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Normalize outputs based on min/max values."""
        # logger.info("Normalizing test outputs based on min/max values.")
        outputs["pred_scores"] = self._normalize(outputs["pred_scores"])
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx=dataloader_idx)

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "max_epochs": 100, "num_sanity_val_steps": 0}


class GanomalyLightning(Ganomaly):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            batch_size=hparams.dataset.train_batch_size,
            input_size=hparams.model.input_size,
            n_features=hparams.model.n_features,
            latent_vec_size=hparams.model.latent_vec_size,
            extra_layers=hparams.model.extra_layers,
            add_final_conv_layer=hparams.model.add_final_conv,
            wadv=hparams.model.wadv,
            wcon=hparams.model.wcon,
            wenc=hparams.model.wenc,
            lr=hparams.model.lr,
            beta1=hparams.model.beta1,
            beta2=hparams.model.beta2,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[Callback]:
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]
