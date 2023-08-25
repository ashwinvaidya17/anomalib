"""Manages metrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from anomalib.models import AnomalyModule


class MetricsManager:
    """Setup and update metrics."""

    def __init__(
        self,
    ):
        pass

    def reset(self, pl_module: AnomalyModule):
        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def set_threshold(self, pl_module: AnomalyModule, threshold: tuple[float, float] | float):
        if isinstance(threshold, float):
            pl_module.image_metrics.set_threshold(threshold)
            pl_module.pixel_metrics.set_threshold(threshold)
        else:
            pl_module.image_metrics.set_threshold(threshold[0])
            pl_module.pixel_metrics.set_threshold(threshold[1])

    @staticmethod
    def update(
        pl_module: AnomalyModule,
        output: dict[str, Any],
    ) -> None:
        pl_module.image_metrics.cpu()
        pl_module.image_metrics.update(output["pred_scores"], output["label"].int())
        if "mask" in output.keys() and "anomaly_maps" in output.keys():
            pl_module.pixel_metrics.cpu()
            pl_module.pixel_metrics.update(torch.squeeze(output["anomaly_maps"]), torch.squeeze(output["mask"].int()))
