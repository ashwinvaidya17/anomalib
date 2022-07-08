"""Selective Feature Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
import torch
from torch import Tensor, nn


class SelectiveFeatureModel(nn.Module):
    """Selective Feature Model.

    Args:
       feature_percentage (float): Percentage of features to keep. Defaults to 0.1.
    """

    def __init__(self, feature_percentage: float = 0.1):
        super().__init__()

        # self.register_buffer("feature_stat", torch.zeros(n_features, n_patches))

        self.feature_percentage = feature_percentage

    def forward(self, max_activation_val: Tensor, sub_class_labels: List[str]):
        """Store top `feature_percentage` features.

        Args:
          max_activation_val (Tensor): Max activation values of embeddings.
          sub_class_labels (List[str]):  Corresponding sub-class labels.
        """
        class_names = np.unique(sub_class_labels)

        for class_name in class_names:
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, Tensor())
            class_max_activations = max_activation_val[sub_class_labels == class_name]
            # sorted values and idx for entire feature set
            max_val, max_idx = torch.sort(class_max_activations, descending=True)
            reduced_range = int(max_val.shape[1] * self.feature_percentage)
            # indexes of top self.feature_percentage features having max values
            top_max_idx = max_idx[:, 0:reduced_range]
            # out of sorted top self.feature_percentage, what features are affiliated the most
            idx, repetitions = torch.unique(top_max_idx, return_counts=True)
            sorted_repetition, sorted_repetition_idx = torch.sort(repetitions, descending=True)
            sorted_idx = idx[sorted_repetition_idx]

            sorted_idx_normalized = sorted_repetition / sorted_repetition.sum()
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))

    def fit(self, max_activation_val: Tensor, class_labels: List[str]):
        """Store top `feature_percentage` features.

        Args:
            max_activation_val (Tensor): Max activation values of embeddings.
            class_labels (List[str]):  Corresponding sub-class labels.
        """
        self.forward(max_activation_val, class_labels)