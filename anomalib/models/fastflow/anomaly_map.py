"""FastFlow Anomaly Map Generator Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size: Union[ListConfig, Tuple]):
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def __call__(self, hidden_variables: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Generate Anomaly Heatmap.

        This implementation generates the heatmap based on the flow maps
        computed from the normalizing flow (NF) FastFlow blocks. Each block
        yields a flow map, which overall is stacked and averaged to an anomaly
        map.

        Args:
            hidden_variables (List[Tensor]): List of hidden variables from each NF FastFlow block.

        Returns:
            Tensor: Anomaly Map.
        """
        flow_maps: List[Tensor] = []

        # Max features is generated by taking the max along h and w dim for each hidden variable
        # and stacking them.
        max_activation_val: Optional[Tensor] = None

        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5

            max_features = torch.exp(-(hidden_variable**2))
            max_features, _ = torch.max(einops.rearrange(max_features, "b c h w -> b c (h w)"), -1)
            if max_activation_val is None:
                max_activation_val = max_features
            else:
                max_activation_val = torch.hstack([max_activation_val, max_features])

            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)

        return anomaly_map, max_activation_val
