# Porting a Model to Anomalib

Implementing a new anomaly detection algorithm in Anomalib requires following a specific structure that separates the pure PyTorch model from the LightningModule wrapper. This separation ensures that the model can be used both as a training component and as a standalone inference module.

## Step 1: Understand the Algorithm

Before starting the implementation, identify the key characteristics of the model:

- **Inputs**: What is the expected image size? Are there specific normalization requirements?
- **Learning Type**: Is it `ONE_CLASS`, `ZERO_SHOT`, or `FEW_SHOT`?
- **State Management**: Does the model require a memory bank, a teacher-student setup, or iterative training?

## Step 2: Create the Directory Structure

New models should be placed in the appropriate subdirectory within `src/anomalib/models/`. For a general image-based model:

```bash
src/anomalib/models/image/my_model/
├── __init__.py
├── torch_model.py
└── lightning_model.py
```

## Step 3: Implement `torch_model.py`

This file contains the core logic of the algorithm using pure PyTorch. It should not depend on PyTorch Lightning.

```python
from torch import nn, Tensor
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.models.components import TimmFeatureExtractor

class MyModelModel(nn.Module):
    """Pure PyTorch implementation of the algorithm."""

    def __init__(self, backbone: str = "resnet18", layers: list[str] = ["layer1", "layer2", "layer3"]):
        super().__init__()
        self.feature_extractor = TimmFeatureExtractor(backbone=backbone, layers=layers)
        # Algorithm-specific layers (e.g., scoring functions, memory modules)

    def forward(self, batch: ImageBatch) -> InferenceBatch:
        """Run inference on a batch of images.

        Args:
            batch: ImageBatch with image tensor of shape [N, C, H, W].

        Returns:
            InferenceBatch with pred_score [N] and anomaly_map [N, 1, H, W].
        """
        features = self.feature_extractor(batch.image)

        # Algorithm-specific processing
        # Example: Calculate distance from normal features
        anomaly_map = ...  # Resulting tensor of shape [N, 1, H, W]

        # Calculate a scalar score for each image
        pred_score = anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values

        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map,
            pred_label=None,
            pred_mask=None
        )
```

## Step 4: Implement `lightning_model.py`

This file defines the `AnomalibModule` which wraps the PyTorch model and handles training logic.

```python
from typing import Any
from torch import Tensor
from anomalib.models import AnomalibModule, LearningType
from .torch_model import MyModelModel

class MyModel(AnomalibModule):
    """AnomalibModule wrapper for MyModel."""

    def __init__(self, backbone: str = "resnet18", layers: list[str] = ["layer1", "layer2", "layer3"]):
        super().__init__()
        self.model = MyModelModel(backbone=backbone, layers=layers)
        self.learning_type = LearningType.ONE_CLASS

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Define training logic for one step."""
        # For memory-bank models, this might just extract features
        # For teacher-student, this would compute a loss
        loss = ...
        return loss

    def configure_pre_processor(self) -> Any:
        """Return the default pre-processor for this model."""
        return True # Uses default based on task

    def configure_post_processor(self) -> Any:
        """Return the default post-processor for this model."""
        return True

    def configure_evaluator(self) -> Any:
        """Return the default evaluator for this model."""
        return True
```

## Step 5: Register the Model

Update `src/anomalib/models/__init__.py` to include your new model class. This enables the library to discover it through the Registry.

```python
# src/anomalib/models/__init__.py
from .image.my_model.lightning_model import MyModel

__all__ = [..., "MyModel"]
```

## Step 6: Create Config YAML

Define the default hyperparameters in `examples/configs/model/my_model.yaml`:

```yaml
model:
  class_path: anomalib.models.MyModel
  init_args:
    backbone: resnet18
    layers: ["layer1", "layer2", "layer3"]
```

## Patterns for Specific Model Types

### Memory Bank Models

Used in algorithms like PatchCore. You need to override the `on_validation_epoch_start` or `on_fit_end` to process accumulated features into a searchable structure. Use the `MemoryBankMixin` to help with state management.

### Teacher-Student Models

Common in EfficientAd. These models involve two networks where one tries to mimic the other. The loss function is typically a distance measure between their outputs on normal data.

### Normalizing Flows

Used in CFlow and FastFlow. These models map image features to a standard normal distribution. During inference, low-probability regions are flagged as anomalies.

### Zero-Shot Models

Like WinCLIP. These models don't require any training on the specific target dataset. Set `self.learning_type = LearningType.ZERO_SHOT` and ensure the `forward` pass handles feature alignment with text prompts or reference normal samples.

## Testing Your Implementation

Verify your model with these essential checks:

1. **Forward Pass**: Ensure the `InferenceBatch` has the correct shapes.
2. **Training Step**: Run a few iterations on a small dataset to check for gradient flow.
3. **Export**: Confirm the model can be exported to OpenVINO and ONNX without issues.

## Reference Implementations

- **Simple**: PaDiM (`models/image/padim/`) — Demonstrates Gaussian memory and a straightforward single-pass approach.
- **Complex**: PatchCore (`models/image/patchcore/`) — Shows how to implement memory banks and coreset subsampling.
- **Advanced**: EfficientAd (`models/image/efficient_ad/`) — A great example of the teacher-student paradigm with complex pre-processing requirements.
