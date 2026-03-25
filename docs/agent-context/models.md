# Anomalib Models — Agent Context

Anomalib follows a standardized pattern for model implementation to ensure consistency, scalability, and seamless export capabilities.

## Architecture Pattern

Models are organized by modality under `src/anomalib/models/image/` or `src/anomalib/models/video/`. Every model consists of two core files:

1. **`torch_model.py`**: A pure PyTorch `nn.Module` that implements the algorithm's logic, such as feature extraction, scoring, or reconstruction.
2. **`lightning_model.py`**: An `AnomalibModule` wrapper that integrates the algorithm into the PyTorch Lightning training and validation loops.

### Base Class: `AnomalibModule`

Defined at `src/anomalib/models/components/base/anomalib_module.py`.

- Extends `ExportMixin`, `pl.LightningModule`, and `ABC`.
- **Abstract Properties**:
  - `trainer_arguments`: Returns a dictionary of Lightning Trainer flags (e.g., `{"max_epochs": 1}`).
  - `learning_type`: Returns a `LearningType` enum value.
- **Overridable Methods**:
  - `training_step()`: Defines the training logic.
  - `validation_step()`: Defines the validation and testing logic.
  - `fit()`: Optional. Often used by memory-bank models (e.g., PatchCore) to build the bank after a single pass.
- **Component Resolution**: Static methods like `configure_pre_processor()` and `configure_post_processor()` define default behaviors.

### Data Flow

- **Pipeline**: `pre_processor(batch) -> model(batch) -> post_processor(batch)`.
- **Output**: The `forward` method should return an `InferenceBatch` (NamedTuple) containing `pred_score`, `pred_label`, `anomaly_map`, and `pred_mask`.
- **Learning Types**: `ONE_CLASS` (unsupervised/semi-supervised), `ZERO_SHOT` (WinCLIP, VLM-AD), and `FEW_SHOT`.

## Available Models

| Model               | Type  | Learning           | Key Idea                                   |
| ------------------- | ----- | ------------------ | ------------------------------------------ |
| CFA                 | Image | ONE_CLASS          | Coupled-hypersphere Feature Adaptation     |
| Cflow               | Image | ONE_CLASS          | Conditional normalizing flows              |
| CSFlow              | Image | ONE_CLASS          | Cross-scale normalizing flows              |
| DFKDE               | Image | ONE_CLASS          | Deep feature kernel density estimation     |
| DFM                 | Image | ONE_CLASS          | Deep feature modeling                      |
| Dinomaly            | Image | ONE_CLASS          | DINOv2-based anomaly detection             |
| DRAEM               | Image | ONE_CLASS          | Discriminatively trained AE reconstruction |
| DSR                 | Image | ONE_CLASS          | Dual subspace re-projection                |
| EfficientAd         | Image | ONE_CLASS          | Teacher-student with autoencoder           |
| FastFlow            | Image | ONE_CLASS          | Fast normalizing flows                     |
| FRE                 | Image | ONE_CLASS          | Feature reconstruction error               |
| GANomaly            | Image | ONE_CLASS          | GAN-based anomaly detection                |
| PaDiM               | Image | ONE_CLASS          | Patch Distribution Modeling                |
| PatchCore           | Image | ONE_CLASS          | Memory bank + coreset subsampling          |
| ReverseDistillation | Image | ONE_CLASS          | Reverse knowledge distillation             |
| STFPM               | Image | ONE_CLASS          | Student-Teacher Feature Pyramid Matching   |
| SuperSimpleNet      | Image | ONE_CLASS          | Simple discriminative network              |
| UFlow               | Image | ONE_CLASS          | U-shaped normalizing flows                 |
| UniNet              | Image | ONE_CLASS          | Unified network                            |
| VLM-AD              | Image | ZERO_SHOT          | Vision-language model anomaly detection    |
| WinCLIP             | Image | ZERO_SHOT/FEW_SHOT | CLIP-based window anomaly detection        |
| AnomalyDINO         | Image | ZERO_SHOT/FEW_SHOT | DINO-based anomaly detection               |
| AI-VAD              | Video | ONE_CLASS          | AI video anomaly detection                 |
| FUVAS               | Video | ONE_CLASS          | Future video anomaly scoring               |

## Implementation Template

### `torch_model.py`

```python
from torch import nn
from anomalib.data import ImageBatch, InferenceBatch

class MyModelModel(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        # Architecture components here

    def forward(self, batch: ImageBatch) -> InferenceBatch:
        # Compute scores and maps
        return InferenceBatch(pred_score=..., anomaly_map=..., pred_label=None, pred_mask=None)
```

### `lightning_model.py`

```python
from anomalib.models.components import AnomalibModule
from anomalib.data import ImageBatch
from .torch_model import MyModelModel

class MyModel(AnomalibModule):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        self.model = MyModelModel(backbone=backbone)

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        # Define default transforms
        ...

    def training_step(self, batch: ImageBatch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        loss = self.model(batch)
        return {"loss": loss}

    def validation_step(self, batch: ImageBatch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs
        predictions = self.model(batch)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    @property
    def trainer_arguments(self) -> dict:
        return {}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS
```

## Registration Checklist

1. **Export Class**: Create `src/anomalib/models/image/my_model/__init__.py` to export the model class.
2. **Registry**: Add the import to `src/anomalib/models/__init__.py` and include it in `__all__`.
3. **Config**: Create a YAML configuration at `examples/configs/model/my_model.yaml`.
4. **Tests**: Implement unit tests for the torch and lightning modules.

## Common Patterns

- **Memory Bank**: Models like PatchCore or PaDiM should override `fit()` to populate the bank after the training pass.
- **Coreset Subsampling**: Use `MemoryBankMixin` for efficient memory management.
- **Feature Extraction**: Prefer `TimmFeatureExtractor` from `anomalib.models.components.feature_extractors` for backbone flexibility.
- **Single Pass Models**: Set `trainer_arguments` to return `{"max_epochs": 1}` for models that only require one epoch to build statistics.

## Common Mistakes

- **Registration**: Forgetting to update `src/anomalib/models/__init__.py`.
- **Batch Handling**: Modifying the batch in-place instead of using `batch.update()`.
- **Export Compatibility**: Using standard `CenterCrop` instead of `ExportableCenterCrop`, which breaks model export.
- **Return Types**: Not returning an `InferenceBatch` from the `torch_model` forward call.
- **Missing Config**: Neglecting to add the corresponding YAML in `examples/configs/model/`.

## Key Files

- **Base Module**: `src/anomalib/models/components/base/anomalib_module.py`
- **Export Mixin**: `src/anomalib/models/components/base/export_mixin.py`
- **Feature Extractors**: `src/anomalib/models/components/feature_extractors/`
- **Registry**: `src/anomalib/models/__init__.py`
- **Reference Implementation**: `src/anomalib/models/image/patchcore/lightning_model.py`
