---
name: anomalib-add-model
description: Use when adding a new anomaly detection model to anomalib. Covers the full lifecycle — understanding the source algorithm (even from JAX/TensorFlow repos), porting to PyTorch, implementing the anomalib wrapper, writing README with benchmarks, creating configs, registering the model, writing tests, and updating documentation. Triggers on phrases like "add model", "port model", "implement model", "new model", "bring in model from".
---

# Adding a New Model to Anomalib

This skill guides the full lifecycle of adding an anomaly detection model to anomalib. The source may be a paper, an existing repo (PyTorch, JAX, TensorFlow, etc.), or a from-scratch implementation.

**Before starting, read these context files:**

- `docs/agent-context/models.md` — Model architecture, base classes, patterns
- `docs/agent-context/porting-models.md` — Step-by-step porting guide
- `docs/agent-context/python-style.md` — Coding conventions, docstrings, linting
- `docs/agent-context/pre-processing.md` — PreProcessor conventions
- `docs/agent-context/post-processing.md` — PostProcessor conventions
- `docs/agent-context/engine.md` — Engine integration
- `docs/agent-context/deploy.md` — Export compatibility requirements

---

## Phase 0: Research & Understanding

Before writing any code, thoroughly understand the source algorithm.

### 0.1 Gather Source Material

- **Paper**: Find and read the arxiv/conference paper. Note the algorithm's key idea, loss function, architecture, and reported benchmarks.
- **Source repo**: Clone or browse the reference implementation. It may be in JAX, TensorFlow, or another framework.
- **Identify**:
  - Input expectations (image size, normalization, channels)
  - Learning type: `ONE_CLASS`, `ZERO_SHOT`, or `FEW_SHOT`
  - Architecture pattern: memory bank, teacher-student, normalizing flow, reconstruction, discriminative
  - State management: Does it need a memory bank? Coreset? Teacher network?
  - Training requirements: Number of epochs, optimizer, scheduler, loss function
  - Any custom layers or operations that need PyTorch equivalents

### 0.2 Cross-Framework Translation (if source is not PyTorch)

When porting from JAX/Flax, TensorFlow/Keras, or other frameworks:

| Source Framework                                  | PyTorch Equivalent                                                          |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| `jax.numpy` operations                            | `torch` tensor operations                                                   |
| `flax.linen.Module`                               | `torch.nn.Module`                                                           |
| `jax.random.PRNGKey`                              | `torch.manual_seed` / `torch.Generator`                                     |
| `tf.keras.layers.Conv2D`                          | `torch.nn.Conv2d` (note: different arg order for kernel size)               |
| `einops.rearrange`                                | `torch.permute` / `torch.reshape` / keep `einops` (it's framework-agnostic) |
| JAX functional transforms (`jax.grad`, `jax.jit`) | PyTorch autograd (automatic), `torch.compile`                               |
| Haiku `hk.transform`                              | Standard `nn.Module` with `forward()`                                       |

**Key differences to watch for:**

- **Channel ordering**: JAX/TF often use NHWC; PyTorch uses NCHW.
- **Convolution padding**: `"SAME"` padding in TF has no direct PyTorch equivalent — compute manually or use `torch.nn.functional.pad`.
- **Weight initialization**: Reproduce the original initialization scheme.
- **Batch norm momentum**: TF uses `momentum` as the decay factor (e.g., 0.99); PyTorch uses `1 - momentum` (e.g., 0.01).
- **State management**: JAX is functional (state passed explicitly); PyTorch uses stateful modules. Map JAX state dicts to PyTorch `state_dict`.

---

## Phase 1: Implementation

### 1.1 Directory Structure

Create the model directory under the appropriate modality:

```text
src/anomalib/models/image/<model_name>/
    __init__.py
    torch_model.py
    lightning_model.py
    README.md
```

If the model requires extra components (custom loss, anomaly map generator, anomaly generator), add them as separate files:

```text
src/anomalib/models/image/<model_name>/
    __init__.py
    torch_model.py
    lightning_model.py
    loss.py                    # Custom loss function (if needed)
    anomaly_map.py             # Custom anomaly map generator (if needed)
    anomaly_generator.py       # Synthetic anomaly generation (if needed)
    README.md
```

### 1.2 `torch_model.py` — Pure PyTorch Implementation

This file contains ONLY pure PyTorch code. No Lightning dependencies.

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""<ModelName>: <Paper Title>.

<Brief description of what this module implements.>

Example:
    >>> from anomalib.models.image.<model_name>.torch_model import <ModelName>Model
    >>> model = <ModelName>Model()
    >>> # ... usage example

Paper: https://arxiv.org/<paper_id>
"""

import logging
from torch import nn, Tensor
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.models.components import TimmFeatureExtractor  # if using backbone

logger = logging.getLogger(__name__)


class <ModelName>Model(nn.Module):
    """Pure PyTorch implementation of <ModelName>.

    Args:
        backbone (str): Name of the backbone CNN. Defaults to ``"resnet18"``.
        layers (list[str]): Feature extraction layers. Defaults to ``["layer2", "layer3"]``.

    Example:
        >>> model = <ModelName>Model()
    """

    def __init__(self, backbone: str = "resnet18", layers: list[str] | None = None) -> None:
        super().__init__()
        # ... architecture components

    def forward(self, batch: ImageBatch) -> InferenceBatch:
        """Run inference.

        Args:
            batch (ImageBatch): Input batch with image tensor [N, C, H, W].

        Returns:
            InferenceBatch: Predictions with pred_score, anomaly_map, pred_label, pred_mask.
        """
        # ... compute anomaly scores and maps
        return InferenceBatch(
            pred_score=pred_score,
            anomaly_map=anomaly_map,
            pred_label=None,
            pred_mask=None,
        )
```

**Rules:**

- Class name: `<ModelName>Model` (e.g., `PatchcoreModel`, `SupersimplenetModel`)
- MUST return `InferenceBatch` from `forward()`
- Use `TimmFeatureExtractor` from `anomalib.models.components.feature_extractors` for backbone feature extraction
- Use `anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values` for deriving `pred_score` from `anomaly_map` (unless the model has its own scoring)
- All type hints required on function signatures

### 1.3 `lightning_model.py` — AnomalibModule Wrapper

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""<ModelName>: <Paper Title>.

<Module docstring with description, example, paper link, and See Also.>

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import <ModelName>
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = <ModelName>()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP

Paper: https://arxiv.org/<paper_id>

See Also:
    :class:`anomalib.models.image.<model_name>.torch_model.<ModelName>Model`:
        PyTorch implementation of the <ModelName> model.
"""

import logging
from typing import Any
from collections.abc import Sequence

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import <ModelName>Model

logger = logging.getLogger(__name__)


class <ModelName>(AnomalibModule):
    """<ModelName> Lightning Module for anomaly detection.

    <Description>

    Args:
        backbone (str): Backbone CNN name. Defaults to ``"resnet18"``.
        pre_processor (PreProcessor | bool, optional): Pre-processor. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer. Defaults to ``True``.

    Example:
        >>> from anomalib.models import <ModelName>
        >>> model = <ModelName>()
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = <ModelName>Model(backbone=backbone)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step.

        Args:
            batch (Batch): Input batch.

        Returns:
            STEP_OUTPUT: Loss dictionary.
        """
        del args, kwargs
        # ... compute loss
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step.

        Args:
            batch (Batch): Input batch.

        Returns:
            STEP_OUTPUT: Updated batch with predictions.
        """
        del args, kwargs
        predictions = self.model(batch)
        return batch.update(
            pred_score=predictions.pred_score,
            anomaly_map=predictions.anomaly_map,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor.

        Args:
            image_size (tuple[int, int] | None): Target image size. Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor.
        """
        if image_size is not None:
            transform = Compose([Resize(image_size, antialias=True), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return PreProcessor(transform=transform)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return trainer arguments.

        Returns:
            dict[str, Any]: Trainer flags.
        """
        return {}  # or {"max_epochs": 1} for single-pass models

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        Returns:
            LearningType: Learning type enum.
        """
        return LearningType.ONE_CLASS
```

**Rules:**

- Class name: `<ModelName>` (PascalCase, e.g., `Patchcore`, `Supersimplenet`, `EfficientAd`)
- MUST implement: `training_step`, `validation_step`, `trainer_arguments`, `learning_type`
- Use `batch.update()` — NEVER modify batch in-place
- Use `del args, kwargs` in step methods
- Pass `pre_processor`, `post_processor`, `evaluator`, `visualizer` through to `super().__init__`
- For memory-bank models: also inherit from `MemoryBankMixin` and override `fit()`
- For single-pass models: return `{"max_epochs": 1}` from `trainer_arguments`

### 1.4 `__init__.py` — Package Init

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""<ModelName>: <Paper Title>.

<Short description of the model.>

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import <ModelName>
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = <ModelName>()

    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

Paper: https://arxiv.org/<paper_id>
"""

from .lightning_model import <ModelName>

__all__ = ["<ModelName>"]
```

---

## Phase 2: Registration

### 2.1 Register in `src/anomalib/models/image/__init__.py`

Add the import in alphabetical order in both the import block and `__all__`:

```python
from .<model_name> import <ModelName>

__all__ = [
    ...,
    "<ModelName>",
    ...,
]
```

Also add a docstring entry in the module docstring's "Available Models" list:

```python
"""
Available Models:
    ...
    - :class:`<ModelName>`: <Short description>
    ...
"""
```

### 2.2 Register in `src/anomalib/models/__init__.py`

Add the import from `.image`:

```python
from .image import (
    ...,
    <ModelName>,
    ...,
)
```

Add to `__all__` in alphabetical order.

Add docstring entries in both the "Image Models:" and module-level docstring sections:

```python
"""
Image Models:
    ...
    - :class:`<ModelName>` (:class:`anomalib.models.image.<ModelName>`)
    ...
"""
```

---

## Phase 3: Configuration

### 3.1 Model Config

Create `examples/configs/model/<model_name>.yaml`:

```yaml
model:
  class_path: anomalib.models.<ModelName>
  init_args:
    backbone: resnet18
    # ... model-specific hyperparameters
```

### 3.2 Docs Config (mirror)

Create `docs/source/examples/configs/model/<model_name>.yaml` with identical content.

### 3.3 CLI Example Script

Create `docs/source/examples/cli/03_models/<model_name>.sh`:

```bash
#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using <ModelName> Model with Anomalib CLI
# ------------------------------------
# This example shows how to use the <ModelName> model for anomaly detection.

# 1. Basic Usage
echo "Training <ModelName> with default settings..."
anomalib train \
    --model <model_name>

# 2. Custom Configuration
echo -e "\nTraining with custom model settings..."
anomalib train \
    --model <model_name> \
    --model.backbone resnet18 \
    --data.category bottle
```

---

## Phase 4: README with Benchmarks

Create `src/anomalib/models/image/<model_name>/README.md`:

```markdown
# <ModelName>

This is the implementation of [<ModelName>](paper_url).

Model Type: Segmentation

## Description

<2-3 paragraph description of the algorithm. Explain the key idea, how training
works, and how inference/scoring works. Mention the architecture pattern
(memory bank, teacher-student, normalizing flow, etc.).>

## Architecture

![<ModelName> Architecture](/docs/source/images/<model_name>/architecture.png "<ModelName> Architecture")

## Usage

`anomalib train --model <ModelName> --data MVTecAD --data.category <category>`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|            |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ---------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| <backbone> | x.xxx | x.xxx  | x.xxx |  x.xxx  | x.xxx | x.xxx | x.xxx  | x.xxx |  x.xxx  |  x.xxx   |   x.xxx   | x.xxx | x.xxx |   x.xxx    |   x.xxx    | x.xxx  |

### Pixel-Level AUC

|            |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ---------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| <backbone> | x.xxx | x.xxx  | x.xxx |  x.xxx  | x.xxx | x.xxx | x.xxx  | x.xxx |  x.xxx  |  x.xxx   |   x.xxx   | x.xxx | x.xxx |   x.xxx    |   x.xxx    | x.xxx  |

### Image F1 Score

|            |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ---------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| <backbone> | x.xxx | x.xxx  | x.xxx |  x.xxx  | x.xxx | x.xxx | x.xxx  | x.xxx |  x.xxx  |  x.xxx   |   x.xxx   | x.xxx | x.xxx |   x.xxx    |   x.xxx    | x.xxx  |
```

**Benchmark guidelines:**

- Use MVTec AD as the primary benchmark dataset (standard in the field).
- Report Image-Level AUC, Pixel-Level AUC, and Image F1 Score.
- All results with seed `42` unless the model has specific requirements.
- Break down by all 15 MVTec AD categories.
- If benchmarks aren't available yet, use `x.xxx` placeholders and note "TODO: Run benchmarks".
- If the source paper reports numbers, include them as "Paper (reported)" alongside "Anomalib (ours)".
- Optionally include VisA dataset benchmarks.

---

## Phase 5: Tests

Create `tests/unit/models/image/<model_name>/__init__.py` (empty) and `tests/unit/models/image/<model_name>/test_torch_model.py`:

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for <ModelName> torch model."""

import pytest
import torch

from anomalib.models.image.<model_name>.torch_model import <ModelName>Model


class Test<ModelName>Model:
    """Test the <ModelName> torch model."""

    @staticmethod
    def test_initialization_defaults() -> None:
        """Test initialization with default arguments."""
        model = <ModelName>Model()
        assert isinstance(model, <ModelName>Model)

    @staticmethod
    def test_forward_pass() -> None:
        """Test forward pass produces correct output shapes."""
        model = <ModelName>Model()
        model.eval()
        # Use small image size for fast tests
        x = torch.randn(2, 3, 128, 128)
        # NOTE: adapt based on what the forward method expects
        output = model(x)
        assert hasattr(output, "pred_score")
        assert hasattr(output, "anomaly_map")
        assert output.pred_score.shape[0] == 2

    @staticmethod
    def test_training_mode() -> None:
        """Test that model works in training mode."""
        model = <ModelName>Model()
        model.train()
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        assert output is not None
```

**Test conventions:**

- Test file location mirrors source: `tests/unit/models/image/<model_name>/`
- Test class name: `Test<ModelName>Model`
- Use `@pytest.fixture` for shared test data
- Use `@pytest.mark.parametrize` for testing multiple configurations
- Use small tensor sizes (128x128) for fast tests
- Test at minimum: initialization, forward pass, training mode
- Use `@staticmethod` for tests that don't need `self`
- Add `__init__.py` to the test directory

---

## Phase 6: Coding Conventions Checklist

Every file MUST follow these conventions (enforced by pre-commit):

### License Header

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
```

### Docstrings

- Google style (`Args:`, `Returns:`, `Raises:`, `Example:`)
- Every public class and method gets a docstring
- Use `>>>` doctest examples
- Use `# doctest: +SKIP` for examples requiring data/GPU
- Double backticks for defaults in docstrings: `Defaults to ``True``.`
- Cross-reference: `:class:\`anomalib.models.MyModel\``

### Error Handling

```python
# CORRECT — assign to msg first
msg = f"Unsupported backbone: {backbone}."
raise ValueError(msg)

# WRONG — will fail linting
raise ValueError(f"Unsupported backbone: {backbone}.")
```

### Type Hints

- All function signatures fully typed
- Use modern syntax: `list[str]`, `dict[str, int]`, `tuple[int, int]`, `X | Y`
- Use `from __future__ import annotations` sparingly

### Imports

- Three groups: stdlib, third-party, local (separated by blank lines)
- Use `TYPE_CHECKING` blocks for annotation-only imports
- `pathlib.Path` instead of `os.path`
- `logging.getLogger(__name__)` for logger (never `print()`)

### Naming

- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Logger: always named `logger`
- Module files: `snake_case`

### Validation

```bash
# Run before committing
pre-commit run --files <changed_files>
```

---

## Phase 7: Export Compatibility

Ensure the model can be exported to ONNX and OpenVINO:

- Use `ExportableCenterCrop` instead of `CenterCrop` in pre-processor transforms
- Avoid dynamic control flow in the forward pass that can't be traced
- Test export works:

  ```python
  engine.export(model=model, export_type=ExportType.OPENVINO)
  engine.export(model=model, export_type=ExportType.ONNX)
  ```

---

## Phase 8: Smoke Test Verification

After implementation, run these checks to verify the model works end-to-end with the anomalib engine.

### 8.1 Basic Import Check

```python
# Verify registration
from anomalib.models import <ModelName>
model = <ModelName>()
print(f"Model: {model.__class__.__name__}")
print(f"Learning type: {model.learning_type}")
print(f"Trainer arguments: {model.trainer_arguments}")
```

### 8.2 Training Smoke Test

```python
from anomalib.data import MVTecAD
from anomalib.models import <ModelName>
from anomalib.engine import Engine

datamodule = MVTecAD(category="bottle")
model = <ModelName>()
engine = Engine(max_epochs=1, devices=1, logger=False)

# Train for 1 epoch — must complete without errors
engine.fit(model=model, datamodule=datamodule)
print("✓ Training passed")
```

### 8.3 Test (Inference) Smoke Test

```python
# Run test — must produce metrics
engine.test(model=model, datamodule=datamodule)
print("✓ Testing passed")
```

### 8.4 OpenVINO Export Smoke Test

```python
from anomalib.deploy import ExportType

# Export to OpenVINO — must produce model.xml + model.bin
engine.export(model=model, export_type=ExportType.OPENVINO, export_root="./export_test")
print("✓ OpenVINO export passed")
```

### 8.5 Export Verification with Inferencer

```python
import numpy as np
from anomalib.deploy import OpenVINOInferencer

# Load exported model and run prediction
inferencer = OpenVINOInferencer(path="./export_test/model.xml", device="CPU")

# Test with a dummy image
dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result = inferencer.predict(dummy_image)

assert result.pred_score is not None, "pred_score is None"
assert result.anomaly_map is not None, "anomaly_map is None"
print(f"✓ Inferencer passed: score={result.pred_score:.4f}, map_shape={result.anomaly_map.shape}")
```

### 8.6 ONNX Export Smoke Test (Optional)

```python
engine.export(model=model, export_type=ExportType.ONNX, export_root="./export_test_onnx")
print("✓ ONNX export passed")
```

### 8.7 Full Smoke Test Script

Combine all checks into one script that can be run after implementation:

```python
"""Smoke test for <ModelName> model."""

import numpy as np
from anomalib.data import MVTecAD
from anomalib.models import <ModelName>
from anomalib.engine import Engine
from anomalib.deploy import ExportType, OpenVINOInferencer

# Setup
datamodule = MVTecAD(category="bottle")
model = <ModelName>()
engine = Engine(max_epochs=1, devices=1, logger=False)

# 1. Train
engine.fit(model=model, datamodule=datamodule)
print("✓ Step 1/4: Training passed")

# 2. Test
engine.test(model=model, datamodule=datamodule)
print("✓ Step 2/4: Testing passed")

# 3. Export to OpenVINO
engine.export(model=model, export_type=ExportType.OPENVINO, export_root="./smoke_test_export")
print("✓ Step 3/4: Export passed")

# 4. Verify exported model
inferencer = OpenVINOInferencer(path="./smoke_test_export/model.xml", device="CPU")
result = inferencer.predict(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
assert result.pred_score is not None
assert result.anomaly_map is not None
print("✓ Step 4/4: Inferencer passed")

print("\n✓ All smoke tests passed for <ModelName>!")
```

---

## Complete Checklist

Use this checklist to verify everything is done:

- [ ] **Research**: Paper read, source repo understood, algorithm classified (learning type, architecture pattern)
- [ ] **torch_model.py**: Pure PyTorch implementation, returns `InferenceBatch`, all type hints
- [ ] **lightning_model.py**: `AnomalibModule` wrapper, `training_step`, `validation_step`, `trainer_arguments`, `learning_type`, `configure_pre_processor`
- [ ] \***\*init**.py\*\*: Package init with docstring, exports model class
- [ ] **Registration (image)**: Import + `__all__` in `src/anomalib/models/image/__init__.py`
- [ ] **Registration (top)**: Import + `__all__` in `src/anomalib/models/__init__.py`
- [ ] **Config**: `examples/configs/model/<model_name>.yaml`
- [ ] **Docs config**: `docs/source/examples/configs/model/<model_name>.yaml`
- [ ] **CLI example**: `docs/source/examples/cli/03_models/<model_name>.sh`
- [ ] **README.md**: Description, architecture diagram placeholder, usage, benchmark tables
- [ ] **Tests**: `tests/unit/models/image/<model_name>/test_torch_model.py` with init, forward, training tests
- [ ] **Tests `init`**: `tests/unit/models/image/<model_name>/__init__.py`
- [ ] **License headers**: All new files have Intel copyright header
- [ ] **Docstrings**: Google style on all public classes and methods
- [ ] **Type hints**: All function signatures typed
- [ ] **Export**: Model can export to ONNX and OpenVINO without errors
- [ ] **Smoke test (train)**: `engine.fit()` completes for 1 epoch without errors
- [ ] **Smoke test (test)**: `engine.test()` completes and produces metrics
- [ ] **Smoke test (export)**: `engine.export(export_type=ExportType.OPENVINO)` produces `model.xml` + `model.bin`
- [ ] **Smoke test (infer)**: `OpenVINOInferencer.predict()` returns non-None `pred_score` and `anomaly_map`
- [ ] **Pre-commit**: `pre-commit run --files <all_new_files>` passes
- [ ] **Benchmarks**: MVTec AD results filled in (or marked as TODO)
- [ ] **`batch.update()`**: Never modified batch in-place

---

## Reference Implementations

Study these existing models as patterns:

| Pattern                               | Model          | Path                                        |
| ------------------------------------- | -------------- | ------------------------------------------- |
| Simple (Gaussian memory, single-pass) | PaDiM          | `src/anomalib/models/image/padim/`          |
| Memory bank + coreset                 | PatchCore      | `src/anomalib/models/image/patchcore/`      |
| Teacher-student + autoencoder         | EfficientAd    | `src/anomalib/models/image/efficient_ad/`   |
| Discriminative + synthetic anomalies  | SuperSimpleNet | `src/anomalib/models/image/supersimplenet/` |
| Normalizing flows                     | FastFlow       | `src/anomalib/models/image/fastflow/`       |
| Zero-shot (CLIP-based)                | WinCLIP        | `src/anomalib/models/image/winclip/`        |
| Zero-shot (DINO-based)                | AnomalyDINO    | `src/anomalib/models/image/anomaly_dino/`   |
| Reconstruction-based                  | DRAEM          | `src/anomalib/models/image/draem/`          |
| GAN-based                             | GANomaly       | `src/anomalib/models/image/ganomaly/`       |
| Video model                           | AI-VAD         | `src/anomalib/models/video/ai_vad/`         |

---

## Common Mistakes to Avoid

1. **Forgetting registration** — Model won't be discoverable via CLI or `get_model()`.
2. **Using `CenterCrop` instead of `ExportableCenterCrop`** — Breaks ONNX/OpenVINO export.
3. **Modifying batch in-place** — Always use `batch.update(pred_score=..., anomaly_map=...)`.
4. **Not returning `InferenceBatch`** — The `torch_model.forward()` MUST return `InferenceBatch`.
5. **Missing config YAML** — Model can't be used from CLI without it.
6. **Inline exception strings** — Must assign to `msg` variable first (Ruff EM101/EM102).
7. **Wrong channel order from JAX/TF** — JAX/TF typically use NHWC; PyTorch uses NCHW. Transpose accordingly.
8. **Missing `del args, kwargs`** — Required in `training_step` and `validation_step` to satisfy linting.
9. **`print()` instead of `logger`** — Always use `logging.getLogger(__name__)`.
10. **Missing license header** — Every `.py` file needs the Intel copyright header.
