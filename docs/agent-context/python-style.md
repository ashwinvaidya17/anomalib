# Python Coding Conventions — Agent Context

Extracted from anomalib source code, `pyproject.toml`, and `.pre-commit-config.yaml`.

## Python Version

Minimum: **3.10** (`requires-python = ">=3.10"`). Use modern syntax: `X | Y` unions, `list[str]` builtins, `match` statements.

## License Header

Every `.py` file starts with a copyright header. The year range ends at the year of last modification.

```python
# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
```

Rules:

- Single start year if file was created and last modified in the same year: `# Copyright (C) 2025 Intel Corporation`
- Year range if modified across years: `# Copyright (C) 2022-2025 Intel Corporation`
- Update the end year when modifying an existing file.
- Enforced by Ruff rule `CPY001` via `[tool.ruff.lint.flake8-copyright]`.

## Linting & Formatting

- **Formatter**: Ruff (replaces Black + isort)
- **Linter**: Ruff with 40+ rule sets enabled
- **Type checker**: mypy
- **Security**: bandit (medium severity, high confidence)
- **Pre-commit hooks**: ruff, ruff-format, mypy, bandit, trailing-whitespace, end-of-file-fixer, markdownlint, prettier
- **Line length**: 120 characters

```bash
# Run manually
ruff format .
ruff check . --fix
mypy src/anomalib/

# Or via pre-commit
pre-commit run --files <file1> [file2 ...]
```

## Docstrings

**Google style** (`convention = "google"` in `[tool.ruff.lint.pydocstyle]`).

Every public module, class, and method requires a docstring. `__init__` methods are exempt (`D107` is ignored).

### Module-level docstring

```python
"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

This module implements the PatchCore model for anomaly detection using a memory bank
of patch features extracted from a pretrained CNN backbone.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Patchcore
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Patchcore(backbone="wide_resnet50_2")

    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

Paper: https://arxiv.org/abs/2106.08265

See Also:
    - :class:`anomalib.models.image.patchcore.torch_model.PatchcoreModel`:
        PyTorch implementation of the PatchCore model architecture
"""
```

### Class docstring

```python
class Patchcore(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning Module for anomaly detection.

    This class implements the PatchCore algorithm which uses a memory bank of patch
    features for anomaly detection.

    Args:
        backbone (str): Name of the backbone CNN network.
            Defaults to ``"wide_resnet50_2"``.
        layers (Sequence[str]): Names of layers to extract features from.
            Defaults to ``("layer2", "layer3")``.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        coreset_sampling_ratio (float, optional): Ratio for coreset sampling.
            Defaults to ``0.1``.

    Example:
        >>> from anomalib.models import Patchcore
        >>> model = Patchcore(backbone="wide_resnet50_2")
    """
```

### Method docstring

```python
def predict(self, image: np.ndarray) -> ImageResult:
    """Run inference on a single image.

    Args:
        image (np.ndarray): Input image as numpy array, shape (H, W, C).

    Returns:
        ImageResult: Prediction containing pred_score, pred_label, anomaly_map.

    Raises:
        ValueError: If image shape is invalid.

    Example:
        >>> result = inferencer.predict("path/to/image.jpg")
        >>> result.pred_score
        0.86
    """
```

Key conventions:

- Args format: `param_name (type): Description. Defaults to ``value``.`
- Use double backticks for default values: `Defaults to ``True``.`
- Use `>>>` doctest-style examples.
- Add `# doctest: +SKIP` for examples that can't actually run in tests.
- Cross-reference classes with `:class:\`anomalib.module.ClassName\``.

## Type Annotations

All function signatures must be fully typed. Return types are always specified (use `-> None` for procedures).

```python
# Modern syntax (preferred — Python 3.10+)
def process(self, path: str | Path, size: tuple[int, int] | None = None) -> list[str]:
    ...

# Use lowercase builtins for generics
items: list[str]           # NOT List[str]
mapping: dict[str, int]    # NOT Dict[str, int]
coords: tuple[int, int]    # NOT Tuple[int, int]
```

- `*args` and `**kwargs` are exempt from type annotations (rules `ANN002`, `ANN003` ignored).
- `from __future__ import annotations` is used sparingly — only in files that need forward references.

### TYPE_CHECKING Pattern

Use `TYPE_CHECKING` blocks to avoid circular imports and runtime overhead for annotation-only imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from anomalib import TaskType
    from anomalib.data.datasets.base.image import AnomalibDataset
```

## Import Order

Three groups separated by blank lines, enforced by Ruff isort (`I` rules):

```python
# 1. Standard library
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# 2. Third-party
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

# 3. Local (anomalib)
from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models.components import AnomalibModule

# 4. Relative imports (same package)
from .torch_model import PatchcoreModel
```

## Logging

Use the standard library `logging` module with `__name__`:

```python
import logging

logger = logging.getLogger(__name__)

# Usage
logger.info("Training started for %s", model_name)
logger.warning("No anomalous images found in validation set.")
logger.debug("Feature shape: %s", features.shape)
```

Never use `print()` for status messages. Always use the logger.

## Error Handling

### Raising exceptions

Always assign the error message to a variable `msg` first. This is enforced by Ruff rules `EM101`/`EM102` (no inline string literals in exception calls):

```python
# CORRECT
msg = f"Unsupported precision type: {precision}."
raise ValueError(msg)

msg = "Pipeline is not available"
raise RuntimeError(msg)

# WRONG — will fail linting
raise ValueError(f"Unsupported precision type: {precision}.")
raise RuntimeError("Pipeline is not available")
```

### Multi-line error messages

```python
msg = (
    f"The part {part} is selected, "
    f"but it is absent in order_of_parts={order_of_parts}"
)
raise ValueError(msg)
```

### Testing exceptions

```python
with pytest.raises(FileNotFoundError):
    AnomalibModule.from_config(config_path="wrong_configs.yaml")

with pytest.raises(RuntimeError, match=r"Training with anomalous samples"):
    model.forward(images=images, labels=labels)
```

## Path Handling

Always use `pathlib.Path`, never `os.path`. Enforced by Ruff rule `PTH` (flake8-use-pathlib):

```python
from pathlib import Path

config_path = Path("examples/configs/model") / f"{model_name}.yaml"
for img_path in Path("test_set/").glob("*.jpg"):
    ...
```

## Naming Conventions

- **Classes**: `PascalCase` — `AnomalibModule`, `PatchcoreModel`, `ImageBatch`
- **Functions/methods**: `snake_case` — `configure_pre_processor`, `training_step`
- **Constants**: `UPPER_SNAKE_CASE` — `STEP_OUTPUT`, `EVAL_DATALOADERS`
- **Private methods**: single underscore prefix — `_setup_anomalib_callbacks`, `_resolve_component`
- **Unused variables**: underscore prefix — `_`, `_unused`
- **Module-level logger**: always `logger` (not `log`, not `LOG`)

## `__all__` Exports

Every `__init__.py` should define `__all__` listing public API:

```python
__all__ = [
    "AnomalibModule",
    "MemoryBankMixin",
    "Patchcore",
]
```

## Testing

- **Framework**: pytest with `--strict-markers` and `--strict-config`
- **Location**: `tests/` mirrors `src/anomalib/` structure
- **Naming**: Test files: `test_*.py`. Test classes: `TestClassName`. Test functions: `test_description`.
- **Fixtures**: Shared fixtures in `tests/conftest.py` (session-scoped). Module-specific fixtures inline.
- **Markers**: `@pytest.mark.gpu` for GPU-only tests, `@pytest.mark.cpu` for CPU-only.
- **Parametrize**: Use `@pytest.mark.parametrize` for testing multiple models/configs.
- **Assertions**: Plain `assert` statements (rule `S101` / bandit `B101` is disabled).

```python
class TestAnomalibModule:
    """Test AnomalibModule."""

    @pytest.fixture(autouse=True)
    def setup(self, model_config_folder_path: str) -> None:
        """Setup test AnomalibModule."""
        self.model_config_folder_path = model_config_folder_path

    @pytest.mark.parametrize("model_name", ["padim", "patchcore", "stfpm"])
    def test_from_config(self, model_name: str) -> None:
        """Test AnomalibModule.from_config."""
        config_path = Path(self.model_config_folder_path) / f"{model_name}.yaml"
        model = AnomalibModule.from_config(config_path=config_path)
        assert isinstance(model, AnomalibModule)
```

## Ruff Configuration Summary

From `pyproject.toml`:

- **Target**: Python 3.10 (`target-version = "py310"`)
- **Line length**: 120
- **Source roots**: `src`, `tests`
- **Docstring convention**: Google
- **Max complexity**: 15 (mccabe)
- **Key ignored rules**: `D107` (missing `__init__` docstring), `ANN002`/`ANN003` (missing `*args`/`**kwargs` annotations), `S101` (assert), `FBT001`/`FBT002` (boolean args)
- **Fixable**: All rules are auto-fixable when running `ruff check --fix`
