---
name: anomalib-add-dataset
description: Use when adding a new dataset or dataloader to anomalib. Covers analyzing folder structures, creating DataModule + Dataset classes, implementing automatic downloads, writing data configs, registering in the data system, writing tests, and running smoke tests. Also handles pointing to external data sources (websites, APIs) and custom folder-based datasets. Triggers on phrases like "add dataset", "add dataloader", "new dataset", "create datamodule", "import dataset", "load data from", "here is my folder".
---

# Adding a New Dataset/Dataloader to Anomalib

This skill guides the full lifecycle of adding a new dataset to anomalib. The source may be a public benchmark (with download URL), a user's local folder structure, or a remote data source.

**Before starting, read these context files:**

- `docs/agent-context/data.md` — Data system architecture, base classes, pipeline
- `docs/agent-context/index.md` — General architecture overview
- `docs/agent-context/python-style.md` — Coding conventions, docstrings, linting

---

## Phase 0: Analyze the Data Source

Before writing any code, understand the data structure.

### 0.1 Determine the Source Type

| Source                                 | Approach                                                             |
| -------------------------------------- | -------------------------------------------------------------------- |
| Public benchmark with download URL     | Full DataModule + Dataset with `prepare_data()` auto-download        |
| User's local folder                    | Check if `Folder` DataModule works; if not, create custom DataModule |
| Remote API / web scraping              | Create custom `prepare_data()` with download logic                   |
| Existing format (MVTec-like structure) | May only need thin wrapper around existing base classes              |

### 0.2 Analyze Folder Structure

Map the data layout. The key question: **where are normal images, abnormal images, and masks?**

```text
# Common patterns:

# Pattern A: MVTec-like (train/test split with categories)
dataset_root/
├── category_1/
│   ├── train/
│   │   └── good/          # Normal images only
│   └── test/
│       ├── good/          # Normal test images
│       ├── defect_type_1/ # Anomalous images
│       └── defect_type_2/
├── category_2/
│   └── ...

# Pattern B: Flat with labels
dataset_root/
├── images/
│   ├── normal_001.png
│   └── anomaly_001.png
├── masks/                  # Optional segmentation masks
│   └── anomaly_001.png
└── labels.csv              # Label mapping

# Pattern C: Single normal class (unsupervised)
dataset_root/
├── train/
│   └── *.png              # All normal
└── test/
    └── *.png              # Mixed normal + anomalous (labels from filename/metadata)
```

### 0.3 Identify Key Properties

Document these before implementation:

- **Modality**: Image, Depth (RGB-D), or Video
- **Has categories?** e.g., MVTec has bottle, cable, etc.
- **Split strategy**: Pre-split (from_dir) or needs automatic splitting
- **Has ground truth masks?** (segmentation task support)
- **Image format**: PNG, JPG, BMP, TIFF, etc.
- **Download source**: Direct URL, Kaggle, academic site, etc.
- **License**: Verify redistribution rights
- **File hash**: For download verification (SHA256 preferred)

---

## Phase 1: Implementation

### 1.1 Directory Structure

Create two files — one for the Dataset class, one for the DataModule class:

```text
src/anomalib/data/datasets/image/<dataset_name>.py     # Dataset class
src/anomalib/data/datamodules/image/<dataset_name>.py   # DataModule class
```

For video datasets, use `video/` instead of `image/`.

### 1.2 Dataset Class — `src/anomalib/data/datasets/image/<dataset_name>.py`

The Dataset class handles sample scanning and DataFrame construction.

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""<DatasetName> dataset.

<Brief description of the dataset, where it comes from, and what it contains.>

Example:
    >>> from anomalib.data.datasets.image.<dataset_name> import <DatasetName>Dataset
    >>> dataset = <DatasetName>Dataset(root="./datasets/<dataset_name>")

Reference:
    <Citation or URL to the dataset source.>
"""

import logging
from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset

logger = logging.getLogger(__name__)


def make_<dataset_name>_dataset(
    root: str | Path,
    split: str | None = None,
    # ... dataset-specific params (category, etc.)
) -> pd.DataFrame:
    """Create <DatasetName> samples DataFrame.

    Args:
        root: Path to the dataset root directory.
        split: Dataset split (``"train"``, ``"test"``, or ``None`` for all).

    Returns:
        DataFrame with columns: ``image_path``, ``split``, ``label_index``,
        ``label``, ``mask_path``.

    Example:
        >>> samples = make_<dataset_name>_dataset(root="./datasets/<dataset_name>")
        >>> samples.head()  # doctest: +SKIP

    Raises:
        FileNotFoundError: If the root directory does not exist.
    """
    root = Path(root)

    if not root.is_dir():
        msg = f"Dataset root not found: {root}"
        raise FileNotFoundError(msg)

    # Scan directories and build DataFrame
    samples_list = []

    # ... scan logic specific to this dataset's folder structure ...
    # For each image found:
    # samples_list.append({
    #     "image_path": str(image_path),
    #     "split": "train" or "test",
    #     "label_index": 0 (normal) or 1 (anomalous),
    #     "label": "good" or defect_type_name,
    #     "mask_path": str(mask_path) or "",
    # })

    samples = pd.DataFrame(samples_list)

    # Set task attribute based on whether masks exist
    if samples["mask_path"].any():
        samples.attrs["task"] = "segmentation"
    else:
        samples.attrs["task"] = "classification"

    # Filter by split if specified
    if split:
        samples = samples[samples["split"] == split]
        samples = samples.reset_index(drop=True)

    return samples


class <DatasetName>Dataset(AnomalibDataset):
    """<DatasetName> dataset class.

    Args:
        root (str | Path): Root directory of the dataset.
        split (str): Dataset split (``"train"`` or ``"test"``).
        augmentations (Transform | None): Augmentations to apply.
            Defaults to ``None``.

    Example:
        >>> dataset = <DatasetName>Dataset(root="./datasets/<dataset_name>", split="train")
        >>> dataset[0].image.shape  # doctest: +SKIP
        torch.Size([3, H, W])
    """

    def __init__(
        self,
        root: str | Path = "./datasets/<dataset_name>",
        split: str = "train",
        augmentations: Transform | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.root = Path(root)
        self.split = split
        self.samples = make_<dataset_name>_dataset(root=self.root, split=self.split)
```

**Rules:**

- Class name: `<DatasetName>Dataset` (e.g., `MVTecADDataset`, `KolektorDataset`)
- The `make_<dataset_name>_dataset()` function MUST return a pandas DataFrame with at minimum: `image_path`, `split`, `label_index`
- Set `samples.attrs["task"]` to `"classification"` or `"segmentation"`
- Use `label_index`: `0` = normal, `1` = anomalous
- The `mask_path` column should be `""` (empty string) when no mask exists
- All type hints required on function signatures

### 1.3 DataModule Class — `src/anomalib/data/datamodules/image/<dataset_name>.py`

The DataModule handles downloading, splitting, and dataloader creation.

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""<DatasetName> DataModule.

<Brief description of the dataset and its use for anomaly detection.>

Example:
    >>> from anomalib.data import <DatasetName>
    >>> datamodule = <DatasetName>()
    >>> datamodule.setup()  # doctest: +SKIP

Reference:
    <Citation or URL.>
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.<dataset_name> import <DatasetName>Dataset
from anomalib.data.utils import DownloadInfo, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="<dataset_name>",
    url="<download_url>",
    hashsum="<sha256_hash>",
    filename="<archive_filename>",  # optional, inferred from URL if not set
)


class <DatasetName>(AnomalibDataModule):
    """<DatasetName> DataModule for anomaly detection.

    Args:
        root (str | Path): Root directory for the dataset.
            Defaults to ``"./datasets/<dataset_name>"``.
        category (str): Sub-category of the dataset.
            Defaults to ``"<default_category>"``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Evaluation batch size.
            Defaults to ``32``.
        num_workers (int): Number of data loading workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Training augmentations.
            Defaults to ``None``.
        val_augmentations (Transform | None): Validation augmentations.
            Defaults to ``None``.
        test_augmentations (Transform | None): Test augmentations.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations applied to all splits.
            Defaults to ``None``.
        test_split_mode (TestSplitMode | str): How test data is split.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode | str): How validation data is split.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for splitting.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data import <DatasetName>
        >>> datamodule = <DatasetName>(root="./datasets/<dataset_name>")
        >>> datamodule.setup()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str | Path = "./datasets/<dataset_name>",
        category: str = "<default_category>",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up train/test datasets.

        Args:
            _stage: Lightning stage (unused, kept for API compatibility).
        """
        self.train_data = <DatasetName>Dataset(
            root=self.root / self.category,
            split="train",
            augmentations=self.train_augmentations,
        )
        self.test_data = <DatasetName>Dataset(
            root=self.root / self.category,
            split="test",
            augmentations=self.test_augmentations,
        )

    def prepare_data(self) -> None:
        """Download the dataset if it doesn't exist.

        Called automatically by Lightning before setup. Downloads once,
        even in distributed training.
        """
        if (self.root / self.category).is_dir():
            logger.info("Found existing dataset at %s", self.root / self.category)
            return
        download_and_extract(self.root, DOWNLOAD_INFO)
```

**Rules:**

- Class name: `<DatasetName>` (PascalCase, e.g., `MVTecAD`, `BTech`, `Kolektor`)
- MUST implement `_setup()` — creates `self.train_data` and `self.test_data`
- MUST implement `prepare_data()` if dataset has a download source
- The base class handles val/test splitting automatically — don't re-implement
- The base class provides `train_dataloader()`, `val_dataloader()`, `test_dataloader()` — don't override
- If the dataset has no categories, remove the `category` parameter
- If the dataset has no download URL, omit `DOWNLOAD_INFO` and `prepare_data()`

### 1.4 Download Mechanism (for downloadable datasets)

Use the `DownloadInfo` dataclass and `download_and_extract` utility:

```python
from anomalib.data.utils import DownloadInfo, download_and_extract

DOWNLOAD_INFO = DownloadInfo(
    name="dataset_name",
    url="https://example.com/dataset.zip",
    hashsum="sha256_hash_of_the_file",
    filename="dataset.zip",  # Optional — inferred from URL if omitted
)
```

**To get the SHA256 hash:**

```bash
sha256sum downloaded_file.zip
```

**`download_and_extract()` behavior:**

- Downloads with a progress bar
- Verifies the SHA256 hash
- Extracts `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, `.tar.xz` archives
- Supports Google Drive URLs (gdrive format)
- Extracts to `root` directory

### 1.5 Handling Custom Folder Structures

If the user says "here is my folder", first check if `Folder` DataModule works:

```python
from anomalib.data import Folder

datamodule = Folder(
    name="my_dataset",
    root="./path/to/data",
    normal_dir="good",
    abnormal_dir="defect",
    mask_dir="ground_truth",
    task="segmentation",
)
```

Only create a custom DataModule when `Folder` can't handle the structure (e.g., non-standard splits, metadata files, category hierarchies, depth data).

---

## Phase 2: Registration

Register the new dataset in **4 files** (all in alphabetical order):

### 2.1 Dataset Registry — `src/anomalib/data/datasets/image/__init__.py`

```python
from .<dataset_name> import <DatasetName>Dataset

__all__ = [
    ...,
    "<DatasetName>Dataset",
    ...,
]
```

### 2.2 DataModule Registry — `src/anomalib/data/datamodules/image/__init__.py`

Two things to add:

**A) Import + `__all__`:**

```python
from .<dataset_name> import <DatasetName>

__all__ = [
    ...,
    "<DatasetName>",
    ...,
]
```

**B) Add to `ImageDataFormat` enum:**

```python
class ImageDataFormat(str, Enum):
    ...
    <DATASET_NAME> = "<dataset_name>"
    ...
```

This enum is used by the test infrastructure and the CLI to discover datasets.

### 2.3 Dataset Re-export — `src/anomalib/data/datasets/__init__.py`

```python
from .image import <DatasetName>Dataset

__all__ = [
    ...,
    "<DatasetName>Dataset",
    ...,
]
```

### 2.4 Top-Level Data Package — `src/anomalib/data/__init__.py`

```python
from .datamodules.image import <DatasetName>
from .datasets.image import <DatasetName>Dataset

__all__ = [
    ...,
    "<DatasetName>",
    "<DatasetName>Dataset",
    ...,
]
```

---

## Phase 3: Configuration

### 3.1 Data Config YAML

Create `examples/configs/data/<dataset_name>.yaml`:

```yaml
class_path: anomalib.data.<DatasetName>
init_args:
  root: ./datasets/<dataset_name>
  category: <default_category> # Omit if no categories
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 8
  test_split_mode: from_dir
  test_split_ratio: 0.2
  val_split_mode: same_as_test
  val_split_ratio: 0.5
  seed: null
```

### 3.2 Docs Config (mirror)

Create `docs/source/examples/configs/data/<dataset_name>.yaml` with identical content.

---

## Phase 4: Tests

### 4.1 Test File

Create `tests/unit/data/datamodule/image/test_<dataset_name>.py`:

```python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for <DatasetName> DataModule."""

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import <DatasetName>

from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestDataset(_TestAnomalibImageDatamodule):
    """Test <DatasetName> DataModule."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path) -> <DatasetName>:
        """Create <DatasetName> DataModule fixture.

        Args:
            dataset_path: Path to the dummy test dataset.

        Returns:
            Configured <DatasetName> DataModule instance.
        """
        return <DatasetName>(
            root=dataset_path / "<dataset_name>",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )

    @pytest.fixture()
    def config_path(self) -> str:
        """Return path to dataset config.

        Returns:
            Config file path string.
        """
        return "examples/configs/data/<dataset_name>.yaml"
```

**Test conventions:**

- Inherit from `_TestAnomalibImageDatamodule` — this provides standard tests for batch shapes, dataloaders, splits, and `from_config`
- `dataset_path` fixture comes from `tests/conftest.py` — it uses `DummyImageDatasetGenerator` to generate synthetic data
- Use `batch_size=4`, `Resize((256, 256))` augmentations (convention)
- The `category="dummy"` matches the generated test data

### 4.2 Update Test Helpers

For the `DummyImageDatasetGenerator` to create test data for the new dataset, update `tests/helpers/data.py`:

1. Add the dataset format to the helper's dataset generation logic
2. Ensure the generated folder structure matches what the Dataset class expects
3. The `ImageDataFormat` enum addition (Phase 2.2B) enables the test infrastructure to auto-generate dummy data

### 4.3 Create Test `__init__.py`

If a new directory was created for the tests, add an empty `__init__.py`:

```text
tests/unit/data/datamodule/image/__init__.py  # Should already exist
```

---

## Phase 5: Smoke Test Verification

After implementation, run these checks to verify everything works end-to-end.

### 5.1 Dataset Samples Check

```python
from anomalib.data.datasets.image.<dataset_name> import make_<dataset_name>_dataset

# Verify the DataFrame structure
samples = make_<dataset_name>_dataset(root="./datasets/<dataset_name>", split="train")
print(f"Columns: {samples.columns.tolist()}")
print(f"Splits: {samples['split'].unique()}")
print(f"Labels: {samples['label_index'].unique()}")
print(f"Task: {samples.attrs.get('task', 'NOT SET')}")
print(f"Num samples: {len(samples)}")

# Must have: image_path, split, label_index
assert "image_path" in samples.columns
assert "split" in samples.columns
assert "label_index" in samples.columns
```

### 5.2 DataModule Dataloader Check

```python
from anomalib.data import <DatasetName>

datamodule = <DatasetName>(root="./datasets/<dataset_name>")
datamodule.setup()

# Check dataloaders
train_dl = datamodule.train_dataloader()
batch = next(iter(train_dl))
print(f"Batch image shape: {batch.image.shape}")  # [N, C, H, W]
print(f"Batch label shape: {batch.gt_label.shape}")  # [N]
```

### 5.3 Full Training Smoke Test

```python
from anomalib.data import <DatasetName>
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

# 1. Train for 1 epoch
datamodule = <DatasetName>(root="./datasets/<dataset_name>")
model = Patchcore()
engine = Engine(max_epochs=1, devices=1)
engine.fit(model=model, datamodule=datamodule)

# 2. Test
engine.test(model=model, datamodule=datamodule)

# 3. Export to OpenVINO
engine.export(model=model, export_type=ExportType.OPENVINO)

print("Smoke test passed!")
```

### 5.4 Run Unit Tests

```bash
pytest tests/unit/data/datamodule/image/test_<dataset_name>.py -v
```

### 5.5 Pre-commit

```bash
pre-commit run --files \
    src/anomalib/data/datasets/image/<dataset_name>.py \
    src/anomalib/data/datamodules/image/<dataset_name>.py \
    src/anomalib/data/datasets/image/__init__.py \
    src/anomalib/data/datamodules/image/__init__.py \
    src/anomalib/data/datasets/__init__.py \
    src/anomalib/data/__init__.py \
    examples/configs/data/<dataset_name>.yaml \
    tests/unit/data/datamodule/image/test_<dataset_name>.py
```

---

## Complete Checklist

- [ ] **Analysis**: Folder structure understood, modality identified, download source documented
- [ ] **Dataset class**: `src/anomalib/data/datasets/image/<dataset_name>.py` with `make_*_dataset()` + `<DatasetName>Dataset`
- [ ] **DataModule class**: `src/anomalib/data/datamodules/image/<dataset_name>.py` with `_setup()`, `prepare_data()`
- [ ] **Download**: `DownloadInfo` with URL + SHA256 hash (if downloadable)
- [ ] **Registration (datasets/image)**: Import + `__all__` in `src/anomalib/data/datasets/image/__init__.py`
- [ ] **Registration (datamodules/image)**: Import + `__all__` + `ImageDataFormat` enum in `src/anomalib/data/datamodules/image/__init__.py`
- [ ] **Registration (datasets)**: Re-export in `src/anomalib/data/datasets/__init__.py`
- [ ] **Registration (data)**: Import both in `src/anomalib/data/__init__.py` + `__all__`
- [ ] **Config**: `examples/configs/data/<dataset_name>.yaml`
- [ ] **Docs config**: `docs/source/examples/configs/data/<dataset_name>.yaml`
- [ ] **Tests**: `tests/unit/data/datamodule/image/test_<dataset_name>.py` inheriting from `_TestAnomalibImageDatamodule`
- [ ] **Test helpers**: Updated `tests/helpers/data.py` for dummy data generation
- [ ] **License headers**: All new files have Intel copyright header
- [ ] **Docstrings**: Google style on all public classes and functions
- [ ] **Type hints**: All function signatures typed
- [ ] **DataFrame**: Has `image_path`, `split`, `label_index` columns; `samples.attrs["task"]` set
- [ ] **Smoke test**: DataModule loads, dataloaders produce correct batch shapes
- [ ] **Training smoke test**: Trains for 1 epoch with Engine, exports to OpenVINO
- [ ] **Unit tests pass**: `pytest tests/unit/data/datamodule/image/test_<dataset_name>.py -v`
- [ ] **Pre-commit passes**: All new/modified files pass linting

---

## Reference Implementations

Study these existing datasets as patterns:

| Pattern                              | Dataset  | Path                                                                     |
| ------------------------------------ | -------- | ------------------------------------------------------------------------ |
| Standard benchmark with categories   | MVTecAD  | `data/datasets/image/mvtecad.py` + `data/datamodules/image/mvtecad.py`   |
| Simple benchmark, single split logic | Kolektor | `data/datasets/image/kolektor.py` + `data/datamodules/image/kolektor.py` |
| Custom folder structure              | Folder   | `data/datasets/image/folder.py` + `data/datamodules/image/folder.py`     |
| Depth data (RGB-D)                   | MVTec 3D | `data/datasets/image/mvtec_3d.py` + `data/datamodules/image/mvtec_3d.py` |
| Video data                           | Avenue   | `data/datasets/video/avenue.py` + `data/datamodules/video/avenue.py`     |

---

## Common Mistakes to Avoid

1. **Forgetting `samples.attrs["task"]`** — Must be set to `"classification"` or `"segmentation"`. Without it, downstream metrics and evaluation fail.
2. **Wrong `label_index` values** — `0` = normal, `1` = anomalous. Swapping these inverts all predictions.
3. **Missing registration in `ImageDataFormat` enum** — Test infrastructure won't generate dummy data.
4. **Forgetting registration in ALL 4 `__init__.py` files** — Dataset won't be discoverable from the top-level `anomalib.data` namespace.
5. **Empty `mask_path` as `None` instead of `""`** — Use empty string `""` for missing masks, not `None`.
6. **Not resetting DataFrame index** — After filtering by split, call `samples.reset_index(drop=True)`.
7. **Hardcoding absolute paths** — Always use `pathlib.Path` and relative paths from root.
8. **`prepare_data()` downloading every time** — Check `if (self.root / self.category).is_dir()` first.
9. **Inline exception strings** — Must assign to `msg` variable first (Ruff EM101/EM102).
10. **Missing `del args, kwargs`** — Not needed in DataModule (only in model step methods), but keep methods clean.
