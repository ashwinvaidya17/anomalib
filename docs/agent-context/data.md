# Data System — Agent Context

The Anomalib data system handles image, depth, and video data for anomaly detection. It uses a structured pipeline from disk to model, leveraging PyTorch Lightning DataModules and custom dataclasses for type safety and consistency.

## Data Pipeline Overview

The end-to-end data flow follows this sequence:

```bash
Disk (images/masks) -> Dataset (__getitem__ -> ImageItem) -> DataModule (DataLoader) -> ImageBatch -> Model
```

- Datasets scan directories to build a samples DataFrame. This includes image_path, split, label, label_index, and mask_path.
- `__getitem__` calls `read_image(path, as_tensor=True)`. It optionally calls `read_mask()`, applies augmentations, and returns an `ImageItem`.
- DataModule wraps datasets into train, val, and test DataLoaders. It uses `collate_fn=ImageBatch.collate` to create batches.
- Prediction uses `PredictDataset` from `data/predict.py`. This dataset accepts a single file or a directory path for inference.

## Available Datasets

| Dataset      | Modality | Module Path                           |
| ------------ | -------- | ------------------------------------- |
| MVTecAD      | Image    | `data.datamodules.image.mvtecad`      |
| MVTec LOCO   | Image    | `data.datamodules.image.mvtec_loco`   |
| MVTec 3D     | Depth    | `data.datamodules.image.mvtec_3d`     |
| BTech        | Image    | `data.datamodules.image.btech`        |
| Visa         | Image    | `data.datamodules.image.visa`         |
| Folder       | Image    | `data.datamodules.image.folder`       |
| Folder3D     | Depth    | `data.datamodules.image.folder_3d`    |
| Avenue       | Video    | `data.datamodules.video.avenue`       |
| ShanghaiTech | Video    | `data.datamodules.video.shanghaitech` |
| UCSDped      | Video    | `data.datamodules.video.ucsd_ped`     |

## Dataclass Hierarchy

Anomalib uses specialized dataclasses defined in `data/dataclasses/`.

Generic fields in `data/dataclasses/generic.py`:

- `_InputFields`: Includes image, gt_label, gt_mask, and mask_path.
- `_ImageInputFields`: Adds image_path to the input fields.
- `_OutputFields`: Includes anomaly_map, pred_score, pred_mask, pred_label, and explanation.

Torch implementations in `data/dataclasses/torch/image.py`:

- `ImageItem`: Represents a single sample. It contains the image (C,H,W tensor), gt_label, gt_mask, image_path, and mask_path.
- `ImageBatch`: Represents a batched set of samples. It contains the image (N,C,H,W), gt_label (N,), gt_mask (N,H,W), and other fields.
- `InferenceBatch`: A NamedTuple containing pred_score, pred_label, anomaly_map, and pred_mask.

Use `batch.update(pred_score=..., anomaly_map=...)` to attach model predictions. Never modify these objects in-place.

## Using Custom Data (Folder DataModule)

The `Folder` DataModule allows using custom datasets without writing new classes.

```python
from anomalib.data import Folder

datamodule = Folder(
    name="my_dataset",
    root="./datasets/my_data",
    normal_dir="good",          # Directory with normal images (required)
    abnormal_dir="defect",      # Directory with anomalous images (optional)
    mask_dir="masks",           # Directory with segmentation masks (optional)
    task="segmentation",        # Set to "classification" or "segmentation"
)
```

Expected directory structure:

```bash
datasets/my_data/
├── good/           # Normal images
│   ├── 001.png
│   └── ...
├── defect/         # Anomalous images
│   ├── 001.png
│   └── ...
└── masks/          # Segmentation masks (match names in defect/)
    ├── 001.png
    └── ...
```

## Transforms

Augmentations are passed to the DataModule as `train_augmentations`, `val_augmentations`, and `test_augmentations`.

- Use torchvision v2 transforms for image processing.
- Use `ExportableCenterCrop` instead of the standard `CenterCrop` to ensure export compatibility.
- `MultiRandomChoice` randomly applies a specified number of transforms from a list.
- `extract_transforms_by_type(transform, Resize)` helps find specific transforms within composed chains.
- Find transform utilities and custom implementations in `data/transforms/`.

## Key Files

- **Base DataModule**: `src/anomalib/data/datamodules/base/image.py` (AnomalibDataModule)
- **Base Dataset**: `src/anomalib/data/datasets/base/image.py` (AnomalibDataset)
- **Dataclasses**: `src/anomalib/data/dataclasses/torch/image.py` (ImageItem, ImageBatch)
- **Generic Fields**: `src/anomalib/data/dataclasses/generic.py`
- **Folder Data**: `src/anomalib/data/datamodules/image/folder.py`
- **Image I/O**: `src/anomalib/data/utils/image.py` (read_image, read_mask, save_image)
- **Predict**: `src/anomalib/data/predict.py` (PredictDataset)
- **Validators**: `src/anomalib/data/validators/torch/image.py`
