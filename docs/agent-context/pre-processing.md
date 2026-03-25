# Pre-processing — Agent Context

## PreProcessor (`src/anomalib/pre_processing/pre_processor.py`)

The PreProcessor component follows the dual-nature pattern (nn.Module + Callback).
It manages image and mask transformations for both training and inference.

### Dual Nature

- **As Callback**: Applies `self.transform` during `on_*_batch_start` hooks.
- **As nn.Module**: The `forward()` method applies `export_transform` during model export.

### Export Transformation

Built via `get_exportable_transform()` to ensure compatibility with ONNX and OpenVINO:

- Replaces standard `CenterCrop` with `ExportableCenterCrop`.
- Disables antialiasing in `Resize` operations.

### Integration

- Models define default transforms via the `configure_pre_processor()` static method.
- Transforms are applied to `batch.image` and `batch.gt_mask`.
- Ensures consistency between training-time and deployment-time data preparation.

## Key Files

- `src/anomalib/pre_processing/pre_processor.py`
- `src/anomalib/pre_processing/utils/transform.py`
- `src/anomalib/pre_processing/utils/core.py`
