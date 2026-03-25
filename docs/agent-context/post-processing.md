# Post-processing — Agent Context

## PostProcessor (`src/anomalib/post_processing/post_processor.py`)

The PostProcessor component follows the dual-nature pattern (nn.Module + Callback).
It handles score normalization, thresholding, and sensitivity.

### Dual Nature

- **As Callback Hooks**:
  - `on_validation_batch_end`: Updates normalization stats and threshold metrics.
  - `on_validation_epoch_end`: Computes final thresholds.
  - `on_test_batch_end` / `on_predict_batch_end`: Applies normalization and thresholding.
- **As nn.Module forward()**: Takes `InferenceBatch`, performs normalization and thresholding, and returns the updated `InferenceBatch`.

### Integration & Export

- **OneClassPostProcessor**: Specialized implementation for one-class models in `post_processing/one_class.py`.
- **Export Compatibility**: The `forward()` method is traced into the model graph. Exported models produce normalized and thresholded outputs directly.
- **Key Properties**: `image_threshold`, `pixel_threshold`.

## Key Files

- `src/anomalib/post_processing/post_processor.py`
- `src/anomalib/post_processing/one_class.py`
- `src/anomalib/post_processing/utils/core.py`
