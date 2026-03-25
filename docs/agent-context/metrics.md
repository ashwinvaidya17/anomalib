# Metrics — Agent Context

## Evaluator (`src/anomalib/metrics/evaluator.py`)

The Evaluator follows the dual-nature pattern (nn.Module + Callback).
It manages validation and testing metrics via `val_metrics` and `test_metrics` (ModuleList).

### Lightning Hooks

- `on_validation_epoch_end`: Computes and logs metrics at the end of validation.
- `on_test_epoch_end`: Computes and logs metrics at the end of testing.

### Configuration

Registered automatically via `AnomalibModule.configure_evaluator()`.
Default evaluator uses: image_AUROC, image_F1Score, pixel_AUROC, pixel_F1Score.

## Available Metrics

| Metric              | File                   | Level         | Description                   |
| ------------------- | ---------------------- | ------------- | ----------------------------- |
| AUROC               | `metrics/auroc.py`     | Image/Pixel   | Area under ROC curve          |
| AUPR                | `metrics/aupr.py`      | Image/Pixel   | Area under PR curve           |
| AUPRO               | `metrics/pro.py`       | Pixel         | Area under per-region overlap |
| F1Score             | (torchmetrics)         | Image/Pixel   | F1 score                      |
| F1Max               | `metrics/f1_score.py`  | Image         | Max F1 across thresholds      |
| F1AdaptiveThreshold | `metrics/threshold.py` | Image/Pixel   | Threshold at max F1           |
| ManualThreshold     | `metrics/threshold.py` | Image/Pixel   | User-set threshold            |
| MinMax              | `metrics/min_max.py`   | Normalization | Min-max normalization stats   |
| PIMO                | `metrics/pimo/pimo.py` | Pixel         | Per-Image Metric Overlap      |

## Key Files

- `src/anomalib/metrics/evaluator.py`
- `src/anomalib/metrics/__init__.py`
- `src/anomalib/metrics/threshold.py`
