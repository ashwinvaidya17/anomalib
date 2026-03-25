# Visualization — Agent Context

## Visualizer (`src/anomalib/visualization/base.py`)

The Visualizer is a base class (nn.Module + Callback).
Concrete implementations like `ImageVisualizer` handle the generation and storage of visualization results.

### ImageVisualizer (`src/anomalib/visualization/image/visualizer.py`)

Concrete implementation that saves visualization images during inference.

- **Hooks**:
  - `on_test_batch_end`: Saves visualizations during testing.
  - `on_predict_batch_end`: Saves visualizations during prediction.
- **Output**: Creates overlay images showing the original image, anomaly heatmap, predicted mask, and ground truth mask.

## Item and Functional Visualization

- **ItemVisualizer**: Per-item visualization logic in `visualization/image/item_visualizer.py`.
- **Functional Helpers**:
  - `visualize_anomaly_map()`: Creates heatmap overlays.
  - `visualize_mask()`: Visualizes segmentation masks.
  - `visualize_image_item()`: Full visualization of an `ImageItem`.

## Key Files

- `src/anomalib/visualization/base.py`
- `src/anomalib/visualization/image/visualizer.py`
- `src/anomalib/visualization/image/item_visualizer.py`
- `src/anomalib/visualization/image/functional.py`
