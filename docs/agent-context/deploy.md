# Deployment and Inference — Agent Context

Anomalib supports exporting trained models to various formats for efficient production deployment. The `deploy/` directory contains the logic for exporting models and performing inference using different backends.

## Export Pipeline

The `Engine.export()` method is the primary entry point for model conversion. It manages the flow from a trained `AnomalibModule` to a standalone deployment format.

- **Engine.export()**: Orchestrates the process. Example: `engine.export(model=model, export_type=ExportType.OPENVINO)`.
- **ExportType**: Enum defined in `deploy/export.py` with members `TORCH`, `ONNX`, and `OPENVINO`.
- **CompressionType**: Supports `FP16`, `INT8`, `INT8_PTQ`, and `INT8_ACQ` for optimized inference.
- **Export Flow Details**:
  - **TORCH**: Uses `model.to_torch()`. Saves the entire model state as a `.pt` file.
  - **ONNX**: Uses `model.to_onnx()`. Runs a dummy forward pass to infer output names and then calls `torch.onnx.export`.
  - **OPENVINO**: Uses `model.to_openvino()`. Converts the ONNX model to OpenVINO IR format (`.xml` and `.bin`). Includes optional NNCF compression.
- **ExportMixin**: Located at `src/anomalib/models/components/base/export_mixin.py`. It implements the core conversion logic and is mixed into the `AnomalibModule`.
- **Baked-in Processing**: Pre and post-processing steps can be embedded directly into the exported graph via exportable transforms.

## Export Code Example

```python
from anomalib.engine import Engine
from anomalib.deploy import ExportType, CompressionType

# Initialize engine and provide the model to export
engine = Engine()
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./exported_models",
    compression_type=CompressionType.INT8,
)
```

## Inferencers

Anomalib provides high-level classes to load exported models and perform predictions without requiring the full training framework.

- **OpenVINOInferencer** (`deploy/inferencers/openvino_inferencer.py`):
  - Loads OpenVINO IR (`.xml/.bin`) or `.onnx` files.
  - Handles input normalization to [0, 1] and NCHW format.
  - Returns a `NumpyImageBatch` containing prediction fields.

  ```python
  from anomalib.deploy import OpenVINOInferencer
  inferencer = OpenVINOInferencer(path="model.xml", device="CPU")
  prediction = inferencer.predict("test_image.jpg")
  # Access results: prediction.pred_score, prediction.anomaly_map
  ```

- **TorchInferencer** (`deploy/inferencers/torch_inferencer.py`):
  - Loads standard `.pt` checkpoints.
  - Returns an `ImageBatch` object.

  ```python
  from anomalib.deploy import TorchInferencer
  inferencer = TorchInferencer(path="model.pt", device="cpu")
  prediction = inferencer.predict("test_image.jpg")
  ```

- **Interface**: Both inherit from `Inferencer` ABC (`deploy/inferencers/base_inferencer.py`), which defines the standard `load_model`, `pre_process`, `forward`, `post_process`, and `predict` lifecycle.
- **Input Types**: All inferencers accept file paths, PIL images, numpy arrays, or torch tensors.

## Model Output Format

- **Torch Backend**: Returns `ImageBatch` derived from the model `InferenceBatch`.
- **OpenVINO Backend**: Returns `NumpyImageBatch` with keys matching the output blob names.
- **Standard Fields**:
  - `pred_score`: Float representing the image-level anomaly score.
  - `anomaly_map`: (H, W) array showing the pixel-level anomaly heatmap.
  - `pred_label`: Integer (0 for normal, 1 for anomalous) based on thresholding.
  - `pred_mask`: (H, W) binary mask identifying anomalous regions.

## Pre/Post Processing in Exports

Embedding transforms into the model graph ensures consistency between training and deployment.

- **Pre-processing**: `PreProcessor.export_transform` embeds logic into the ONNX/OpenVINO graph.
  - Uses `get_exportable_transform()` to swap incompatible layers (e.g., `CenterCrop` becomes `ExportableCenterCrop`).
  - If transforms aren't exportable, pre-processing must be performed manually by the user before calling the model.
- **Post-processing**: The `PostProcessor` handles normalization and thresholding.
  - During export, the `forward()` method of the `PostProcessor` is traced.
  - Exported models can output normalized scores and final labels directly.
  - Threshold constants (image and pixel) are baked into the graph.

## OpenVINO IR Structure (for C++ usage)

The IR consists of a `model.xml` file for the graph and a `model.bin` file for the weights.

- **Input**: Expects a `[1, 3, H, W]` float32 tensor normalized to the [0, 1] range.
- **Output Blobs**: Named according to the `InferenceBatch` fields.
- **C++ Usage Pattern**:

  ```cpp
  ov::Core core;
  auto model = core.read_model("model.xml");
  auto compiled = core.compile_model(model, "CPU");
  auto infer_request = compiled.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();
  auto output = infer_request.get_output_tensor();
  ```

## Key Files

- **Export Types**: `src/anomalib/deploy/export.py`
- **ExportMixin**: `src/anomalib/models/components/base/export_mixin.py`
- **OpenVINO Inferencer**: `src/anomalib/deploy/inferencers/openvino_inferencer.py`
- **Torch Inferencer**: `src/anomalib/deploy/inferencers/torch_inferencer.py`
- **Base Inferencer**: `src/anomalib/deploy/inferencers/base_inferencer.py`
- **Inference Script**: `tools/inference/openvino_inference.py`
