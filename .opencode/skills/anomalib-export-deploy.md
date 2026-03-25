---
name: anomalib-export-deploy
description: Use when exporting anomalib models to production formats (OpenVINO, ONNX, PyTorch), creating inference scripts, deploying models in Python or C++, setting up visualization of predictions, or integrating exported models into external applications. Covers the full train-export-deploy-visualize pipeline. Triggers on phrases like "export model", "deploy model", "inference script", "OpenVINO", "ONNX", "C++ inference", "production deployment", "export to", "create inference", "run inference", "visualize predictions".
---

# Export & Deploy Anomalib Models

This skill guides the full pipeline from training a model to deploying it in production. Covers export formats, Python and C++ inference, compression, visualization, and integration into external applications.

**Before starting, read these context files:**

- `docs/agent-context/deploy.md` — Export pipeline, inferencers, output format
- `docs/agent-context/openvino-inference.md` — Full Python + C++ inference guide with CMake
- `docs/agent-context/visualization.md` — Visualization callback and functional helpers
- `docs/agent-context/engine.md` — Engine orchestration (train, test, export)
- `docs/agent-context/pre-processing.md` — Pre-processing baked into exports
- `docs/agent-context/post-processing.md` — Post-processing baked into exports

---

## Phase 1: Train + Export

### 1.1 Full Train-Export Pipeline (Python API)

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType, CompressionType

# Setup
datamodule = MVTecAD(category="bottle")
model = Patchcore()
engine = Engine(max_epochs=1)

# Train
engine.fit(model=model, datamodule=datamodule)

# Test (optional — evaluates metrics)
engine.test(model=model, datamodule=datamodule)

# Export
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./exported_models",
)
```

### 1.2 Full Train-Export Pipeline (CLI)

```bash
# Train + test + export in one command
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category bottle

# Export a trained checkpoint
anomalib export \
    --model Patchcore \
    --ckpt_path path/to/checkpoint.ckpt \
    --export_type OPENVINO \
    --export_root ./exported_models

# Export with compression
anomalib export \
    --model Patchcore \
    --ckpt_path path/to/checkpoint.ckpt \
    --export_type OPENVINO \
    --compression_type INT8
```

### 1.3 Convenience: `engine.train()`

`engine.train()` runs fit + test + export in a single call:

```python
engine.train(model=model, datamodule=datamodule)
```

---

## Phase 2: Export Formats

### 2.1 Available Export Types

| Export Type           | Format      | File(s)                   | Use Case                          |
| --------------------- | ----------- | ------------------------- | --------------------------------- |
| `ExportType.TORCH`    | PyTorch     | `model.pt`                | Python-only deployment, debugging |
| `ExportType.ONNX`     | ONNX        | `model.onnx`              | Cross-framework, ONNX Runtime     |
| `ExportType.OPENVINO` | OpenVINO IR | `model.xml` + `model.bin` | Intel hardware, production, edge  |

### 2.2 Compression Options

| Compression | Enum                       | Size vs FP32 | Speed         | Accuracy        | Best For                |
| ----------- | -------------------------- | ------------ | ------------- | --------------- | ----------------------- |
| None (FP32) | —                          | Baseline     | Baseline      | Best            | Development, testing    |
| FP16        | `CompressionType.FP16`     | ~50%         | Faster on GPU | High            | General production      |
| INT8 (PTQ)  | `CompressionType.INT8_PTQ` | ~25%         | 2-4x faster   | Good            | Edge devices, CPU-bound |
| INT8 (ACQ)  | `CompressionType.INT8_ACQ` | ~25%         | 2-4x faster   | Better than PTQ | Accuracy-sensitive edge |
| INT8        | `CompressionType.INT8`     | ~25%         | 2-4x faster   | Good            | General INT8            |

### 2.3 Export with Compression

```python
from anomalib.deploy import ExportType, CompressionType

# FP16 — good default for production
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    compression_type=CompressionType.FP16,
)

# INT8 PTQ — best for edge/CPU
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    compression_type=CompressionType.INT8_PTQ,
)
```

### 2.4 What Gets Baked Into the Export

Anomalib exports embed pre-processing and post-processing into the model graph:

- **Pre-processing**: Normalization, resize, crop are traced into the ONNX/OpenVINO graph via `PreProcessor.export_transform`
- **Post-processing**: Score normalization and thresholding are traced via `PostProcessor.forward()`
- **Result**: Exported models accept raw images and produce final predictions — no external pipeline needed

**Important**: Use `ExportableCenterCrop` instead of standard `CenterCrop` in pre-processor transforms. Standard `CenterCrop` breaks ONNX/OpenVINO tracing.

---

## Phase 3: Python Inference

### 3.1 OpenVINO Inferencer

The recommended production inferencer for Intel hardware.

```python
from anomalib.deploy import OpenVINOInferencer

# Load the exported model
inferencer = OpenVINOInferencer(
    path="exported_models/model.xml",
    device="CPU",  # "CPU", "GPU", "AUTO"
)

# Single image prediction
result = inferencer.predict("path/to/test_image.jpg")

# Access prediction fields
print(f"Anomaly Score: {result.pred_score:.4f}")
print(f"Label: {'Anomalous' if result.pred_label else 'Normal'}")
print(f"Anomaly Map Shape: {result.anomaly_map.shape}")  # (H, W)
print(f"Pred Mask Shape: {result.pred_mask.shape}")       # (H, W)
```

### 3.2 Torch Inferencer

For PyTorch-based deployment without OpenVINO.

```python
from anomalib.deploy import TorchInferencer

inferencer = TorchInferencer(
    path="exported_models/model.pt",
    device="cpu",  # "cpu", "cuda", "xpu"
)

result = inferencer.predict("path/to/test_image.jpg")
print(f"Anomaly Score: {result.pred_score:.4f}")
```

### 3.3 Accepted Input Types

Both inferencers accept multiple input formats:

```python
# File path (string or Path)
result = inferencer.predict("test.jpg")
result = inferencer.predict(Path("test.jpg"))

# PIL Image
from PIL import Image
img = Image.open("test.jpg")
result = inferencer.predict(img)

# NumPy array (H, W, C) uint8 or float
import numpy as np
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result = inferencer.predict(img)

# Torch tensor (C, H, W) float
import torch
img = torch.randn(3, 256, 256)
result = inferencer.predict(img)
```

### 3.4 Batch Processing Script

```python
from pathlib import Path
from anomalib.deploy import OpenVINOInferencer

inferencer = OpenVINOInferencer(path="model.xml", device="CPU")

results = []
for img_path in sorted(Path("test_images/").glob("*.jpg")):
    result = inferencer.predict(str(img_path))
    results.append({
        "image": img_path.name,
        "score": float(result.pred_score),
        "label": "anomalous" if result.pred_label else "normal",
    })
    if result.pred_label:
        print(f"⚠ Anomaly detected: {img_path.name} (score: {result.pred_score:.4f})")

# Summary
n_anomalous = sum(1 for r in results if r["label"] == "anomalous")
print(f"\nTotal: {len(results)} images, {n_anomalous} anomalies detected")
```

### 3.5 Prediction Output Fields

| Field         | Type           | Description                             |
| ------------- | -------------- | --------------------------------------- |
| `pred_score`  | float          | Image-level anomaly score               |
| `pred_label`  | int            | 0 = normal, 1 = anomalous (thresholded) |
| `anomaly_map` | ndarray (H, W) | Pixel-level anomaly heatmap             |
| `pred_mask`   | ndarray (H, W) | Binary mask of anomalous regions        |

- **OpenVINO inferencer** returns `NumpyImageBatch` (numpy arrays)
- **Torch inferencer** returns `ImageBatch` (torch tensors)

---

## Phase 4: C++ Inference (OpenVINO)

For high-performance production applications requiring C++ integration.

### 4.1 Full C++ Inference Example

```cpp
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.xml> <image.jpg>" << std::endl;
        return 1;
    }

    // 1. Initialize OpenVINO and load model
    ov::Core core;
    auto model = core.read_model(argv[1]);
    auto compiled = core.compile_model(model, "CPU");
    auto infer_request = compiled.create_infer_request();

    // 2. Read and preprocess image with OpenCV
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << argv[2] << std::endl;
        return 1;
    }

    // Get model input dimensions
    auto input_shape = model->input().get_shape();
    int input_h = input_shape[2];
    int input_w = input_shape[3];

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w, input_h));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // Normalize to [0,1]
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3. Convert HWC to NCHW tensor
    float* blob_data = new float[1 * 3 * input_h * input_w];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                blob_data[c * input_h * input_w + h * input_w + w] =
                    resized.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    auto input_tensor = ov::Tensor(ov::element::f32, input_shape, blob_data);
    infer_request.set_input_tensor(input_tensor);

    // 4. Run inference
    infer_request.infer();

    // 5. Get results
    auto score_tensor = infer_request.get_output_tensor(0);
    float anomaly_score = score_tensor.data<float>()[0];

    std::cout << "Anomaly Score: " << anomaly_score << std::endl;
    std::cout << (anomaly_score > 0.5 ? "ANOMALOUS" : "NORMAL") << std::endl;

    // 6. Get anomaly map (if available)
    if (compiled.outputs().size() > 1) {
        auto map_tensor = infer_request.get_output_tensor(1);
        auto map_shape = map_tensor.get_shape();
        int map_h = map_shape[2];
        int map_w = map_shape[3];

        cv::Mat anomaly_map(map_h, map_w, CV_32F, map_tensor.data<float>());
        // ... process anomaly map for visualization
    }

    delete[] blob_data;
    return 0;
}
```

### 4.2 CMake Configuration

```cmake
cmake_minimum_required(VERSION 3.10)
project(AnomalibInference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(anomalib_detector main.cpp)
target_link_libraries(anomalib_detector PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
)
```

Build:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./anomalib_detector ../model.xml ../test_image.jpg
```

### 4.3 C++ Heatmap Visualization

```cpp
cv::Mat create_heatmap_overlay(
    const cv::Mat& original,
    const cv::Mat& anomaly_map,
    double alpha = 0.4
) {
    // Normalize anomaly map to 0-255
    cv::Mat normalized;
    cv::normalize(anomaly_map, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);

    // Resize to match original image
    cv::Mat resized_map;
    cv::resize(normalized, resized_map, original.size());

    // Apply colormap
    cv::Mat heatmap;
    cv::applyColorMap(resized_map, heatmap, cv::COLORMAP_JET);

    // Blend with original
    cv::Mat result;
    cv::addWeighted(original, 1.0 - alpha, heatmap, alpha, 0, result);
    return result;
}

// Usage:
cv::Mat overlay = create_heatmap_overlay(original_image, anomaly_map);
cv::imshow("Anomaly Detection", overlay);
cv::waitKey(0);
cv::imwrite("result.jpg", overlay);
```

### 4.4 Important C++ Notes

- **Input shape**: `[1, 3, H, W]` float32, normalized to `[0, 1]`
- **Channel order**: RGB (not BGR — convert from OpenCV's default BGR)
- **Pre-processing**: If baked into the model, you can skip normalization — check the `model.xml` metadata
- **Output blob names**: Use OpenVINO's `benchmark_app` or `model_viewer` to verify output blob names and indices
- **Dynamic shapes**: If the model was exported with fixed dimensions, your input must match exactly

---

## Phase 5: Python Visualization

### 5.1 Built-in Visualization (During Training/Testing)

Anomalib's `ImageVisualizer` callback automatically saves visualizations during `engine.test()` and `engine.predict()`.

```python
from anomalib.visualization import ImageVisualizer

# Enabled by default — visualizations saved to results directory
engine = Engine()
engine.test(model=model, datamodule=datamodule)
# Check results/{model_name}/{dataset}/{run_id}/images/ for visualizations
```

To disable:

```python
model = Patchcore(visualizer=False)
```

### 5.2 Functional Visualization API

For custom visualization in inference scripts:

```python
from anomalib.visualization.image.functional import (
    visualize_anomaly_map,
    visualize_mask,
    visualize_image_item,
)

# Overlay anomaly heatmap on an image
heatmap_overlay = visualize_anomaly_map(
    image=original_image,       # (H, W, 3) uint8 or (3, H, W) tensor
    anomaly_map=result.anomaly_map,  # (H, W) float
)

# Visualize predicted mask
mask_overlay = visualize_mask(
    image=original_image,
    mask=result.pred_mask,      # (H, W) binary
)

# Full visualization of an ImageItem (shows original, heatmap, mask, GT)
full_vis = visualize_image_item(image_item)
```

### 5.3 Custom Inference + Visualization Script

```python
"""Inference script with visualization output.

Usage:
    python inference.py --model model.xml --input test_images/ --output results/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer


def create_heatmap_overlay(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a heatmap overlay on the original image.

    Args:
        image: Original image (H, W, 3) BGR uint8.
        anomaly_map: Anomaly map (H, W) float.
        alpha: Blending factor for the heatmap.

    Returns:
        Blended overlay image (H, W, 3) BGR uint8.
    """
    # Normalize anomaly map to 0-255
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # Resize to match original
    normalized = cv2.resize(normalized, (image.shape[1], image.shape[0]))

    # Apply colormap and blend
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    return result


def main() -> None:
    """Run inference on images and save visualizations."""
    parser = argparse.ArgumentParser(description="Anomalib Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model.xml")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold")
    args = parser.parse_args()

    # Setup
    inferencer = OpenVINOInferencer(path=args.model, device=args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))

    # Process
    for img_path in image_paths:
        result = inferencer.predict(str(img_path))
        original = cv2.imread(str(img_path))

        # Create visualization
        overlay = create_heatmap_overlay(original, result.anomaly_map)

        # Add score text
        label = "ANOMALOUS" if result.pred_score > args.threshold else "NORMAL"
        color = (0, 0, 255) if label == "ANOMALOUS" else (0, 255, 0)
        cv2.putText(
            overlay,
            f"{label} ({result.pred_score:.3f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # Save
        save_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(save_path), overlay)
        print(f"{'⚠' if label == 'ANOMALOUS' else '✓'} {img_path.name}: {label} ({result.pred_score:.3f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
```

---

## Phase 6: Creating Inference Scripts for External Integration

When users need to integrate anomalib models into their own applications.

### 6.1 Minimal Python Inference Module

A self-contained module that external Python applications can import:

```python
"""Anomaly detection inference module.

This module provides a simple interface for running anomaly detection
using an exported OpenVINO model. No anomalib training dependencies required.

Requirements:
    pip install openvino numpy opencv-python

Usage:
    from anomaly_detector import AnomalyDetector

    detector = AnomalyDetector("model.xml")
    score, is_anomalous = detector.detect("image.jpg")
"""

from pathlib import Path

import cv2
import numpy as np
import openvino as ov


class AnomalyDetector:
    """Lightweight anomaly detector using an exported OpenVINO model.

    Args:
        model_path: Path to the OpenVINO model.xml file.
        device: Inference device (CPU, GPU, AUTO).
        threshold: Anomaly score threshold.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "CPU",
        threshold: float = 0.5,
    ) -> None:
        core = ov.Core()
        model = core.read_model(str(model_path))
        self.compiled = core.compile_model(model, device)
        self.threshold = threshold

        # Get input shape
        input_shape = model.input().shape
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]

    def detect(self, image: str | Path | np.ndarray) -> tuple[float, bool]:
        """Detect anomalies in an image.

        Args:
            image: File path or numpy array (H, W, 3) BGR uint8.

        Returns:
            Tuple of (anomaly_score, is_anomalous).
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        # Preprocess
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        # Infer
        result = self.compiled([nchw])
        score = float(result[0].flatten()[0])
        return score, score > self.threshold

    def detect_with_map(
        self,
        image: str | Path | np.ndarray,
    ) -> tuple[float, bool, np.ndarray]:
        """Detect anomalies and return the anomaly heatmap.

        Args:
            image: File path or numpy array (H, W, 3) BGR uint8.

        Returns:
            Tuple of (anomaly_score, is_anomalous, anomaly_map).
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        # Preprocess
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        # Infer
        result = self.compiled([nchw])
        score = float(result[0].flatten()[0])
        anomaly_map = result[1].squeeze()  # (H, W)
        return score, score > self.threshold, anomaly_map
```

### 6.2 When Users Need Different Languages

| Language            | Approach                                             | Key Library                 |
| ------------------- | ---------------------------------------------------- | --------------------------- |
| **Python**          | Use `OpenVINOInferencer` or lightweight module above | `openvino`, `opencv-python` |
| **C++**             | Use OpenVINO C++ Runtime API (see Phase 4)           | `openvino::runtime`, OpenCV |
| **C#**              | Use OpenVINO C# API (NuGet: `OpenVINO.CSharp.API`)   | `OpenVINO.CSharp.API`       |
| **Java**            | Use OpenVINO Java API or ONNX Runtime Java           | `ai.onnxruntime`            |
| **Rust**            | Use `openvino-rs` crate or `ort` (ONNX Runtime)      | `openvino`, `ort`           |
| **JavaScript/Node** | Use ONNX Runtime Web or Node.js bindings             | `onnxruntime-node`          |
| **Go**              | Use ONNX Runtime Go bindings                         | `onnxruntime_go`            |

**General integration pattern for any language:**

1. Export model to ONNX (most portable) or OpenVINO IR (best on Intel)
2. Load model using the language's runtime bindings
3. Preprocess: resize to model input shape, normalize to [0, 1], convert HWC → NCHW
4. Run inference
5. Parse outputs: `pred_score` (scalar), `anomaly_map` (H×W heatmap)

### 6.3 ONNX Runtime Integration (Cross-Platform)

For maximum portability, export to ONNX and use ONNX Runtime:

```python
# Export to ONNX
engine.export(model=model, export_type=ExportType.ONNX, export_root="./exported")

# Inference with ONNX Runtime (works anywhere)
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("exported/model.onnx")
input_name = session.get_inputs()[0].name

# Preprocess
image = cv2.imread("test.jpg")
image = cv2.resize(image, (256, 256))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]

# Run
outputs = session.run(None, {input_name: image})
score = outputs[0].flatten()[0]
anomaly_map = outputs[1].squeeze()
```

---

## Phase 7: Workflow Recipes

### 7.1 Recipe: Train and Deploy to Edge (Intel CPU)

```python
from anomalib.data import MVTecAD
from anomalib.models import EfficientAd  # Fast inference model
from anomalib.engine import Engine
from anomalib.deploy import ExportType, CompressionType

# Train
model = EfficientAd()  # Optimized for edge
datamodule = MVTecAD(category="bottle")
engine = Engine(max_epochs=100)
engine.fit(model=model, datamodule=datamodule)

# Export with INT8 compression for edge
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./edge_model",
    compression_type=CompressionType.INT8_PTQ,
)

# Resulting files: edge_model/model.xml, edge_model/model.bin
# Deploy these two files to the edge device
```

### 7.2 Recipe: Train on Custom Data and Create Inference API

```python
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType, OpenVINOInferencer

# 1. Train on custom data
datamodule = Folder(
    name="my_product",
    root="./data/my_product",
    normal_dir="good",
    abnormal_dir="defective",
    mask_dir="masks",
)
model = Patchcore()
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# 2. Export
engine.export(model=model, export_type=ExportType.OPENVINO, export_root="./deployed")

# 3. Create inference function
inferencer = OpenVINOInferencer(path="./deployed/model.xml", device="CPU")

def check_product(image_path: str) -> dict:
    """Check a product image for defects."""
    result = inferencer.predict(image_path)
    return {
        "score": float(result.pred_score),
        "is_defective": bool(result.pred_label),
        "anomaly_map": result.anomaly_map,
    }
```

### 7.3 Recipe: Compare Export Formats

```python
from anomalib.deploy import ExportType, CompressionType

# Export all formats for comparison
for export_type in [ExportType.TORCH, ExportType.ONNX, ExportType.OPENVINO]:
    engine.export(
        model=model,
        export_type=export_type,
        export_root=f"./exports/{export_type.value}",
    )

# Export with different compression levels
for compression in [CompressionType.FP16, CompressionType.INT8_PTQ]:
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=f"./exports/openvino_{compression.value}",
        compression_type=compression,
    )
```

### 7.4 Recipe: Verify Export Works End-to-End

```python
from anomalib.deploy import ExportType, OpenVINOInferencer
import numpy as np

# Export
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./verify_export",
)

# Verify with inferencer
inferencer = OpenVINOInferencer(path="./verify_export/model.xml", device="CPU")

# Test with a dummy image
dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result = inferencer.predict(dummy)

# Verify outputs
assert result.pred_score is not None, "pred_score is None"
assert result.anomaly_map is not None, "anomaly_map is None"
assert result.anomaly_map.shape[0] > 0, "anomaly_map is empty"
print(f"Export verified: score={result.pred_score:.4f}, map_shape={result.anomaly_map.shape}")
```

---

## Export Compatibility Checklist

Before exporting, verify the model is export-friendly:

- [ ] **No `CenterCrop`** — Use `ExportableCenterCrop` instead
- [ ] **No dynamic control flow** — Avoid `if/else` based on tensor values in `forward()`
- [ ] **No Python-only ops** — All operations must be traceable by `torch.onnx.export`
- [ ] **Pre-processor uses exportable transforms** — Check via `get_exportable_transform()`
- [ ] **Test all three formats**: `ExportType.TORCH`, `ONNX`, `OPENVINO`
- [ ] **Verify inference** — Load exported model with appropriate inferencer and run `predict()`
- [ ] **Check output fields** — `pred_score`, `anomaly_map` should be non-None

---

## Troubleshooting

| Problem                              | Cause                                                  | Fix                                                              |
| ------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------- |
| ONNX export fails with tracing error | Dynamic control flow or unsupported ops                | Simplify forward pass, avoid tensor-dependent if/else            |
| OpenVINO export fails                | ONNX model uses unsupported ONNX ops                   | Check OpenVINO supported ops list, may need opset version change |
| Inference gives wrong results        | Pre-processing mismatch between training and inference | Ensure input normalization matches training transforms           |
| INT8 export fails                    | Missing calibration data                               | Provide `datamodule` when using `INT8_PTQ` or `INT8_ACQ`         |
| C++ inference shape mismatch         | Wrong input dimensions                                 | Check `model.xml` for expected input shape                       |
| Score always 0 or 1                  | Post-processing thresholds baked incorrectly           | Re-export after a proper `engine.test()` run                     |

---

## Key Files Reference

| What                                         | Where                                                    |
| -------------------------------------------- | -------------------------------------------------------- |
| Export types + compression                   | `src/anomalib/deploy/export.py`                          |
| ExportMixin (to_torch, to_onnx, to_openvino) | `src/anomalib/models/components/base/export_mixin.py`    |
| OpenVINO Inferencer                          | `src/anomalib/deploy/inferencers/openvino_inferencer.py` |
| Torch Inferencer                             | `src/anomalib/deploy/inferencers/torch_inferencer.py`    |
| Base Inferencer ABC                          | `src/anomalib/deploy/inferencers/base_inferencer.py`     |
| ImageVisualizer callback                     | `src/anomalib/visualization/image/visualizer.py`         |
| Functional visualization                     | `src/anomalib/visualization/image/functional.py`         |
| Engine (export entry point)                  | `src/anomalib/engine/engine.py`                          |
| Deploy public API                            | `src/anomalib/deploy/__init__.py`                        |
| Exportable transforms                        | `src/anomalib/data/transforms/`                          |
| Inference script example                     | `tools/inference/openvino_inference.py`                  |
| C++ inference guide                          | `docs/agent-context/openvino-inference.md`               |
