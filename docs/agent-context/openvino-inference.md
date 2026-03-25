# OpenVINO Inference Guide

OpenVINO (Open Visual Inference and Neural Network Optimization) is a toolkit that accelerates deep learning inference across Intel hardware. Anomalib provides built-in support for exporting models to the OpenVINO Intermediate Representation (IR) and running inference in both Python and C++.

## Exporting to OpenVINO IR

Use the `Engine` to convert an Anomalib model into an OpenVINO-compatible format. This step produces two files: `model.xml` for the network graph and `model.bin` for the trained weights.

```python
from anomalib.engine import Engine
from anomalib.deploy import ExportType, CompressionType

engine = Engine()
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./exported_model",
    compression_type=CompressionType.FP16,
)
```

## Understanding the IR Model

- **Input**: Typically a 4D tensor of shape `[1, 3, H, W]` with float32 data. Images should be normalized to the range [0, 1] unless pre-processing is baked into the model.
- **Outputs**:
  - `pred_score`: A scalar value representing the anomaly score for the entire image.
  - `anomaly_map`: A heatmap of shape `[1, 1, H, W]` showing the location of anomalies.
- **Pre/Post-Processing**: Anomalib often embeds transforms directly into the IR graph, simplifying the inference pipeline.

## Python Inference with `OpenVINOInferencer`

The `OpenVINOInferencer` class handles the details of loading the model and preparing inputs for inference.

```python
from anomalib.deploy import OpenVINOInferencer

inferencer = OpenVINOInferencer(path="exported_model/model.xml", device="CPU")

# Single image prediction
result = inferencer.predict("path/to/test.jpg")
print(f"Anomaly Score: {result.pred_score:.4f}")
print(f"Heatmap Shape: {result.anomaly_map.shape}")

# Batch processing
from pathlib import Path
for img_path in Path("test_set/").glob("*.jpg"):
    result = inferencer.predict(str(img_path))
    if result.pred_score > 0.5:
        print(f"Anomaly detected in {img_path.name}")
```

## C++ Inference with OpenVINO Runtime

For high-performance applications, use the OpenVINO C++ API. This requires linking against the `openvino::runtime` library.

```cpp
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Initialize and Load Model
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled = core.compile_model(model, "CPU");
    auto infer_request = compiled.create_infer_request();

    // 2. Prepare Input with OpenCV
    cv::Mat image = cv::imread("test.jpg");
    cv::resize(image, image, cv::Size(256, 256));
    image.convertTo(image, CV_32F, 1.0 / 255.0);  // Scale to [0,1]
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // RGB conversion

    // 3. Convert HWC Mat to NCHW Tensor
    float* blob_data = new float[1 * 3 * 256 * 256];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 256; ++h) {
            for (int w = 0; w < 256; ++w) {
                blob_data[c * 256 * 256 + h * 256 + w] = image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    auto input_tensor = ov::Tensor(ov::element::f32, {1, 3, 256, 256}, blob_data);
    infer_request.set_input_tensor(input_tensor);

    // 4. Perform Inference
    infer_request.infer();

    // 5. Retrieve Outputs
    auto score_tensor = infer_request.get_output_tensor(0);
    float anomaly_score = score_tensor.data<float>()[0];

    auto map_tensor = infer_request.get_output_tensor(1);
    // map_tensor can be converted to cv::Mat for further processing

    delete[] blob_data;
    return 0;
}
```

## C++ Heatmap Visualization

```cpp
cv::Mat create_heatmap_overlay(const cv::Mat& original, const cv::Mat& anomaly_map, double alpha = 0.4) {
    cv::Mat normalized;
    cv::normalize(anomaly_map, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);

    cv::Mat heatmap;
    cv::applyColorMap(normalized, heatmap, cv::COLORMAP_JET);

    cv::Mat result;
    cv::addWeighted(original, 1.0 - alpha, heatmap, alpha, 0, result);
    return result;
}
```

## CMake Configuration

```cmake
cmake_minimum_required(VERSION 3.10)
project(AnomalibInference)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(detector main.cpp)
target_link_libraries(detector PRIVATE openvino::runtime ${OpenCV_LIBS})
```

## Optimization Options

| Compression | Size     | Speed         | Accuracy | Use Case                        |
| ----------- | -------- | ------------- | -------- | ------------------------------- |
| None (FP32) | Baseline | Baseline      | Best     | Development and initial testing |
| FP16        | ~50%     | Faster on GPU | High     | General production use          |
| INT8 (PTQ)  | ~25%     | 2-4x faster   | Good     | Edge devices and CPU-bound apps |

### Important Notes

- **Pre-processing Integration**: Check the `model.xml` metadata. If pre-processing is baked in, you can feed raw uint8 images to the model.
- **Output Naming**: Use the OpenVINO `benchmark_app` or `model_viewer` tools to verify the exact names of output blobs if they aren't at indices 0 and 1.
- **Model Resize**: If the model was trained with specific dimensions (e.g., 224x224), ensure your input tensor matches this shape unless dynamic input shapes were enabled during export.
