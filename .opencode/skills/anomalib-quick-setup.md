---
name: anomalib-quick-setup
description: Use when setting up anomalib training, running models on datasets, using the CLI, configuring data pipelines, or doing any quick-start workflow. Covers Python API, CLI commands, custom datasets, model selection, export, and inference.
---

# Anomalib Quick Setup

You are helping set up anomalib — a PyTorch Lightning-based anomaly detection library. Use the context files in `docs/agent-context/` for deep dives into specific subsystems. This skill covers the common quick-start paths.

## Context File Routing

Before diving in, read the relevant context file for the subsystem you're working in:

| Working In                      | Read First                              |
| ------------------------------- | --------------------------------------- |
| `src/anomalib/models/`          | `docs/agent-context/models.md`          |
| `src/anomalib/data/`            | `docs/agent-context/data.md`            |
| `src/anomalib/engine/`          | `docs/agent-context/engine.md`          |
| `src/anomalib/deploy/`          | `docs/agent-context/deploy.md`          |
| `src/anomalib/callbacks/`       | `docs/agent-context/callbacks.md`       |
| `src/anomalib/metrics/`         | `docs/agent-context/metrics.md`         |
| `src/anomalib/pre_processing/`  | `docs/agent-context/pre-processing.md`  |
| `src/anomalib/post_processing/` | `docs/agent-context/post-processing.md` |
| `src/anomalib/visualization/`   | `docs/agent-context/visualization.md`   |

---

## 1. Installation

```bash
# Development install (pick your hardware backend)
uv sync --extra dev --extra cpu     # CPU
uv sync --extra dev --extra cu124   # CUDA 12.4
uv sync --extra dev --extra cu126   # CUDA 12.6
uv sync --extra dev --extra rocm    # AMD ROCm
uv sync --extra dev --extra xpu     # Intel XPU

# Production install
uv pip install "anomalib[cpu]"

# With optional features
uv pip install "anomalib[openvino,cu124]"
```

---

## 2. Python API — Training

### Minimal training (3 lines)

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

engine.fit(model=model, datamodule=datamodule)
```

### Full workflow (train + test + export)

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

datamodule = MVTecAD(category="bottle")
model = Patchcore()
engine = Engine(max_epochs=1)

# Train
engine.fit(datamodule=datamodule, model=model)

# Test
engine.test(datamodule=datamodule, model=model)

# Predict
predictions = engine.predict(datamodule=datamodule, model=model)

# Export
engine.export(model=model, export_type=ExportType.OPENVINO)
```

### Convenience method — `engine.train()`

`engine.train()` runs fit + test + export in one call.

---

## 3. CLI Usage

### Subcommands

| Command             | What it does                      |
| ------------------- | --------------------------------- |
| `anomalib fit`      | Train the model                   |
| `anomalib test`     | Evaluate a trained model          |
| `anomalib validate` | Run validation                    |
| `anomalib train`    | Fit + test + export (convenience) |
| `anomalib predict`  | Run inference                     |
| `anomalib export`   | Export to ONNX/OpenVINO/Torch     |
| `anomalib install`  | Install optional dependencies     |

### Common CLI patterns

```bash
# Train with inline arguments
anomalib fit --model Patchcore --data MVTecAD --data.category bottle

# Train with config files
anomalib fit -c examples/configs/model/patchcore.yaml --data examples/configs/data/mvtec.yaml

# Full train + test + export
anomalib train --model Patchcore --data MVTecAD

# Predict from checkpoint
anomalib predict --model Patchcore --data MVTecAD --ckpt_path path/to/checkpoint.ckpt

# Export a trained model
anomalib export --model Patchcore --ckpt_path path/to/checkpoint.ckpt --export_type OPENVINO
```

### Config file structure

Model config (`examples/configs/model/patchcore.yaml`):

```yaml
model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2
    layers:
      - layer2
      - layer3
    pre_trained: true
    coreset_sampling_ratio: 0.1
    num_neighbors: 9
```

Data config (`examples/configs/data/mvtec.yaml`):

```yaml
class_path: anomalib.data.MVTecAD
init_args:
  root: ./datasets/MVTecAD
  category: bottle
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 8
  test_split_mode: from_dir
  test_split_ratio: 0.2
  val_split_mode: same_as_test
  val_split_ratio: 0.5
```

---

## 4. Available Models

| Model                   | Learning Type      | Best For                                         |
| ----------------------- | ------------------ | ------------------------------------------------ |
| **Patchcore**           | ONE_CLASS          | General purpose, strong baseline, memory bank    |
| **PaDiM**               | ONE_CLASS          | Fast, lightweight patch distribution modeling    |
| **EfficientAd**         | ONE_CLASS          | Teacher-student with autoencoder, fast inference |
| **STFPM**               | ONE_CLASS          | Student-teacher feature pyramid matching         |
| **FastFlow**            | ONE_CLASS          | Normalizing flows, good accuracy                 |
| **Cflow**               | ONE_CLASS          | Conditional normalizing flows                    |
| **DRAEM**               | ONE_CLASS          | Reconstruction-based with synthetic anomalies    |
| **ReverseDistillation** | ONE_CLASS          | Reverse knowledge distillation                   |
| **WinCLIP**             | ZERO_SHOT/FEW_SHOT | CLIP-based, no training needed for zero-shot     |
| **AnomalyDINO**         | ZERO_SHOT/FEW_SHOT | DINO-based, no training needed for zero-shot     |
| **VLM-AD**              | ZERO_SHOT          | Vision-language model based                      |
| **Dinomaly**            | ONE_CLASS          | DINOv2-based feature detection                   |
| **SuperSimpleNet**      | ONE_CLASS          | Simple discriminative network                    |

**Learning types:**

- `ONE_CLASS` — Trained only on normal data (most models).
- `ZERO_SHOT` — No training needed. Inference only (WinCLIP, AnomalyDINO, VLM-AD).
- `FEW_SHOT` — Adapts with minimal samples.

All models are importable from `anomalib.models`:

```python
from anomalib.models import Patchcore, Padim, EfficientAd, Stfpm, WinClip
```

---

## 5. Available Datasets

| Dataset           | Import                       | Modality |
| ----------------- | ---------------------------- | -------- |
| MVTecAD           | `anomalib.data.MVTecAD`      | Image    |
| MVTec LOCO        | `anomalib.data.MVTecLoco`    | Image    |
| MVTec 3D          | `anomalib.data.MVTec3D`      | Depth    |
| BTech             | `anomalib.data.BTech`        | Image    |
| Visa              | `anomalib.data.Visa`         | Image    |
| Folder (custom)   | `anomalib.data.Folder`       | Image    |
| Folder3D (custom) | `anomalib.data.Folder3D`     | Depth    |
| Avenue            | `anomalib.data.Avenue`       | Video    |
| ShanghaiTech      | `anomalib.data.ShanghaiTech` | Video    |
| UCSDped           | `anomalib.data.UCSDped`      | Video    |

---

## 6. Custom Datasets (Folder DataModule)

Use the `Folder` DataModule for any custom image dataset — no new code needed.

```python
from anomalib.data import Folder

datamodule = Folder(
    name="my_dataset",
    root="./datasets/my_data",
    normal_dir="good",            # Required: directory with normal images
    abnormal_dir="defect",        # Optional: directory with anomalous images
    mask_dir="masks",             # Optional: segmentation masks
    task="segmentation",          # "classification" or "segmentation"
)
```

Expected directory layout:

```text
datasets/my_data/
  good/           # Normal images
  defect/         # Anomalous images (optional)
  masks/          # Ground truth masks matching defect/ names (optional)
```

Folder data config (`examples/configs/data/folder.yaml`):

```yaml
class_path: anomalib.data.Folder
init_args:
  name: my_dataset
  root: "datasets/my_data"
  normal_dir: "train/good"
  abnormal_dir: "test/broken_large"
  normal_test_dir: "test/good"
  mask_dir: "ground_truth/broken_large"
  extensions: [".png"]
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 8
```

---

## 7. Export & Inference

### Export types

| Type                  | Format      | Use case                   |
| --------------------- | ----------- | -------------------------- |
| `ExportType.TORCH`    | `.pt`       | PyTorch deployment         |
| `ExportType.ONNX`     | `.onnx`     | Cross-framework            |
| `ExportType.OPENVINO` | `.xml/.bin` | Intel hardware, production |

### Compression options

`CompressionType.FP16`, `INT8`, `INT8_PTQ`, `INT8_ACQ`

### Python export

```python
from anomalib.deploy import ExportType, CompressionType

engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
    export_root="./exported_models",
    compression_type=CompressionType.INT8,
)
```

### Inference with exported models

```python
from anomalib.deploy import OpenVINOInferencer, TorchInferencer

# OpenVINO
inferencer = OpenVINOInferencer(path="model.xml", device="CPU")
prediction = inferencer.predict("test_image.jpg")
print(prediction.pred_score, prediction.anomaly_map)

# Torch
inferencer = TorchInferencer(path="model.pt", device="cpu")
prediction = inferencer.predict("test_image.jpg")
```

---

## 8. Key Architecture Rules

1. **Always use `Engine`** — never use `pl.Trainer` directly.
2. **Dual-nature pattern** — PreProcessor, PostProcessor, Evaluator, Visualizer are both `nn.Module` and Lightning `Callback`.
3. **Forward pipeline** — `pre_processor(batch) -> model(batch) -> post_processor(batch)`.
4. **Component resolution** — `True` = default, `False` = disable, instance = override.
5. **Use `ExportableCenterCrop`** — standard `CenterCrop` breaks model export.
6. **Use `batch.update()`** — never modify batch objects in-place.
7. **Config-driven** — every model has a YAML config in `examples/configs/model/`.
8. **Model structure** — each model has `torch_model.py` (pure nn.Module) and `lightning_model.py` (AnomalibModule wrapper).

---

## 9. Intel XPU Training

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Stfpm

engine = Engine(strategy=SingleXPUStrategy(), accelerator=XPUAccelerator())
engine.train(Stfpm(), datamodule=MVTecAD())
```

CLI:

```bash
anomalib train --model Padim --data MVTecAD --trainer.accelerator xpu --trainer.strategy xpu_single
```

---

## 10. Quick Debugging Checklist

- **Model not found?** Check it's registered in `src/anomalib/models/__init__.py`.
- **Export failing?** Check for non-exportable transforms (use `ExportableCenterCrop`).
- **Zero-shot model with `fit`?** Engine auto-skips training for `ZERO_SHOT` learning type.
- **Custom data not loading?** Verify directory structure matches `Folder` expectations.
- **Config not applying?** CLI args override config file values. Check argument precedence.

---

## Key Files Reference

| What              | Where                                                    |
| ----------------- | -------------------------------------------------------- |
| Engine            | `src/anomalib/engine/engine.py`                          |
| CLI               | `src/anomalib/cli/cli.py`                                |
| Model base class  | `src/anomalib/models/components/base/anomalib_module.py` |
| Model registry    | `src/anomalib/models/__init__.py`                        |
| Data base class   | `src/anomalib/data/datamodules/base/image.py`            |
| Folder DataModule | `src/anomalib/data/datamodules/image/folder.py`          |
| Export types      | `src/anomalib/deploy/export.py`                          |
| Inferencers       | `src/anomalib/deploy/inferencers/`                       |
| Model configs     | `examples/configs/model/*.yaml`                          |
| Data configs      | `examples/configs/data/*.yaml`                           |
| API examples      | `examples/api/01_getting_started/`                       |
