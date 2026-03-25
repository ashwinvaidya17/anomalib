# Anomalib — Agent Context Index

Anomalib is a deep learning library for anomaly detection in image and video data. It provides a modular framework for building, training, and deploying anomaly detection models. The library is built on PyTorch Lightning to ensure scalability and ease of use.

## Architecture Overview

The following diagram illustrates the relationship between core components in the Anomalib ecosystem.

```bash
┌─────────────────────────────────────────────────────────┐
│                        Engine                            │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐              │
│  │DataModule│→ │  Model   │→ │  Export   │              │
│  │(data/)   │  │(models/) │  │(deploy/)  │              │
│  └──────────┘  └──────────┘  └───────────┘              │
│       ↑              ↑              ↑                    │
│  ┌─────────┐   ┌──────────┐  ┌───────────┐              │
│  │Transforms│  │PreProc   │  │PostProc   │              │
│  │(data/   │  │(pre_proc)│  │(post_proc)│              │
│  │transforms)│ └──────────┘  └───────────┘              │
│  └─────────┘        ↓              ↓                    │
│              ┌──────────┐  ┌───────────┐                │
│              │Evaluator │  │Visualizer │                │
│              │(metrics/) │  │(visual/)  │                │
│              └──────────┘  └───────────┘                │
└─────────────────────────────────────────────────────────┘
```

## Component Reference

| Component      | Purpose                            | Context File                             |
| -------------- | ---------------------------------- | ---------------------------------------- |
| Models         | Anomaly detection algorithms       | [models.md](models.md)                   |
| Data           | Dataset loading & transforms       | [data.md](data.md)                       |
| Engine         | Training orchestration             | [engine.md](engine.md)                   |
| Deploy         | Export & inference                 | [deploy.md](deploy.md)                   |
| Callbacks      | Lightning lifecycle hooks          | [callbacks.md](callbacks.md)             |
| Metrics        | Evaluation metrics                 | [metrics.md](metrics.md)                 |
| PreProcessing  | Input transforms                   | [pre-processing.md](pre-processing.md)   |
| PostProcessing | Score normalization & thresholding | [post-processing.md](post-processing.md) |
| Visualization  | Result visualization               | [visualization.md](visualization.md)     |

## Extended Guides

| Guide                                          | Purpose                                                      |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [porting-models.md](porting-models.md)         | Step-by-step guide for porting new models into the library   |
| [openvino-inference.md](openvino-inference.md) | Comprehensive guide for OpenVINO inference in Python and C++ |

## Core Design Principles

1. **Dual-nature pattern**: Core components like PreProcessor, PostProcessor, Evaluator, and Visualizer function as both `nn.Module` (for inference and graph integration) and `Callback` (for training lifecycle integration).
2. **Component resolution**: System configuration uses a flexible resolution pattern. Set to `True` for defaults, `False` to disable, or provide a specific instance to override.
3. **Forward pipeline**: The standard inference sequence is `pre_processor → model → post_processor`. This entire chain is often baked directly into exported model graphs.
4. **Config-driven**: Every model and dataset is defined through YAML configuration files, allowing for reproducible experiments and easy parameter tuning.
5. **Export-first**: All pre and post-processing steps are designed with export compatibility in mind, ensuring a seamless transition from training to production environments.

## Quick Start

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

engine.fit(model=model, datamodule=datamodule)
engine.export(model=model, export_type="openvino")
```

## Key Modules

### Engine

The Engine class serves as a high-level wrapper around the PyTorch Lightning Trainer. It manages the training, validation, testing, and prediction loops while handling callbacks and logging.

### Models

Anomalib supports various learning types:

- **ONE_CLASS**: Trained only on normal data.
- **ZERO_SHOT**: Inference without any training on the target dataset.
- **FEW_SHOT**: Adapt to new domains with minimal samples.

### Data

DataModules handle dataset downloading, splitting, and loading. They apply transforms that are compatible with the model's expected input format.

### Deploy

The deployment module handles model conversion to efficient formats like OpenVINO, ONNX, and TorchScript. It ensures that inference results remain consistent across different backends.
