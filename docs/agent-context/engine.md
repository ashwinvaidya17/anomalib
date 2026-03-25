# Engine (Orchestration) — Agent Context

The `Engine` class (located at `src/anomalib/engine/engine.py`) wraps the PyTorch Lightning `Trainer`. It acts as the primary entry point for training, evaluating, and deploying anomaly detection models.

## Engine Overview

- **Never** use `pl.Trainer` directly. Always use `Engine`.
- Handles workspace setup, callback wiring, and model/data binding.
- Manages versioned output directories (`v0`, `v1`, ...) with a `latest` symlink.
- Core Methods: `fit()`, `validate()`, `test()`, `predict()`, `export()`.
- Convenience Method: `train()` performs fit, test, and export in one call.
- Factory: `Engine.from_config()` creates an instance from a YAML file or dictionary.

## Basic Usage

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

datamodule = MVTecAD(category="bottle")
model = Patchcore()
engine = Engine(max_epochs=1)

# Train and Test
engine.fit(datamodule=datamodule, model=model)
engine.test(datamodule=datamodule, model=model)

# Inference and Export
engine.predict(datamodule=datamodule, model=model)
engine.export(model=model, export_type=ExportType.OPENVINO)
```

## How Engine Wires Components

The engine automatically handles callback registration via `_setup_anomalib_callbacks()`:

- **Automatic Callbacks**: `ModelCheckpoint` (if missing) and `TimerCallback` (always).
- **Dual-Nature Components**: `PreProcessor`, `PostProcessor`, `Evaluator`, and `Visualizer` are both `nn.Module` and `Callback`.
- **Registration**: These components are registered through `AnomalibModule.configure_callbacks()` and added to the Lightning Trainer's callback list.
- **Arguments**: Engine merges `model.trainer_arguments` (like `max_epochs`) with any user-provided arguments.

## LearningType and Special Flows

The engine inspects `model.learning_type` to adjust the execution flow:

- **ONE_CLASS**: Standard training using only normal images.
- **ZERO_SHOT**: Training is skipped. Engine bypasses `fit()` and goes directly to testing or prediction.
- **FEW_SHOT**: Optimization for minimal training data.

## CLI Usage

Anomalib provides a CLI for running the engine:

```bash
# Using config files
anomalib fit -c examples/configs/model/patchcore.yaml --data examples/configs/data/mvtec.yaml

# Using CLI arguments
anomalib fit --model Patchcore --data MVTecAD --data.category bottle
```

## Key Files

- **Engine Core**: `src/anomalib/engine/engine.py`
- **XPU Support**: `src/anomalib/engine/accelerator/xpu.py`
- **Configs**: `examples/configs/model/*.yaml`, `examples/configs/data/*.yaml`
- **API Example**: `examples/api/01_getting_started/basic_training.py`
