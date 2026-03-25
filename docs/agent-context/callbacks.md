# Callbacks — Agent Context

Anomalib extends the PyTorch Lightning callback system, adding custom hooks for anomaly detection workflows and model export.

## Callback Architecture

- Anomalib components (PreProcessor, PostProcessor, Evaluator, Visualizer) follow the **dual-nature pattern**: they are both `nn.Module` and `pl.Callback`.
- **As Callback**: They hook into the Lightning lifecycle (e.g., `on_train_batch_start`, `on_validation_epoch_end`).
- **As nn.Module**: They are part of the model graph during export, where `forward()` is traced.
- Component registration happens via `AnomalibModule.configure_callbacks()`.
- Additional utility callbacks are added by the `Engine` via `_setup_anomalib_callbacks()`.

## Built-in Callbacks

| Callback                   | File                               | Purpose                      | Key Hooks                       |
| -------------------------- | ---------------------------------- | ---------------------------- | ------------------------------- |
| ModelCheckpoint            | `callbacks/checkpoint.py`          | Save best model weights      | (Lightning built-in)            |
| GraphLogger                | `callbacks/graph.py`               | Log model graph to logger    | `on_train_start`                |
| LoadModelCallback          | `callbacks/model_loader.py`        | Load model weights from path | `setup`                         |
| TilerConfigurationCallback | `callbacks/tiler_configuration.py` | Configure image tiling       | `setup`                         |
| TimerCallback              | `callbacks/timer.py`               | Log time & throughput        | `on_fit_start`, `on_test_start` |
| NNCFCallback               | `callbacks/nncf/callback.py`       | NNCF model compression       | `setup`, `on_train_batch_start` |

## Dual-Nature Components (nn.Module + Callback)

| Component     | File                                | Callback Hooks                     | Module Usage                |
| ------------- | ----------------------------------- | ---------------------------------- | --------------------------- |
| PreProcessor  | `pre_processing/pre_processor.py`   | `on_*_batch_start`                 | `forward()` in export graph |
| PostProcessor | `post_processing/post_processor.py` | `on_*_batch_end`, `on_*_epoch_end` | `forward()` in export graph |
| Evaluator     | `metrics/evaluator.py`              | `on_validation_epoch_end`          | Holds metric modules        |
| Visualizer    | `visualization/image/visualizer.py` | `on_test_batch_end`                | Visualization logic         |

> See also: [pre-processing.md](pre-processing.md) and [post-processing.md](post-processing.md) for specific implementation details.

## Creating a Custom Callback

```python
from lightning.pytorch import Callback

class MyCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Access model via pl_module
        # Access metrics via trainer.callback_metrics
        print(f"Epoch {trainer.current_epoch} done")

# Register with Engine
engine = Engine(callbacks=[MyCallback()])
```

## How Callbacks Are Registered

1. **Model components**: `AnomalibModule.configure_callbacks()` collects the dual-nature components (PreProcessor, PostProcessor, Evaluator, Visualizer).
2. **Engine defaults**: `_setup_anomalib_callbacks()` adds `ModelCheckpoint` and `TimerCallback`.
3. **User callbacks**: Passed via `Engine(callbacks=[...])` or defined in the config.
4. **NNCF callback**: Dynamically loaded if an NNCF configuration is provided.

## Key Files

- **Registry**: `src/anomalib/callbacks/__init__.py`
- **Core Callbacks**: `src/anomalib/callbacks/checkpoint.py`, `graph.py`, `model_loader.py`, `tiler_configuration.py`, `timer.py`
- **NNCF Support**: `src/anomalib/callbacks/nncf/callback.py`
