# AGENTS.md — Anomalib

Anomalib is a deep learning library for anomaly detection, supporting image and video modalities. Built on PyTorch Lightning. Supports export to OpenVINO, ONNX, and PyTorch formats for deployment.

## Essential Commands

```bash
# Install (development)
uv sync --extra dev --extra cpu    # or cu124, cu126, cu118, cu130, rocm, xpu

# Install (production)
uv pip install "anomalib[cpu]"     # or cu124, cu126, cu118, cu130, rocm, xpu

# Install with optional features
uv pip install "anomalib[openvino,cu124]"

# Run tests
pytest tests/unit/path/to/test.py -v

# Lint
pre-commit run --files <file1> [file2 ...]

# Type check
mypy src/anomalib/
```

## Routing — Read the Relevant File

| Working In                      | Read                                       |
| ------------------------------- | ------------------------------------------ |
| `src/anomalib/models/`          | `docs/agent-context/models.md`             |
| `src/anomalib/data/`            | `docs/agent-context/data.md`               |
| `src/anomalib/engine/`          | `docs/agent-context/engine.md`             |
| `src/anomalib/deploy/`          | `docs/agent-context/deploy.md`             |
| `src/anomalib/callbacks/`       | `docs/agent-context/callbacks.md`          |
| `src/anomalib/metrics/`         | `docs/agent-context/metrics.md`            |
| `src/anomalib/pre_processing/`  | `docs/agent-context/pre-processing.md`     |
| `src/anomalib/post_processing/` | `docs/agent-context/post-processing.md`    |
| `src/anomalib/visualization/`   | `docs/agent-context/visualization.md`      |
| Porting a new model             | `docs/agent-context/porting-models.md`     |
| OpenVINO/C++ inference          | `docs/agent-context/openvino-inference.md` |
| General architecture overview   | `docs/agent-context/index.md`              |

## Core Architecture

- All models inherit from `AnomalibModule` (extends `ExportMixin + pl.LightningModule`).
- Forward pipeline: `pre_processor(batch) → model(batch) → post_processor(batch)`.
- PreProcessor, PostProcessor, and Evaluator are BOTH `nn.Module` AND Lightning `Callback` (dual-nature pattern).
- Models override `configure_pre_processor()`, `configure_post_processor()`, `configure_evaluator()`, `configure_visualizer()`.
- Component resolution: `True` = use default, `False` = disable, instance = use that instance.
- LearningType: ONE_CLASS, ZERO_SHOT, FEW_SHOT.
- InferenceBatch: NamedTuple with pred_score, pred_label, anomaly_map, pred_mask.

## Non-Obvious Conventions

- Always inherit from AnomalibModule for new models.
- Each model has two files: torch_model.py (pure PyTorch nn.Module) and lightning_model.py (AnomalibModule wrapper).
- Config-driven: every model gets a YAML config in examples/configs/model/.
- Export bakes pre/post-processing into the model graph (via exportable transforms).
- Use ExportableCenterCrop instead of CenterCrop for export compatibility.
- The Engine class wraps Lightning Trainer. Don't use Trainer directly.
- Folder datamodule enables custom datasets without writing new dataset classes.

## Boundaries

### Always

- Add type hints to all function signatures.
- Write unit tests for new features.
- Run pre-commit before committing.
- Register new models in `src/anomalib/models/__init__.py`.

### Never

- Don't use `as any`, `@ts-ignore` equivalents.
- Don't modify the Engine without understanding Lightning callbacks.
- Don't hardcode paths. Use pathlib.
- Don't add new dependencies without checking existing deps.
- Don't skip the dual-nature pattern (nn.Module + Callback) when creating processing components.
