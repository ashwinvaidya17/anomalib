"""Test OpenVINO inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from importlib.util import find_spec
import torch
import pytest

from anomalib.deploy import ExportMode, export
from anomalib.models import get_model
from anomalib.trainer import AnomalibTrainer

sys.path.append("tools/inference")


@pytest.mark.order(5)
class TestOpenVINOInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from openvino_inference.py"""
        if find_spec("openvino_inference") is not None:
            from tools.inference.openvino_inference import get_parser, infer
        else:
            raise Exception("Unable to import openvino_inference.py for testing")
        return get_parser, infer

    def test_openvino_inference(
        self, get_functions, get_config, project_path, get_dummy_inference_image, transforms_config
    ):
        """Test openvino_inference.py"""
        get_parser, infer = get_functions

        model = get_model(get_config("padim"))

        trainer = AnomalibTrainer()
        trainer.normalization_connector.metric.max = torch.tensor(1.0)
        trainer.normalization_connector.metric.min = torch.tensor(0.0)

        # export OpenVINO model
        export(
            transform=transforms_config,
            trainer=trainer,
            input_size=(100, 100),
            model=model,
            export_mode=ExportMode.OPENVINO,
            export_root=project_path,
        )

        arguments = get_parser().parse_args(
            [
                "--config",
                "src/anomalib/models/padim/config.yaml",
                "--weights",
                project_path + "/weights/openvino/model.bin",
                "--metadata",
                project_path + "/weights/openvino/metadata.json",
                "--input",
                get_dummy_inference_image,
                "--output",
                project_path + "/output.png",
            ]
        )
        infer(arguments)
