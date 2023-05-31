"""Test Gradio inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from importlib.util import find_spec

import pytest

from anomalib.deploy import ExportMode, OpenVINOInferencer, TorchInferencer, export
from anomalib.models import get_model

sys.path.append("tools/inference")


@pytest.mark.order(6)
class TestGradioInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs.

    Note: This does not launch the gradio server. It only checks if the right inferencer is called.
    """

    @pytest.fixture
    def get_functions(self):
        """Get functions from Gradio_inference.py"""
        if find_spec("gradio_inference") is not None:
            from tools.inference.gradio_inference import get_inferencer, get_parser
        else:
            raise Exception("Unable to import gradio_inference.py for testing")
        return get_parser, get_inferencer

    def test_torch_inference(self, get_functions, project_path, get_config, trainer):
        """Test gradio_inference.py"""
        parser, inferencer = get_functions
        model = get_model(get_config("padim"))

        # export torch model
        export(
            trainer=trainer,
            input_size=(100, 100),
            model=model,
            export_mode=ExportMode.TORCH,
            export_root=project_path,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                project_path + "/weights/torch/model.pt",
            ]
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), TorchInferencer)

    def test_openvino_inference(self, get_functions, project_path, get_config, trainer):
        """Test gradio_inference.py"""
        parser, inferencer = get_functions
        model = get_model(get_config("padim"))

        # export OpenVINO model
        export(
            trainer=trainer,
            input_size=(100, 100),
            model=model,
            export_mode=ExportMode.OPENVINO,
            export_root=project_path,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                project_path + "/weights/openvino/model.bin",
                "--metadata",
                project_path + "/weights/openvino/metadata.json",
            ]
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), OpenVINOInferencer)

        # test error is raised when metadata is not provided to openvino model
        with pytest.raises(ValueError):
            arguments = parser().parse_args(
                [
                    "--weights",
                    project_path + "/weights/openvino/model.bin",
                ]
            )
            inferencer(arguments.weights, arguments.metadata)
