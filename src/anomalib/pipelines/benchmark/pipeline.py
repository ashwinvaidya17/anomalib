"""Benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .generator import BenchmarkJobGenerator


class Benchmark(Pipeline):
    """Benchmarking orchestrator."""

    def _setup_runners(self, args: Namespace) -> list[Runner]:
        """Setup the runners for the pipeline."""
        accelerators = args.accelerator if isinstance(args.accelerator, list) else [args.accelerator]
        runners: list[Runner] = []
        for accelerator in accelerators:
            if accelerator == "cpu":
                runners.append(SerialRunner(BenchmarkJobGenerator("cpu")))
            elif accelerator == "cuda":
                runners.append(
                    ParallelRunner(BenchmarkJobGenerator("cuda"), n_jobs=torch.cuda.device_count()),
                )
            else:
                msg = f"Unsupported accelerator: {accelerator}"
                raise ValueError(msg)
        return runners

    def get_parser(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """Add arguments to the parser."""
        parser = super().get_parser(parser)
        parser.add_argument(
            "--accelerator",
            type=str | list[str],
            default="cuda",
            help="Hardware to run the benchmark on.",
        )
        BenchmarkJobGenerator.add_arguments(parser)
        return parser
