"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import LightningArgumentParser
from torch.utils.data import DataLoader

from anomalib.data.inference import InferenceDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyModule, get_model


def get_parser() -> LightningArgumentParser:
    """Get parser.

    Returns:
        LightningArgumentParser: The parser object.
    """
    parser = LightningArgumentParser(description="Inference on Anomaly models in Lightning format.")
    parser.add_lightning_class_args(AnomalyModule, "model", subclass_mode=True)
    parser.add_lightning_class_args(Callback, "--callbacks", subclass_mode=True, required=False)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model weights")
    parser.add_class_arguments(InferenceDataset, "--data", instantiate=False)
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )

    return parser


def infer(args: Namespace) -> None:
    """Run inference."""
    visualization = {"show_images": args.show, "mode": args.visualization_mode, "save_images": False}
    if args.output:  # overwrite save path
        visualization["save_images"] = True
        visualization["image_save_path"] = args.output

    callbacks = None if not hasattr(args, "callbacks") else args.callbacks
    engine = Engine(callbacks=callbacks, visualization=visualization, devices=1)
    model = get_model(args.model)

    # create the dataset
    dataset = InferenceDataset(**args.data)
    dataloader = DataLoader(dataset)

    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    infer(args)
