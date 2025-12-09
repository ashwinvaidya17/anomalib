# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2

from entities.base_opencv_stream import BaseOpenCVStream
from pydantic_models import SourceType


class WebcamStream(BaseOpenCVStream):
    """Video stream implementation using webcam via OpenCV."""

    def __init__(self, device_id: int = 0, backend: int = cv2.CAP_ANY) -> None:
        """Initialize webcam stream."""
        self.backend = backend
        super().__init__(source=device_id, source_type=SourceType.WEBCAM, device_id=device_id)

    def _initialize_capture(self) -> None:
        """Initialize the OpenCV VideoCapture."""
        self.cap = cv2.VideoCapture(self.source, apiPreference=self.backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source} with backend: {self.backend}")

    def is_real_time(self) -> bool:
        return True
