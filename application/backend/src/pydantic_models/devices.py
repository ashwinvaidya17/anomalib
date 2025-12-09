# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class DeviceList(BaseModel):
    devices: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": ["CPU", "XPU", "NPU"],
            }
        }
    }


class Camera(BaseModel):
    index: int
    name: str
    backend: int  # eg cv2.CAP_DSHOW


class CameraList(BaseModel):
    devices: list[Camera]

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": [
                    {"index": 1200, "name": "camera_name1", "backend": 1},
                    {"index": 1201, "name": "camera_name2", "backend": 1},
                    {"index": 1202, "name": "camera_name3", "backend": 0},
                ]
            }
        }
    }
