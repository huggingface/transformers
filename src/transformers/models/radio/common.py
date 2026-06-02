# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import dataclass

from .radio_model import Resolution


@dataclass
class RadioResource:
    url: str
    patch_size: int
    max_resolution: int
    preferred_resolution: Resolution
    supports_vitdet: bool = True
    vitdet_num_windowed: int | None = None
    vitdet_num_global: int | None = None


RESOURCE_MAP = {
    # RADIOv2.5
    "radio_v2.5-b": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-b_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-l": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio-v2.5-l_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-h": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-h-norm": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-h-norm.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=4,
    ),
    "radio_v2.5-g": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.5-g.pth.tar?download=true",
        patch_size=14,
        max_resolution=1792,
        preferred_resolution=(896, 896),
        vitdet_num_global=8,
    ),
    # RADIO
    "radio_v2.1": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.1_bf16.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(432, 432),
        vitdet_num_windowed=5,
    ),
    "radio_v2": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v2.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(432, 432),
        vitdet_num_windowed=5,
    ),
    "radio_v1": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/radio_v1.pth.tar?download=true",
        patch_size=14,
        max_resolution=1050,
        preferred_resolution=Resolution(378, 378),
    ),
    # E-RADIO
    "e-radio_v2": RadioResource(
        "https://huggingface.co/nvidia/RADIO/resolve/main/eradio_v2.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
    ),
    # C-RADIO
    "c-radio_v2-g": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv2-g for more information.
        "https://huggingface.co/nvidia/C-RADIOv2-g/resolve/main/c-radio_v2-g_half.pth.tar",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=8,
    ),
    "c-radio_v2-vlm-h": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv2-VLM-H for more information.
        "https://huggingface.co/nvidia/C-RADIOv2-VLM-H/resolve/main/c-radio_v2-vlm-h.pth.tar",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=(768, 768),
        vitdet_num_global=8,
    ),
    "c-radio_v3-b": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv3-B for more information.
        "https://huggingface.co/nvidia/C-RADIOv3-B/resolve/main/c-radio_v3-b_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
    "c-radio_v3-l": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv3-L for more information.
        "https://huggingface.co/nvidia/C-RADIOv3-L/resolve/main/c-radio-v3_l_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
    "c-radio_v3-h": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv3-H for more information.
        "https://huggingface.co/nvidia/C-RADIOv3-H/resolve/main/c-radio_v3-h_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
    "c-radio_v3-g": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv3-g for more information.
        "https://huggingface.co/nvidia/C-RADIOv3-g/resolve/main/c-radio_v3-g_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
    "c-radio_v4-so400m": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv4-SO400M for more information.
        "https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/main/c-radio_v4-so400m_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
    "c-radio_v4-h": RadioResource(
        # NOTE: C-RADIO models are bound by different license terms than that present in the LICENSE file.
        # Please refer to the readme, or to https://huggingface.co/nvidia/C-RADIOv4-H for more information.
        "https://huggingface.co/nvidia/C-RADIOv4-H/resolve/main/c-radio_v4-h_half.pth.tar?download=true",
        patch_size=16,
        max_resolution=2048,
        preferred_resolution=Resolution(512, 512),
        supports_vitdet=False,
    ),
}

DEFAULT_VERSION = "c-radio_v4-h"
