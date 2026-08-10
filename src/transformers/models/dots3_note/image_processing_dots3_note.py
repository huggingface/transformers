# Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor for Dots 3 Note Preview."""

import math

from PIL import Image

from ...image_processing_backends import PilBackend
from ...image_utils import PILImageResampling, SizeDict
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Resize exactly as the Dots 3 Note Preview training and serving preprocessors."""
    if min(height, width) < factor // 4:
        raise ValueError(f"Image height and width must be at least {factor // 4}, got {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must be smaller than 200")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
        if resized_height * resized_width > max_pixels:
            scale = math.sqrt(resized_height * resized_width / max_pixels)
            resized_height = max(factor, math.floor(resized_height / scale / factor) * factor)
            resized_width = max(factor, math.floor(resized_width / scale / factor) * factor)
    return resized_height, resized_width


class Dots3NoteImageProcessor(Qwen2VLImageProcessorPil):
    """PIL preprocessing numerically identical to the published Dots 3 Note Preview path."""

    def convert_to_rgb(self, image):
        if not isinstance(image, Image.Image):
            return image
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            return background
        return image if image.mode == "RGB" else image.convert("RGB")

    def resize(
        self,
        image,
        size: SizeDict,
        resample: PILImageResampling | int | None,
        factor: int,
        **kwargs,
    ):
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` must contain `shortest_edge` and `longest_edge`, got {size}")
        height, width = image.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return PilBackend.resize(
            self,
            image=image,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )


__all__ = ["Dots3NoteImageProcessor"]
