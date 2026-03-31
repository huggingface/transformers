# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for BridgeTower."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.bridgetower.image_processing_bridgetower.BridgeTowerImageProcessorKwargs
class BridgeTowerImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
        The size by which to make sure both the height and width can be divided.
    """

    size_divisor: int


# adapted from transformers.models.bridgetower.image_processing_bridgetower.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: np.ndarray,
    shorter: int = 800,
    longer: int = 1333,
    size_divisor: int = 32,
) -> tuple[int, int]:
    """Get output image size after resizing with size_divisor."""
    input_height, input_width = get_image_size(input_image, channel_dim=ChannelDimension.FIRST)

    min_size, max_size = shorter, longer
    scale = min_size / min(input_height, input_width)

    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    return new_height, new_width


@auto_docstring
class BridgeTowerImageProcessorPil(PilBackend):
    """PIL backend for BridgeTower with custom resize and center_crop."""

    valid_kwargs = BridgeTowerImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask"]

    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 288}
    default_to_square = False
    crop_size = {"shortest_edge": 288}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    size_divisor = 32

    def __init__(self, **kwargs: Unpack[BridgeTowerImageProcessorKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | None",
        size_divisor: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Resize with size_divisor support."""

        if not size.shortest_edge:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        shorter = size.shortest_edge
        longer = int(1333 / 800 * shorter)
        output_height, output_width = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor
        )
        return super().resize(
            image,
            size=SizeDict(height=output_height, width=output_width),
            resample=resample,
            **kwargs,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        size_divisor: int = 32,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for BridgeTower."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, size_divisor)
            if do_center_crop:
                image = self.center_crop(
                    image, size=SizeDict(height=crop_size.shortest_edge, width=crop_size.shortest_edge)
                )
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        data = {}
        if do_pad:
            processed_images, processed_masks = self.pad(
                processed_images,
                return_mask=True,
            )
            data["pixel_mask"] = processed_masks

        data["pixel_values"] = processed_images

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["BridgeTowerImageProcessorPil"]
