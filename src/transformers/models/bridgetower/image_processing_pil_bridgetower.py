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
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_bridgetower import BridgeTowerImageProcessorKwargs, get_resize_output_image_size


@requires(backends=("vision", "torch", "torchvision"))
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
        resample: "PILImageResampling | int | None",
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
        resample: "PILImageResampling | int | None",
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
