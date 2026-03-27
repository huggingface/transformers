# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for PoolFormer."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_poolformer import PoolFormerImageProcessorKwargs


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class PoolFormerImageProcessorPil(PilBackend):
    """PIL backend for PoolFormer with custom resize (crop_pct)."""

    valid_kwargs = PoolFormerImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    crop_pct = 0.9
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[PoolFormerImageProcessorKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None" = None,
        crop_pct: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize with crop_pct: scale size by 1/crop_pct when crop_pct is set."""

        if crop_pct is not None:
            if size.shortest_edge:
                scale_size = int(size.shortest_edge / crop_pct)
            elif size.height and size.width:
                if size.height == size.width:
                    scale_size = int(size.height / crop_pct)
                else:
                    scale_size = (int(size.height / crop_pct), int(size.width / crop_pct))
            else:
                raise ValueError(f"Invalid size for resize: {size}")
            output_size = get_resize_output_image_size(
                image, size=scale_size, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=output_size[0], width=output_size[1])
        return super().resize(
            image,
            size=size,
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
        crop_pct: float = 0.9,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for PoolFormer (pass crop_pct to resize)."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, crop_pct=crop_pct)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["PoolFormerImageProcessorPil"]
