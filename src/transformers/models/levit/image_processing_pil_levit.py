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
"""Image processor class for LeViT."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


@auto_docstring
class LevitImageProcessorPil(PilBackend):
    """PIL backend for LeViT with custom resize (shortest_edge * 256/224)."""

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize: shortest_edge is rescaled to int((256/224) * shortest_edge)."""
        if size.shortest_edge:
            shortest_edge = int((256 / 224) * size.shortest_edge)
            new_size_height, new_size_width = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=new_size_height, width=new_size_width)
        elif not size.height or not size.width:
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {list(size.keys())}."
            )
        return super().resize(
            image,
            size=size,
            resample=resample,
            **kwargs,
        )


__all__ = ["LevitImageProcessorPil"]
