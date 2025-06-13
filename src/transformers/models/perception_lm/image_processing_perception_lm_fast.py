# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for PerceptionLM."""

from typing import List, Optional, Tuple, Union

import numpy as np

from transformers.models.perception_lm.image_transform import get_image_transform

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
)
from ...image_utils import PILImageResampling, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

if is_torch_available():
    import torch

class PerceptionLMFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    vision_input_type: str = "thumb+tile"
    tile_size: int = 448
    max_num_tiles: int = 36


@add_start_docstrings(
    "Constructs a fast PerceptionLM image processor.",
)
class PerceptionLMImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = PerceptionLMFastImageProcessorKwargs


    def __init__(self, **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)
        self.image_transform = get_image_transform(
            vision_input_type=self.vision_input_type,
            image_res=self.tile_size,
            max_num_tiles=self.max_num_tiles,
        )

    def to_dict(self):
        dictionary = super().to_dict()
        dictionary["image_transform"] = self.image_transform.to_dict()
        return dictionary

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        do_rescale: Optional[bool],
        rescale_factor: Optional[Union[int, float]],
        do_normalize: Optional[bool],
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]
    ) -> BatchFeature:
        # Group images by size for batched transformation
        if images:
            grouped_images, grouped_images_index = group_images_by_shape(images)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                if do_resize:
                    stacked_images, _ = self.image_transform(stacked_images)
                resized_images_grouped[shape] = stacked_images
            resized_images = reorder_images(resized_images_grouped, grouped_images_index)
                
            grouped_images, grouped_images_index = group_images_by_shape(resized_images)
            processed_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                # Fused rescale and normalize
                stacked_images = self.rescale_and_normalize(
                    stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_images_grouped[shape] = stacked_images
            processed_images = reorder_images(processed_images_grouped, grouped_images_index)
            
            processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
            return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)
        else:
            return BatchFeature(data={"pixel_values": None}, tensor_type=return_tensors)


__all__ = ["PerceptionLMImageProcessorFast"]
