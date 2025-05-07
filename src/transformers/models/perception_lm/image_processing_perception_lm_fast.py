# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for PerceptionLM."""

from typing import List, Optional, Tuple, Union

import numpy as np

from transformers.models.perception_lm.image_transform import get_image_transform

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    VideoInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)


if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class PerceptionLMFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    vision_input_type: str = "thumb+tile"
    image_res: int = 448
    max_num_tiles: int = 36
    normalize_img: bool = True
    return_tensors: Optional[Union[str, TensorType]] = None


@add_start_docstrings(
    "Constructs a fast PerceptionLM image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pad (`bool`, *optional*, defaults to `self.do_pad`):
            Whether to pad the image to a square based on the longest edge. Can be overridden by the `do_pad` parameter
    """,
)
class PerceptionLMImageProcessorFast(BaseImageProcessorFast):
    valid_kwargs = PerceptionLMFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)
        self.image_transform = get_image_transform(
            vision_input_type=kwargs.get("vision_input_type", "thumb+tile"),
            image_res=kwargs.get("image_res", 448),
            max_num_tiles=kwargs.get("max_num_tiles", 36),
            normalize_img=kwargs.get("normalize_img", True),
        )
        self.video_transform = get_image_transform(
            vision_input_type="vanilla",
            image_res=kwargs.get("image_res", 448),
            max_num_tiles=kwargs.get("max_frame_tiles", 1),
            normalize_img=kwargs.get("normalize_img", True),
        )

    def to_dict(self):
        dictionary = super().to_dict()
        dictionary["image_transform"] = self.image_transform.to_dict()
        dictionary["video_transform"] = self.video_transform.to_dict()
        return dictionary

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to a square based on the longest edge. Can be overridden by the `do_pad` parameter
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)


    def _preprocess(
        self,
        images: List["torch.Tensor"],
        videos: List["torch.Tensor"],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]
    ) -> BatchFeature:
        # Group images by size for batched transformation
        del kwargs
        if images:
            grouped_images, grouped_images_index = group_images_by_shape(images)
            processed_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                stacked_images, _ = self.image_transform(stacked_images)
                print("stacked_images shape: ", stacked_images.shape)
                processed_images_grouped[shape] = stacked_images
            processed_images = reorder_images(processed_images_grouped, grouped_images_index)
            processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
            return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)
        elif videos:
            videos = [torch.from_numpy(np.array(v)).flatten(0, 1).permute(0, 3, 1, 2) for v in videos]
            processed_videos = [self.video_transform(v)[0].squeeze(1) for v in videos]
            processed_videos = torch.stack(processed_videos, dim=0) if return_tensors else processed_videos
            return BatchFeature(data={"pixel_values": processed_videos}, tensor_type=return_tensors)
        else:
            return BatchFeature(data={"pixel_values": None}, tensor_type=return_tensors)


__all__ = ["PerceptionLMImageProcessorFast"]
