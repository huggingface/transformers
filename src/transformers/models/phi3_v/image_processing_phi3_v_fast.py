# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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


from typing import Optional, Union

import numpy as np
import torch
import torchvision

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    filter_out_non_signature_kwargs,
)


class Phi3VFastImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    num_crops (`int`, *optional*, defaults to 4):
        The max number of crops to generate from each image.
    """

    num_crops: int = 4


def pad_to_max_num_crops(images, max_crops):
    """Pad each image to max crops."""
    num_crops, _, H, W = images.shape
    if num_crops < max_crops:
        pad = torch.zeros(max_crops - num_crops, 3, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images


@auto_docstring
class Phi3VImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_rescale = True
    do_resize = True
    do_normalize = True
    size = 336
    num_crops = 4
    valid_kwargs = Phi3VFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Phi3VFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def get_new_scaled_dims(self, image, size):
        height, width = image.size()[-2:]

        is_transposed = False
        if width < height:
            image = image.transpose(2, 3)
            is_transposed = True
            height, width = image.size()[-2:]

        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= self.num_crops:
            scale += 1
        scale -= 1
        new_width = int(scale * size)
        new_height = int(new_width / ratio)
        return new_height, new_width, is_transposed

    def _pad(self, image, size):
        """Pad the image to the target size."""
        height = image.size()[-2]
        target = int(np.ceil(height / size) * size)

        top_padding = int((target - height) / 2)
        bottom_padding = target - height - top_padding
        left_padding = 0
        right_padding = 0

        padding = [left_padding, top_padding, right_padding, bottom_padding]
        padded_image = torchvision.transforms.functional.pad(image, padding, fill=255)
        return padded_image

    def _resize(
        self,
        image,
        size: SizeDict,
        interpolation: PILImageResampling = PILImageResampling.BILINEAR,
        **kwargs,
    ) -> torch.Tensor:
        new_height, new_width, is_transposed = self.get_new_scaled_dims(image, size["height"])
        new_size = SizeDict(height=new_height, width=new_width)

        image = self.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs,
        )
        image = self._pad(image=image, size=size["height"])

        if is_transposed:
            image = image.transpose(2, 3)
        return image

    @filter_out_non_signature_kwargs()
    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        disable_grouping: Optional[bool] = None,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._resize(image=stacked_images, size=size, interpolation=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        images = reorder_images(processed_images_grouped, grouped_images_index)

        # Create global images from images by resizing.
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = torch.nn.functional.interpolate(
                stacked_images,
                size=(size["height"], size["width"]),
                mode="bicubic",
            )
            processed_images_grouped[shape] = stacked_images
        global_images = reorder_images(processed_images_grouped, grouped_images_index)

        images_combined, image_sizes = [], []
        #  Process each image to create local crops of shape (num_images, num_crops, 3, 336, 336)
        #  and combine with global image.
        for image, global_image in zip(images, global_images):
            h, w = image.shape[1], image.shape[2]
            reshaped = image.reshape(1, 3, h // size["height"], size["height"], w // size["width"], size["width"])
            permuted = reshaped.permute(0, 2, 4, 1, 3, 5)
            reshaped_final = permuted.reshape(-1, 3, size["height"], size["width"]).contiguous()
            combined_image = torch.cat([global_image.unsqueeze(0), reshaped_final], axis=0)
            images_combined.append(combined_image)
            image_sizes.append([h, w])

        # Calculate the number of image tokens for each image based on height and width dynamically.
        num_img_tokens = [
            int(((h // size["height"]) * (w // size["width"]) + 1) * 144 + 1 + (h // size["height"] + 1) * 12)
            for h, w in image_sizes
        ]

        # Pad all the images to max_crops.
        processed_images = [pad_to_max_num_crops(image, self.num_crops + 1) for image in images_combined]
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        encoded_outputs = BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes, "num_img_tokens": num_img_tokens},
            tensor_type=return_tensors,
        )
        return encoded_outputs


__all__ = ["Phi3VImageProcessorFast"]
