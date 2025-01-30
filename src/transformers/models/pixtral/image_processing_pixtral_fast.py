# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Pixtral."""

from typing import Dict, List, Optional, Union

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .image_processing_pixtral import (
    get_resize_output_image_size,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_vision_available():
        pass

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


@add_start_docstrings(
    r"Constructs a fast ConvNeXT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        patch_size (`Dict[str, int]` *optional*, defaults to `{"height": 16, "width": 16}`):
            Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.

    """,
)
class PixtralImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    patch_size = {"height": 16, "width": 16}
    size = {"longest_edge": 1024}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_extra_kwargs = ["patch_size"]

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        patch_size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dict containing the longest possible edge of the image.
            patch_size (`SizeDict`):
                Patch size used to calculate the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Resampling filter to use when resiizing the image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.longest_edge:
            size = (size.longest_edge, size.longest_edge)
        elif size.height and size.width:
            size = (size.height, size.width)
        else:
            raise ValueError("size must contain either 'longest_edge' or 'height' and 'width'.")

        if patch_size.height and patch_size.width:
            patch_size = (patch_size.height, patch_size.width)
        else:
            raise ValueError("patch_size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(image, size=size, patch_size=patch_size)
        return F.resize(image, size=output_size, interpolation=interpolation, **kwargs)

    # Adapted from transformers.models.pixtral.image_processing_pixtral.PixtralImageProcessor._pad_for_batching
    def _pad_for_batching(
        self,
        pixel_values: List[torch.Tensor],
        image_sizes: List[List[int]],
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.
        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `channels`, `height`, `width`)
            image_sizes (`List[List[int]]`):
                A list of sizes for each image in `pixel_values` in (height, width) format.
        Returns:
            List[`torch.Tensor`]: The padded images.
        """

        max_shape = (max([size[0] for size in image_sizes]), max([size[1] for size in image_sizes]))
        pixel_values = [
            torch.nn.functional.pad(image, pad=(0, max_shape[1] - size[1], 0, max_shape[0] - size[0]))
            for image, size in zip(pixel_values, image_sizes)
        ]
        return torch.stack(pixel_values)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        patch_size: Dict[str, int],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: Dict[str, int],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        patch_size = get_size_dict(patch_size, default_to_square=True)
        patch_size = SizeDict(**patch_size)
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, patch_size=patch_size, interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        batch_image_sizes = [grouped_images_index[i][0] for i in range(len(grouped_images_index))]
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        padded_images = self._pad_for_batching(
            pixel_values=processed_images,
            image_sizes=batch_image_sizes,
        )

        return BatchFeature(
            data={"pixel_values": padded_images, "image_sizes": batch_image_sizes}, tensor_type=return_tensors
        )


__all__ = ["PixtralImageProcessorFast"]
