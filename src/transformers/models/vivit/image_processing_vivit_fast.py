# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Fast Image processor class for Vivit."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import torch


class VivitFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    offset (`bool`, *optional*):
        Whether to scale the image in both negative and positive directions. If `True`, the image has its values
        rescaled by `rescale_factor` and then offset by 1. If `rescale_factor` is 1/127.5, the image is rescaled
        between [-1, 1]. If `False`, and `rescale_factor` is 1/255, the image is rescaled between [0, 1].
    """

    offset: Optional[bool]


@auto_docstring
class VivitImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values"]
    valid_kwargs = VivitFastImageProcessorKwargs

    # Default values from the slow image processor
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 127.5
    offset = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[VivitFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images, **kwargs: Unpack[VivitFastImageProcessorKwargs]):
        return super().preprocess(images, **kwargs)

    def rescale(self, image, scale: float, offset: bool = True, **kwargs) -> "torch.Tensor":
        """
        Rescale an image by a scale factor with optional offset.

        Args:
            image:
                Image to rescale. Can be numpy array or torch tensor.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            offset (`bool`, *optional*, defaults to `True`):
                Whether to scale the image in both negative and positive directions.

        Returns:
            `torch.Tensor`: The rescaled image.
        """
        # Convert to torch tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        # Match the slow processor behavior: first rescale, then apply offset
        rescaled_image = image * scale

        if offset:
            rescaled_image = rescaled_image - 1.0

        return rescaled_image

    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        offset: bool = True,
    ) -> "torch.Tensor":
        """
        Rescale and normalize images with optional offset.

        Args:
            images (`torch.Tensor`):
                Images to rescale and normalize.
            do_rescale (`bool`):
                Whether to rescale the images.
            rescale_factor (`float`):
                The scaling factor to rescale pixel values by.
            do_normalize (`bool`):
                Whether to normalize the images.
            image_mean (`float` or `list[float]`):
                Mean to use for normalization.
            image_std (`float` or `list[float]`):
                Standard deviation to use for normalization.
            offset (`bool`, *optional*, defaults to `True`):
                Whether to scale the image in both negative and positive directions.

        Returns:
            `torch.Tensor`: The rescaled and normalized images.
        """
        # Apply rescaling and normalization in sequence to match slow processor
        # Don't use fused approach for ViViT due to offset parameter
        if do_rescale:
            images = self.rescale(images, rescale_factor, offset=offset)

        if do_normalize:
            images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)

        return images

    def _prepare_images_structure(
        self,
        images: ImageInput,
        **kwargs,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_nested_list_of_images(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size,
        interpolation,
        do_center_crop: bool,
        crop_size,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        disable_grouping: Optional[bool],
        return_tensors,
        offset: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos using the fast image processor with batched processing.
        """
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        processed_images_grouped = {}
        for shape, stacked_frames in grouped_images.items():
            # Resize if needed
            if do_resize:
                stacked_frames = self.resize(stacked_frames, size, interpolation)

            # Center crop if needed
            if do_center_crop:
                stacked_frames = self.center_crop(stacked_frames, crop_size)

            # Rescale and normalize with offset parameter
            stacked_frames = self.rescale_and_normalize(
                stacked_frames, do_rescale, rescale_factor, do_normalize, image_mean, image_std, offset=offset
            )

            processed_images_grouped[shape] = stacked_frames

        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)
        if return_tensors == "pt":
            processed_images = [torch.stack(images, dim=0) for images in processed_images]
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["VivitImageProcessorFast"]
