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


from typing import Optional, Union

import torch

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    SizeDict,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging
from .image_processing_idefics2 import convert_to_rgb


if is_torchvision_available():
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


def get_resize_output_image_size(image: "torch.Tensor", size: SizeDict) -> tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`torch.Tensor`):
            Image to resize.
        size (`SizeDict`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".

    Returns:
        The output size of the image after resizing.
    """
    height, width = image.size()[-2:]

    min_len = size.shortest_edge
    max_len = size.longest_edge
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def get_max_height_width(images_list: list[list["torch.Tensor"]]) -> tuple[int, int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(image.size()[-2:])

    max_height = max(size[0] for size in image_sizes)
    max_width = max(size[1] for size in image_sizes)
    return (max_height, max_width)


def make_pixel_mask(image: "torch.Tensor", output_size: tuple[int, int]) -> "torch.Tensor":
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`torch.Tensor`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = image.size()[-2:]
    mask = torch.zeros(output_size, dtype=torch.int64, device=image.device)
    mask[:input_height, :input_width] = 1
    return mask


class Idefics2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    do_image_splitting (`bool`, *optional*, defaults to `False`):
        Whether to split the image into a sequence 4 equal sub-images concatenated with the original image.
    do_pad (`bool`, *optional*, defaults to `True`):
        Whether to pad images to the largest height and width in the batch.
    """

    do_image_splitting: Optional[bool]
    do_pad: Optional[bool]


@auto_docstring
class Idefics2ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    do_convert_rgb = True
    do_image_splitting = False
    size = {"shortest_edge": 378, "longest_edge": 980}
    model_input_names = ["pixel_values", "pixel_attention_mask"]
    valid_kwargs = Idefics2FastImageProcessorKwargs

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """
        Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
        as is.
        """
        return convert_to_rgb(image)

    def resize(
        self, image: torch.Tensor, size: SizeDict, interpolation: Optional["F.InterpolationMode"] = None, **kwargs
    ) -> torch.Tensor:
        """
        Resize an image using torchvision's functional resize.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR

        if size.shortest_edge and size.longest_edge:
            new_size = get_resize_output_image_size(image, size)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError("Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys.")

        image = F.resize(image, size=new_size, interpolation=interpolation, **kwargs)
        return image

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        """
        Prepare a nested images structure for processing.
        """
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def split_images(
        self,
        images: "torch.Tensor",
    ) -> list["torch.Tensor"]:
        """
        Split a batch of images into 4 equal sub-images, and concatenate that sequence with the original image.
        """
        height, width = images.size()[-2:]

        mid_width = width // 2
        mid_height = height // 2

        batch_split_images = [
            images[..., :mid_height, :mid_width],
            images[..., :mid_height, mid_width:],
            images[..., mid_height:, :mid_width],
            images[..., mid_height:, mid_width:],
            images,
        ]

        # transpose the batch dimension to the first dimension
        batch_split_images = [[image[i] for image in batch_split_images] for i in range(len(batch_split_images[0]))]
        return batch_split_images

    def pad(
        self, image: "torch.Tensor", padded_size: tuple[int, int], fill: int = 0
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Pad an image to the specified size and create the corresponding pixel mask.
        """
        original_size = image.shape[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]

        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )

        # Only pad if necessary
        if original_size != padded_size:
            # torchvision's pad takes a 4-element tuple for 2D padding: (left, top, right, bottom)
            padding = (0, 0, padding_right, padding_bottom)
            # Use constant padding to match slow implementation
            image = F.pad(image, padding, fill=fill, padding_mode="constant")

        # Create pixel mask to match the slow implementation
        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Idefics2FastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: Optional[bool],
        do_image_splitting: Optional[bool],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Process a batch of images for the model.
        """
        grouped_images, grouped_images_index = group_images_by_shape(
            images, is_nested=True, disable_grouping=disable_grouping
        )
        split_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_image_splitting:
                stacked_images = self.split_images(stacked_images)
            split_images_grouped[shape] = stacked_images
        split_images = reorder_images(split_images_grouped, grouped_images_index, is_nested=True)
        if do_image_splitting:
            # flattenened the doubly nested list to a nested list
            for i, group_images in enumerate(split_images):
                split_images[i] = [image for sublist in group_images for image in sublist]

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(
            split_images, is_nested=True, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index, is_nested=True)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, is_nested=True, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)

        if do_pad:
            # Get max images per batch
            max_num_images = max(len(images_) for images_ in processed_images)
            max_height, max_width = get_max_height_width(processed_images)

            processed_images_padded = torch.zeros(
                len(processed_images),
                max_num_images,
                *(processed_images[0][0].shape[0], max_height, max_width),
                device=processed_images[0][0].device,
            )
            pixel_attention_masks = torch.zeros(
                len(processed_images),
                max_num_images,
                *(max_height, max_width),
                device=processed_images[0][0].device,
            )
            for i, images in enumerate(processed_images):
                for j, image in enumerate(images):
                    processed_images_padded[i, j], pixel_attention_masks[i, j] = self.pad(
                        image, (max_height, max_width)
                    )
            processed_images = processed_images_padded
        if do_pad:
            data = {"pixel_values": processed_images, "pixel_attention_mask": pixel_attention_masks}
        elif return_tensors == "pt":
            data = {"pixel_values": torch.stack([torch.stack(images) for images in processed_images])}
        else:
            data = {"pixel_values": processed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Idefics2ImageProcessorFast"]
