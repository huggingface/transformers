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
"""Image processor class for Idefics2."""

import numpy as np
import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available, is_vision_available


if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


def get_resize_output_image_size(image, size: SizeDict) -> tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Images are always channels-first (CHW).
    """
    height, width = image.shape[-2:]

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


def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    """
    if not is_vision_available() or not isinstance(image, Image.Image):
        return image

    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class Idefics2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_image_splitting (`bool`, *optional*, defaults to `self.do_image_splitting`):
        Whether to split the image into a sequence 4 equal sub-images concatenated with the original image.
    """

    do_image_splitting: bool


def get_max_height_width(images_list: list[list["torch.Tensor|np.ndarray"]]) -> tuple[int, int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(image.shape[-2:])

    max_height = max(size[0] for size in image_sizes)
    max_width = max(size[1] for size in image_sizes)
    return (max_height, max_width)


def make_pixel_mask(image: "torch.Tensor", output_size: tuple[int, int]) -> "torch.Tensor":
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
    """
    input_height, input_width = image.shape[-2:]
    mask = torch.zeros(output_size, dtype=torch.int64, device=image.device)
    mask[:input_height, :input_width] = 1
    return mask


@auto_docstring
class Idefics2ImageProcessor(TorchvisionBackend):
    valid_kwargs = Idefics2ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    do_convert_rgb = True
    do_image_splitting = False
    default_to_square = False
    size = {"shortest_edge": 378, "longest_edge": 980}
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def __init__(self, **kwargs: Unpack[Idefics2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Idefics2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """Convert an image to RGB format."""
        return convert_to_rgb(image)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize using Idefics2 shortest_edge/longest_edge logic."""
        if size.shortest_edge and size.longest_edge:
            new_size = get_resize_output_image_size(image, size)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError("Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys.")

        return super().resize(image, SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        """Prepare a nested images structure for processing."""
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def split_images(self, images: "torch.Tensor") -> list[list["torch.Tensor"]]:
        """
        Split a batch of images into 4 equal sub-images, and concatenate that sequence with the original image.
        """
        height, width = images.shape[-2:]

        mid_width = width // 2
        mid_height = height // 2

        batch_split_images = [
            images[..., :mid_height, :mid_width],
            images[..., :mid_height, mid_width:],
            images[..., mid_height:, :mid_width],
            images[..., mid_height:, mid_width:],
            images,
        ]

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

        if original_size != padded_size:
            padding = (0, 0, padding_right, padding_bottom)
            image = tvF.pad(image, padding, fill=fill, padding_mode="constant")

        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        split_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_image_splitting:
                stacked_images = self.split_images(stacked_images)
            split_images_grouped[shape] = stacked_images
        split_images = reorder_images(split_images_grouped, grouped_images_index, is_nested=True)
        if do_image_splitting:
            for i, group_images in enumerate(split_images):
                split_images[i] = [image for sublist in group_images for image in sublist]

        grouped_images, grouped_images_index = group_images_by_shape(
            split_images, disable_grouping=disable_grouping, is_nested=True
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index, is_nested=True)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping, is_nested=True
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)

        if do_pad:
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


__all__ = ["Idefics2ImageProcessor"]
