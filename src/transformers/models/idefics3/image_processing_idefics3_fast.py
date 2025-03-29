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

import math
from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
)
from ...image_transforms import (
    ChannelDimension,
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size_for_max_height_width,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)
from .image_processing_idefics3 import get_resize_output_image_size as get_resize_output_max_side_image_size


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def get_max_height_width(images: list["torch.Tensor"]) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    max_height = max_width = float("-inf")
    for image in images:
        height, width = image.size()[-2:]
        max_height = max(height, max_height)
        max_width = max(width, max_width)
    return (max_height, max_width)


class Idefics3FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]
    do_image_splitting: Optional[bool]
    max_image_size: Optional[Dict[str, int]]


@add_start_docstrings(
    "Constructs a fast Idefics3 image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pad (`bool`, *optional*):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        do_image_splitting (`bool`, *optional*, defaults to `True`):
                Whether to split the image into sub-images concatenated with the original image. They are split into patches
                such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
        max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 364}`):
                Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
    """,
)
class Idefics3ImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"longest_edge": 4 * 364}
    max_image_size = {"longest_edge": 364}
    default_to_square = None
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_image_splitting = True
    do_pad = True
    valid_kwargs = Idefics3FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Idefics3FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
            do_image_splitting (`bool`, *optional*, defaults to `True`):
                    Whether to split the image into sub-images concatenated with the original image. They are split into patches
                    such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
            max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 364}`):
                    Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Idefics3FastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.longest_edge:
            new_size = get_resize_output_max_side_image_size(
                image, resolution_max_side=size.longest_edge, input_data_format=ChannelDimension.FIRST
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(image, new_size, interpolation=interpolation, antialias=antialias)

    def split_image(
        self,
        image: torch.Tensor,
        max_image_size: Dict[str, int],
        interpolation: "F.InterpolationMode" = None,
    ):
        """
        Split an image into squares of side max_image_size and the original image resized to max_image_size.
        That means that a single image becomes a sequence of images.
        This is a "trick" to spend more compute on each image with no changes in the vision encoder.
        1) If one side of the original image is larger than `max_image_size`, resize it to `max_image_size` while preserving the aspect ratio.
        2) Divide the resulting image into `ceil(height / max_image_size)` x `ceil(width / max_image_size)`
        sub-images of the same size each (image_size, image_size). Typically, 364x364.
        3) Returns the list of the crops and the original image, in addition to the number of splits for the height and the width.
        Args:
            image (`torch.Tensor`):
                Images to split.
            max_image_size (`Dict[str, int]`):
                Maximum size of the output image. If the image is larger than this size, it will be split into
                patches of this size, and the original image will be concatenated with the patches, resized to max_size.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.
        """
        batch_size, num_channels, height, width = image.size()
        height_dim, width_dim = 2, 3

        max_height = max_width = max_image_size["longest_edge"]

        frames = []
        if height > max_height or width > max_width:
            # Calculate the number of splits
            num_splits_h = math.ceil(height / max_height)
            num_splits_w = math.ceil(width / max_width)

            # Split the image by height, then by width
            frames = (
                image.unfold(height_dim, size=max_height, step=max_height)
                .unfold(width_dim, size=max_width, step=max_width)
                .contiguous()
                .view(batch_size, num_channels, -1, max_height, max_width)
                .permute(0, 2, 1, 3, 4)
            )  # batch_size x n_frames x num_channels x height x width

            # For the global image at the end, we resize it to match the max_image_size, for cpu memory efficiency
            global_image_height, global_image_width = max_height, max_width
            image = self.resize(
                image, SizeDict(height=global_image_height, width=global_image_width), interpolation=interpolation
            )

            frames = torch.cat((frames, image.unsqueeze(1)), dim=1)
        else:
            num_splits_h, num_splits_w = 0, 0
            frames = image.unsqueeze(1)

        num_splits_h = [num_splits_h] * batch_size
        num_splits_w = [num_splits_w] * batch_size

        return frames, num_splits_h, num_splits_w

    def resize_for_vision_encoder(
        self,
        image: torch.Tensor,
        vision_encoder_max_size: int,
        interpolation: "F.InterpolationMode" = None,
    ):
        """
        Resize images to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.
        Args:
            image (`torch.Tensor`):
                Images to resize.
            vision_encoder_max_size (`int`):
                Maximum size of the output image. If the image is larger than this size, it will be split into
                patches of this size, and the original image will be concatenated with the patches, resized to max_size.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.
        """
        height, width = image.size()[-2:]

        aspect_ratio = width / height
        if width >= height:
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
        elif height > width:
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
        new_size = SizeDict(height=height, width=width)
        return self.resize(image, size=new_size, interpolation=interpolation)

    def pad(
        self,
        image: torch.Tensor,
        padded_size: Tuple[int, int],
        fill: int = 0,
        return_pixel_mask: bool = True,
    ):
        """Pads the sample with empty images to the padded_size
        Args:
            image (`torch.Tensor`):
                Image to pad.
            padded_size (`Tuple[int, int]`):
                Height and width to pad.
            fill (`int`, *optional*):
                The value to use for the padding.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
        """
        original_size = image.size()[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]
        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )
        if original_size != padded_size:
            padding = [0, 0, padding_right, padding_bottom]
            image = F.pad(image, padding, fill=fill)

        # Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
        pixel_mask = None
        if return_pixel_mask:
            pixel_mask = torch.zeros_like(image, dtype=torch.int64)
            pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        do_image_splitting: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        max_image_size: Optional[Dict[str, int]],
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        return_row_col_info: bool = False,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            if do_image_splitting:
                # We first resize both height and width of each image to the nearest max_image_size multiple, disregarding the aspect ratio
                # for size=(10, max_image_size) -> rescaled_size=(max_image_size, max_image_size)
                # for size=(11, max_image_size+1) -> rescaled_size=(max_image_size, max_image_size*2)
                stacked_images = self.resize_for_vision_encoder(
                    image=stacked_images,
                    vision_encoder_max_size=max_image_size["longest_edge"],
                    interpolation=interpolation,
                )
            else:
                # We square the images to max_image_size
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=max_image_size["longest_edge"], width=max_image_size["longest_edge"]),
                    interpolation=interpolation,
                )

            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        processed_rows_grouped = {}
        processed_cols_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_image_splitting:
                stacked_images, rows, cols = self.split_image(
                    stacked_images, max_image_size=max_image_size, interpolation=interpolation
                )
            else:
                stacked_images, rows, cols = (
                    stacked_images.unsqueeze(1),
                    [0] * stacked_images.size(0),
                    [0] * stacked_images.size(0),
                )

            processed_cols_grouped[shape] = cols
            processed_rows_grouped[shape] = rows

            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # For a list of images, for each images, pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width.
        # For each sample in the batch, pads the sample with empty images to the max_number of images per sample in the batch. Optionally returns a pixel mask.
        if do_pad:
            pad_size = get_max_height_width(processed_images)
            grouped_images, grouped_images_index = group_images_by_shape(processed_images)
            processed_padded_mask_grouped = {}
            processed_images_grouped = {}

            for shape, stacked_images in grouped_images.items():
                stacked_images, padded_masks = self.pad(stacked_images, padded_size=pad_size)

                processed_images_grouped[shape] = stacked_images
                processed_padded_mask_grouped[shape] = padded_masks

            processed_images = reorder_images(processed_images_grouped, grouped_images_index)
            pixel_attention_mask = reorder_images(processed_padded_mask_grouped, grouped_images_index)

        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        data = {"pixel_values": processed_images}

        if do_pad:
            data["pixel_attention_mask"] = (
                torch.stack(pixel_attention_mask, dim=0)
                if do_pad and return_tensors is not None
                else pixel_attention_mask
            )

        encoding = BatchFeature(data=data, tensor_type=return_tensors)

        if return_row_col_info:
            processed_rows = reorder_images(processed_rows_grouped, grouped_images_index)
            processed_cols = reorder_images(processed_cols_grouped, grouped_images_index)
            encoding["rows"] = processed_rows
            encoding["cols"] = processed_cols

        return encoding


__all__ = ["Idefics3ImageProcessorFast"]
