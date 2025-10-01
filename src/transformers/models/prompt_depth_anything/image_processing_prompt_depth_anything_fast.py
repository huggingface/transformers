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
"""Fast Image processor class for PromptDepthAnything."""

import math
from typing import TYPE_CHECKING, Optional, Union

from ...image_processing_utils import BatchFeature
from ...processing_utils import Unpack


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput
import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import (
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
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    auto_docstring,
    requires_backends,
)


def _constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
    """Constrain a value to be a multiple of another value."""
    x = round(val / multiple) * multiple

    if max_val is not None and x > max_val:
        x = math.floor(val / multiple) * multiple

    if x < min_val:
        x = math.ceil(val / multiple) * multiple

    return x


def _get_resize_output_image_size(
    input_image: "torch.Tensor",
    output_size: tuple[int, int],
    keep_aspect_ratio: bool,
    multiple: int,
) -> tuple[int, int]:
    """Get the output size for resizing an image."""
    input_height, input_width = input_image.shape[-2:]
    output_height, output_width = output_size

    # determine new height and width
    scale_height = output_height / input_height
    scale_width = output_width / input_width

    if keep_aspect_ratio:
        # scale as little as possible
        if abs(1 - scale_width) < abs(1 - scale_height):
            # fit width
            scale_height = scale_width
        else:
            # fit height
            scale_width = scale_height

    new_height = _constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = _constrain_to_multiple_of(scale_width * input_width, multiple=multiple)

    return (new_height, new_width)


class PromptDepthAnythingFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    keep_aspect_ratio (`bool`, *optional*):
        If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
    ensure_multiple_of (`int`, *optional*):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value.
    do_pad (`bool`, *optional*):
        Whether to apply center padding.
    size_divisor (`int`, *optional*):
        If `do_pad` is `True`, pads the image dimensions to be divisible by this value.
    prompt_scale_to_meter (`float`, *optional*):
        Scale factor to convert the prompt depth to meters.
    """

    keep_aspect_ratio: Optional[bool]
    ensure_multiple_of: Optional[int]
    do_pad: Optional[bool]
    size_divisor: Optional[int]
    prompt_scale_to_meter: Optional[float]


@auto_docstring
class PromptDepthAnythingImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values", "prompt_depth"]

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    keep_aspect_ratio = False
    ensure_multiple_of = 1
    do_pad = False
    size_divisor = None
    prompt_scale_to_meter = 0.001
    valid_kwargs = PromptDepthAnythingFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        prompt_depth: Optional[ImageInput] = None,
        **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        prompt_depth (`ImageInput`, *optional*):
            Prompt depth to preprocess.
        """
        return super().preprocess(images, prompt_depth, **kwargs)

    def resize_with_aspect_ratio(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        interpolation: Optional["F.InterpolationMode"] = None,
    ) -> "torch.Tensor":
        """
        Resize an image to target size while optionally maintaining aspect ratio and ensuring dimensions are multiples.
        """
        # Set default interpolation to BICUBIC to match the slow processor (causes slight numerical differences otherwise)
        if interpolation is None:
            interpolation = F.InterpolationMode.BICUBIC

        # Custom resize with aspect ratio preservation and ensure_multiple_of constraint
        output_size = _get_resize_output_image_size(
            image,
            output_size=(size["height"], size["width"]),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )

        # Standard resize method with calculated output size
        return self.resize(
            image=image,
            size=SizeDict(height=output_size[0], width=output_size[1]),
            interpolation=interpolation,
        )

    def pad_image(
        self,
        image: "torch.Tensor",
        size_divisor: int,
    ) -> "torch.Tensor":
        """
        Center pad an image to be a multiple of size_divisor.
        """

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        height, width = image.shape[-2:]

        # Match slow processor and PyTorch convention: width->left/right, height->top/bottom
        pad_size_left, pad_size_right = _get_pad(width, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(height, size_divisor)

        # Use torchvision padding for fast processing
        # /!\ NB: torchvision F.pad expects (left, top, right, bottom) for the last two dims (W then H)
        # Source: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Pad.html
        # So: (left=width_pad, top=height_pad, right=width_pad, bottom=height_pad)
        padding = [pad_size_left, pad_size_top, pad_size_right, pad_size_bottom]
        padded_image = F.pad(image, padding=padding)

        return padded_image

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        prompt_depth: Optional[ImageInput],
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        prompt_scale_to_meter: Optional[float] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs, including the main images and optional prompt depth.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=False, input_data_format=input_data_format, device=device
        )  # always use do_convert_rgb=False rather than defining it as a param to match slow processor

        # Process images with the standard pipeline
        pixel_values = self._preprocess(images, return_tensors=return_tensors, **kwargs)

        data = {"pixel_values": pixel_values}

        # Process prompt depth if provided
        if prompt_depth is not None:
            processed_prompt_depths = self._prepare_image_like_inputs(
                images=prompt_depth,
                do_convert_rgb=False,  # Depth maps should not be converted
                input_data_format=input_data_format,
                device=images[0].device if images else device,
                expected_ndims=2,
            )

            # Validate prompt_depths has same length as images as in slow processor
            if len(processed_prompt_depths) != len(images):
                raise ValueError(
                    f"Number of prompt depth images ({len(processed_prompt_depths)}) does not match number of input images ({len(images)})"
                )

            final_prompt_depths = []
            for depth in processed_prompt_depths:
                depth = depth * prompt_scale_to_meter

                # Handle case where depth is constant (min == max)
                if depth.min() == depth.max():
                    depth[0, 0] = depth[0, 0] + 1e-6  # Add small variation to avoid numerical issues

                if depth.ndim == 2:  # Add channel dimension if needed
                    depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W] (channels first)

                depth = depth.float()  # Convert to float32 to match slow processor
                final_prompt_depths.append(depth)

            if return_tensors:
                # Stack while preserving the [H, W, C] format that the slow processor uses
                final_prompt_depths = torch.stack(final_prompt_depths, dim=0)

            data["prompt_depth"] = final_prompt_depths

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        keep_aspect_ratio: Optional[bool],
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: Optional[bool],
        disable_grouping: Optional[bool],
        ensure_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        size_divisor: Optional[int] = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Override the base _preprocess method to handle custom PromptDepthAnything parameters.
        """
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize_with_aspect_ratio(
                    image=stacked_images,
                    size=size,
                    keep_aspect_ratio=keep_aspect_ratio,
                    ensure_multiple_of=ensure_multiple_of,
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            if do_pad and size_divisor is not None:
                stacked_images = self.pad_image(stacked_images, size_divisor)

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Only stack tensors if they all have the same shape and return_tensors is specified
        if return_tensors == "pt":
            processed_images = torch.stack(processed_images, dim=0)

        return processed_images

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: Optional[Union[TensorType, list[tuple[int, int]], None]] = None,
    ) -> list[dict[str, TensorType]]:
        """
        Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`DepthEstimatorOutput`]):
                Raw outputs of the model.
            target_sizes (`TensorType` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `list[dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()

            results.append({"predicted_depth": depth})

        return results


__all__ = ["PromptDepthAnythingImageProcessorFast"]
