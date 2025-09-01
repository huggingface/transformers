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
"""Image processor class for PromptDepthAnything."""

import math
from typing import Optional, Union

from ...image_processing_utils import BatchFeature
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
    get_image_size,
)
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


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
    output_size: Union[int, tuple[int, int]],
    keep_aspect_ratio: bool,
    multiple: int,
) -> tuple[int, int]:
    """Get the output size for resizing an image."""
    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    input_height, input_width = get_image_size(input_image, ChannelDimension.FIRST)
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
    Extended kwargs for PromptDepthAnything fast image processor.

    Args:
        keep_aspect_ratio (`bool`, *optional*):
            If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
        ensure_multiple_of (`int`, *optional*):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value.
        do_pad (`bool`, *optional*):
            Whether to apply center padding.
        size_divisor (`int`, *optional*):
            If `do_pad` is `True`, pads the image dimensions to be divisible by this value.
        prompt_depth (`ImageInput`, *optional*):
            Prompt depth to preprocess.
        prompt_scale_to_meter (`float`, *optional*):
            Scale factor to convert the prompt depth to meters.
    """

    keep_aspect_ratio: Optional[bool]
    ensure_multiple_of: Optional[int]
    do_pad: Optional[bool]
    size_divisor: Optional[int]
    prompt_depth: Optional[ImageInput]
    prompt_scale_to_meter: Optional[float]


@auto_docstring
class PromptDepthAnythingImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast PromptDepthAnything image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]` *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the image after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Defines the resampling filter to use if resizing the image.
        keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
            If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
        ensure_multiple_of (`int`, *optional*, defaults to 1):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*, defaults to `False`):
            Whether to apply center padding.
        size_divisor (`int`, *optional*):
            If `do_pad` is `True`, pads the image dimensions to be divisible by this value.
        prompt_scale_to_meter (`float`, *optional*, defaults to 0.001):
            Scale factor to convert the prompt depth to meters.
    """

    model_input_names = ["pixel_values", "prompt_depth"]

    # Default values checked against the slow image processor
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    keep_aspect_ratio = False
    ensure_multiple_of = 1
    do_pad = False
    size_divisor = None
    prompt_scale_to_meter = 0.001
    valid_kwargs = PromptDepthAnythingFastImageProcessorKwargs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        prompt_depth: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            prompt_depth (`ImageInput`, *optional*):
                Prompt depth to preprocess, which can be sparse depth obtained from multi-view geometry or
                low-resolution depth from a depth sensor. If it is None, the output depth will be a monocular relative depth.
        """
        # Handle prompt_depth processing
        if prompt_depth is not None:
            kwargs["prompt_depth"] = prompt_depth

        return super().preprocess(images, **kwargs)

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
        if not keep_aspect_ratio:
            # Standard resize
            return self.resize(image=image, size=size, interpolation=interpolation)

        # Custom resize with aspect ratio preservation
        output_size = _get_resize_output_image_size(
            image,
            output_size=(size["height"], size["width"]),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )

        # Use the resize method with calculated output size
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

        height, width = get_image_size(image, ChannelDimension.FIRST)

        pad_size_left, pad_size_right = _get_pad(height, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)

        # Use torchvision pad function: [left, right, top, bottom]
        padding = [pad_size_top, pad_size_bottom, pad_size_left, pad_size_right]
        padded_image = F.pad(image, padding=padding)

        return padded_image

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        *args,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        prompt_depth: Optional[ImageInput] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess image-like inputs including prompt depth.
        """
        # Prepare input images
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Process prompt depth if provided
        processed_prompt_depths = None
        if prompt_depth is not None:
            prompt_depths = self._prepare_image_like_inputs(
                images=prompt_depth,
                do_convert_rgb=False,  # Depth maps should not be converted to RGB
                input_data_format=input_data_format,
                device=device,
                expected_ndims=2,  # Depth maps are typically 2D
            )

            # Validate prompt_depths has same length as images
            if len(prompt_depths) != len(images):
                raise ValueError(
                    f"Number of prompt depth images ({len(prompt_depths)}) does not match number of input images ({len(images)})"
                )

            prompt_scale_to_meter = kwargs.get("prompt_scale_to_meter", self.prompt_scale_to_meter)
            processed_prompt_depths = []

            for depth in prompt_depths:
                # Scale to meters
                depth = depth * prompt_scale_to_meter

                # Handle case where depth is constant (min == max)
                if depth.min() == depth.max():
                    # Add small variation to avoid numerical issues
                    depth[0, 0] = depth[0, 0] + 1e-6

                # Add channel dimension if needed (depth maps are typically 2D)
                if depth.ndim == 2:
                    depth = depth.unsqueeze(0)

                processed_prompt_depths.append(depth)

        result = self._preprocess(images, *args, **kwargs)

        # Add prompt depth to the result if it was processed
        if processed_prompt_depths is not None:
            # Stack processed depths if return_tensors is set
            if kwargs.get("return_tensors"):
                processed_prompt_depths = torch.stack(processed_prompt_depths, dim=0)
            result.data["prompt_depth"] = processed_prompt_depths

        return result

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        keep_aspect_ratio: Optional[bool] = None,
        ensure_multiple_of: Optional[int] = None,
        do_pad: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Override the base _preprocess method to handle custom PromptDepthAnything parameters.
        """
        # Set defaults for custom parameters
        keep_aspect_ratio = keep_aspect_ratio if keep_aspect_ratio is not None else self.keep_aspect_ratio
        ensure_multiple_of = ensure_multiple_of if ensure_multiple_of is not None else self.ensure_multiple_of
        do_pad = do_pad if do_pad is not None else self.do_pad
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor

        # Default values for required parameters
        if disable_grouping is None:
            disable_grouping = False

        # Ensure image_mean and image_std are not None
        if image_mean is None:
            image_mean = self.image_mean
        if image_std is None:
            image_std = self.image_std

        # Group images by size for batched processing
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

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                # Convert SizeDict to regular dict for center_crop
                crop_dict = {"height": crop_size["height"], "width": crop_size["width"]}
                stacked_images = self.center_crop(stacked_images, crop_dict)

            # Apply padding if requested
            if do_pad and size_divisor is not None:
                stacked_images = self.pad_image(stacked_images, size_divisor)

            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        # Remove fast-specific runtime attributes that shouldn't be saved
        encoder_dict.pop("crop_size", None)
        encoder_dict.pop("device", None)
        encoder_dict.pop("disable_grouping", None)
        encoder_dict.pop("do_center_crop", None)
        encoder_dict.pop("do_convert_rgb", None)
        encoder_dict.pop("input_data_format", None)
        encoder_dict.pop("prompt_depth", None)
        encoder_dict.pop("return_tensors", None)
        return encoder_dict


__all__ = ["PromptDepthAnythingImageProcessorFast"]


__all__ = ["PromptDepthAnythingImageProcessorFast"]
