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
    requires_backends,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

# Copied from transformers.models.prompt_depth_anything.image_processing_prompt_depth_anything (i.e. the slow processor)
def _constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
    """Constrain a value to be a multiple of another value."""
    x = round(val / multiple) * multiple

    if max_val is not None and x > max_val:
        x = math.floor(val / multiple) * multiple

    if x < min_val:
        x = math.ceil(val / multiple) * multiple

    return x

# Copied from transformers.models.prompt_depth_anything.image_processing_prompt_depth_anything (i.e. the slow processor)
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

    # Default values - match the classical processor
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
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

    def __init__(self, **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[PromptDepthAnythingFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        This wrapper exposes custom kwargs in the signature for documentation and typing.
        Delegates to BaseImageProcessorFast.preprocess.
        """
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
        # Set default interpolation to BICUBIC to match the slow processor (causes slight numerical differences otherwise)
        if interpolation is None:
            interpolation = F.InterpolationMode.BICUBIC

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

        height, width = get_image_size(image, ChannelDimension.FIRST)

        pad_w_left, pad_w_right = _get_pad(width, size_divisor)
        pad_h_top, pad_h_bottom = _get_pad(height, size_divisor)

        # Use torchvision padding for fast processing
        # /!\ NB: torchvision F.pad expects (left, top, right, bottom) for the last two dims (W then H)
        # Source: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Pad.html
        padding = [pad_w_left, pad_h_top, pad_w_right, pad_h_bottom]
        padded_image = F.pad(image, padding=padding)

        return padded_image

    def _further_process_kwargs(
        self,
        prompt_depth: Optional[ImageInput] = None,
        prompt_scale_to_meter: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """
        Override the base method to handle custom PromptDepthAnything parameters.
        No significant processing is needed here, but we just need to store parameters for later use in _preprocess.
        """
        kwargs = super()._further_process_kwargs(**kwargs)

        # Handle prompt_depth by storing it for later use in _preprocess
        if prompt_depth is not None:
            kwargs["prompt_depth"] = prompt_depth
        if prompt_scale_to_meter is not None:
            kwargs["prompt_scale_to_meter"] = prompt_scale_to_meter

        return kwargs

    def _validate_preprocess_kwargs(
        self,
        keep_aspect_ratio: Optional[bool] = None,
        ensure_multiple_of: Optional[int] = None,
        do_pad: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        prompt_depth: Optional[ImageInput] = None,
        prompt_scale_to_meter: Optional[float] = None,
        **kwargs,
    ):
        """
        Overrides the base method to add custom validation.
        Validate the kwargs for the preprocess method, including PromptDepthAnything-specific parameters.
        """
        # Call parent validation
        super()._validate_preprocess_kwargs(**kwargs)

        # Custom validation for PromptDepthAnything parameters
        if do_pad and size_divisor is None:
            raise ValueError("size_divisor must be provided when do_pad is True")

        if ensure_multiple_of is not None and ensure_multiple_of < 1:
            raise ValueError("ensure_multiple_of must be >= 1")

        if prompt_scale_to_meter is not None and prompt_scale_to_meter <= 0:
            raise ValueError("prompt_scale_to_meter must be > 0")

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
        prompt_depth: Optional[ImageInput] = None,
        prompt_scale_to_meter: Optional[float] = None,
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
        prompt_scale_to_meter = (
            prompt_scale_to_meter if prompt_scale_to_meter is not None else self.prompt_scale_to_meter
        )

        # Set default
        if disable_grouping is None:
            disable_grouping = False
        if image_mean is None:
            image_mean = self.image_mean
        if image_std is None:
            image_std = self.image_std

        # Process prompt depth
        processed_prompt_depths = None
        if prompt_depth is not None:
            # Convert prompt depth to tensor format similar to images
            prompt_depths = self._prepare_image_like_inputs(
                images=prompt_depth,
                do_convert_rgb=False,  # Depth maps should not be converted to RGB
                input_data_format=None,
                device=images[0].device if images else None,
                expected_ndims=2,
            )

            processed_prompt_depths = []
            for depth in prompt_depths:
                # Scale
                depth = depth * prompt_scale_to_meter

                # When depth is constant, we need to add a small variation to avoid numerical issues
                if depth.min() == depth.max():
                    depth[0, 0] = depth[0, 0] + 1e-6

                # Add channel dimension if needed
                if depth.ndim == 2:
                    depth = depth.unsqueeze(0)

                processed_prompt_depths.append(depth)

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
                crop_dict = {"height": crop_size["height"], "width": crop_size["width"]}
                stacked_images = self.center_crop(stacked_images, crop_dict)

            if do_pad and size_divisor is not None:
                stacked_images = self.pad_image(stacked_images, size_divisor) # Apply padding if requested

            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        # Prepare result data
        result_data = {"pixel_values": processed_images}

        # Add prompt depth to the result if it was processed
        if processed_prompt_depths is not None:
            # Stack processed depths if return_tensors is set
            if return_tensors:
                processed_prompt_depths = torch.stack(processed_prompt_depths, dim=0)
            result_data["prompt_depth"] = processed_prompt_depths

        return BatchFeature(data=result_data, tensor_type=return_tensors)

    # Copied from transformers.models.dpt.image_processing_dpt.DPTImageProcessor.post_process_depth_estimation with DPT->PromptDepthAnything
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

    def to_dict(self):
        """
        Return a plain config dict suitable for saving.

        We drop fast-only runtime knobs (e.g. device, return_tensors, grouping/conversion flags,
        and transient inputs like prompt_depth) so the saved config mirrors the slow processor,
        stays portable across environments and passes all tests. Necessary to pass tests.
        """
        encoder_dict = super().to_dict()
        for k in (
            "_valid_processor_keys",
            "crop_size",
            "device",
            "disable_grouping",
            "do_center_crop",
            "do_convert_rgb",
            "input_data_format",
            "prompt_depth",
            "return_tensors",
        ):
            encoder_dict.pop(k, None)
        return encoder_dict


__all__ = ["PromptDepthAnythingImageProcessorFast"]
