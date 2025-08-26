# coding=utf-8
# Copyright 2025 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for BridgeTower."""

from collections.abc import Iterable
from typing import Optional, Union

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    ImageInput,
    SizeDict,
    TensorType,
    Unpack,
    get_max_height_width,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import auto_docstring, is_torch_available, is_torchvision_available, is_torchvision_v2_available


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def make_pixel_mask(
    image: "torch.Tensor",
    output_size: tuple[int, int],
) -> "torch.Tensor":
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = image.shape[-2:]
    batch_size = image.size(0)
    mask = torch.zeros((batch_size, *output_size), dtype=torch.long)
    mask[:input_height, :input_width] = 1
    return mask


def get_resize_output_image_size(
    input_image: "torch.Tensor",
    shorter: int = 800,
    longer: int = 1333,
    size_divisor: int = 32,
) -> tuple[int, int]:
    input_height, input_width = input_image.shape[-2:]
    min_size, max_size = shorter, longer

    scale = min_size / min(input_height, input_width)

    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    return new_height, new_width


class BridgeTowerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        size_divisor (`int`, *optional*, defaults to 32):
            The size by which to make sure both the height and width can be divided. Only has an effect if `do_resize`
            is set to `True`. Can be overridden by the `size_divisor` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch. Can be overridden by
            the `do_pad` parameter in the `preprocess` method.
    """

    size_divisor: Optional[int]
    do_pad: Optional[bool]


@auto_docstring
class BridgeTowerImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 288}
    default_to_square = False
    crop_size = {"shortest_edge": 288}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    size_divisor = 32
    valid_kwargs = BridgeTowerFastImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(self, **kwargs: Unpack[BridgeTowerFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[BridgeTowerFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        size_divisor: int = 32,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image.

        Resizes the shorter side of the image to `size["shortest_edge"]` while preserving the aspect ratio. If the
        longer side is larger than the max size `(int(`size["shortest_edge"]` * 1333 / 800))`, the longer side is then
        resized to the max size while preserving the aspect ratio.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            size_divisor (`int`, *optional*, defaults to 32):
                The image is resized to a size that is a multiple of this value.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if not size.shortest_edge:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        shorter = size.shortest_edge
        longer = int(1333 / 800 * shorter)
        output_height, output_width = get_resize_output_image_size(
            image,
            shorter=shorter,
            longer=longer,
            size_divisor=size_divisor,
        )
        return super().resize(
            image=image,
            size=SizeDict(height=output_height, width=output_width),
            interpolation=interpolation,
            antialias=antialias,
        )

    def center_crop(
        self,
        image: "torch.Tensor",
        size: dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`torch.Tensor`):
                Image to center crop.
            size (`dict[str, int]`):
                Size of the output image in the form `{"height": h, "width": w}`.
        """
        output_size = size.shortest_edge
        return F.center_crop(
            image,
            output_size=(output_size, output_size),
            **kwargs,
        )

    def _pad_image(
        self,
        image: "torch.Tensor",
        output_size: tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
    ) -> "torch.Tensor":
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = image.shape[-2:]
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = (0, 0, pad_right, pad_bottom)
        padded_image = F.pad(
            image,
            padding,
            fill=constant_values,
        )
        return padded_image

    def pad(
        self,
        images: list["torch.Tensor"],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        disable_grouping: Optional[bool] = False,
    ) -> tuple:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            image (`torch.Tensor`):
                Image to pad.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            disable_grouping (`bool`, *optional*, defaults to `False`):
                Whether to disable grouping of images by size.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        pad_size = get_max_height_width(images)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_masks_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self._pad_image(
                stacked_images,
                pad_size,
                constant_values=constant_values,
            )
            processed_images_grouped[shape] = stacked_images

            if return_pixel_mask:
                stacked_masks = make_pixel_mask(image=stacked_images, output_size=pad_size)
                processed_masks_grouped[shape] = stacked_masks

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        processed_masks = None
        if return_pixel_mask:
            processed_masks = reorder_images(processed_masks_grouped, grouped_images_index)

        return processed_images, processed_masks

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        size_divisor: Optional[int],
        interpolation: Optional["F.InterpolationMode"],
        do_pad: bool,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, size_divisor=size_divisor, interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
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

        data = {}
        if do_pad:
            processed_images, processed_masks = self.pad(
                processed_images, return_pixel_mask=True, disable_grouping=disable_grouping
            )
            processed_masks = torch.stack(processed_masks, dim=0) if return_tensors else processed_masks
            data["pixel_mask"] = processed_masks

        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        data["pixel_values"] = processed_images

        return BatchFeature(data=data, tensor_type=return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        encoder_dict.pop("crop_size", None)
        return encoder_dict


__all__ = ["BridgeTowerImageProcessorFast"]
