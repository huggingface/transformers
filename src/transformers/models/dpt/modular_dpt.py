# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

from transformers.image_processing_base import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.models.beit.image_processing_beit_fast import BeitImageProcessorFast

from ...image_processing_utils_fast import (
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    requires_backends,
)


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

if is_torch_available():
    import torch

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.transforms import functional as F


def get_resize_output_image_size(
    input_image: "torch.Tensor",
    output_size: Union[int, Iterable[int]],
    keep_aspect_ratio: bool,
    multiple: int,
) -> SizeDict:
    def constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple

        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

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

    new_height = constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constrain_to_multiple_of(scale_width * input_width, multiple=multiple)

    return SizeDict(height=new_height, width=new_width)


class DPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    ensure_multiple_of (`int`, *optional*, defaults to 1):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overidden
        by `ensure_multiple_of` in `preprocess`.
    do_pad (`bool`, *optional*, defaults to `False`):
        Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in
        combination with DPT.
    size_divisor (`int`, *optional*):
        If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the
        DINOv2 paper, which uses the model in combination with DPT.
    keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
        If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
        be overidden by `keep_aspect_ratio` in `preprocess`.
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    ensure_multiple_of: Optional[int]
    size_divisor: Optional[int]
    do_pad: Optional[bool]
    keep_aspect_ratio: Optional[bool]
    do_reduce_labels: Optional[bool]


@auto_docstring
class DPTImageProcessorFast(BeitImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = False
    rescale_factor = 1 / 255
    ensure_multiple_of = 1
    keep_aspect_ratio = False
    do_reduce_labels = False
    crop_size = None
    do_center_crop = None
    do_reduce_labels = None

    valid_kwargs = DPTFastImageProcessorKwargs

    def from_dict():
        raise NotImplementedError("No need to override this method")

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        ensure_multiple_of: Optional[int] = 1,
        keep_aspect_ratio: bool = False,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing when resizing the image
            ensure_multiple_of (`int`, *optional*):
                If `do_resize` is `True`, the image is resized to a size that is a multiple of this value
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, and `do_resize` is `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.

        Returns:
            `torch.Tensor`: The resized image.
        """
        if not size.height or not size.width:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size.keys()}")

        output_size = get_resize_output_image_size(
            image,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )
        return BeitImageProcessorFast().resize(image, output_size, interpolation=interpolation, antialias=antialias)

    def pad_image(
        self,
        image: "torch.Tensor",
        size_divisor: int = 1,
    ) -> "torch.Tensor":
        r"""
        Center pad a batch of images to be a multiple of `size_divisor`.

        Args:
            image (`torch.Tensor`):
                Image to pad.  Can be a batch of images of dimensions (N, C, H, W) or a single image of dimensions (C, H, W).
            size_divisor (`int`):
                The width and height of the image will be padded to a multiple of this number.
        """
        height, width = image.shape[-2:]

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        pad_top, pad_bottom = _get_pad(height, size_divisor)
        pad_left, pad_right = _get_pad(width, size_divisor)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(image, padding)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
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
        keep_aspect_ratio: bool,
        ensure_multiple_of: Optional[int],
        do_pad: bool,
        size_divisor: Optional[int],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        if do_reduce_labels:
            images = self.reduce_label(images)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images,
                    size=size,
                    interpolation=interpolation,
                    ensure_multiple_of=ensure_multiple_of,
                    keep_aspect_ratio=keep_aspect_ratio,
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
            if do_pad:
                stacked_images = self.pad_image(stacked_images, size_divisor)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images})

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
            target_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
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


__all__ = ["DPTImageProcessorFast"]
