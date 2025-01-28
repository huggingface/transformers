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
"""Fast Image processor class for ConvNeXT."""

from typing import Dict, List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
)
from ...utils import (
    TensorType,
    add_start_docstrings,
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


@add_start_docstrings(
    r"Constructs a fast ConvNeXT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        crop_pct (`float`, *optional*):
            Percentage of the image to crop. Only has an effect if size < 384. Can be
            overridden by `crop_pct` in the`preprocess` method.
    """,
)
class ConvNextImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 384}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    crop_pct = 224 / 256
    valid_extra_kwargs = ["crop_pct"]

    def __init__(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        default_to_square: bool = None,
        resample: Union[PILImageResampling, "F.InterpolationMode"] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = None,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_rgb: bool = None,
        # Additional arguments
        crop_pct=None,
        **kwargs,
    ):
        super().__init__(
            do_resize=do_resize,
            size=size,
            default_to_square=default_to_square,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            crop_pct=crop_pct,
            **kwargs,
        )

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        crop_pct (`float`, *optional*):
            Percentage of the image to crop. Only has an effect if size < 384. Can be
            overridden by `crop_pct` in the`preprocess` method.
        """,
    )
    def preprocess(self, *args, **kwargs) -> BatchFeature:
        return super().preprocess(*args, **kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: Dict[str, int],
        crop_pct: float,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"shortest_edge": int}`, specifying the size of the output image. If
                `size["shortest_edge"]` >= 384 image is resized to `(size["shortest_edge"], size["shortest_edge"])`.
                Otherwise, the smaller edge of the image will be matched to `int(size["shortest_edge"] / crop_pct)`,
                after which the image is cropped to `(size["shortest_edge"], size["shortest_edge"])`.
            crop_pct (`float`):
                Percentage of the image to crop. Only has an effect if size < 384.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.

        Returns:
            `torch.Tensor`: Resized image.
        """
        if not size.shortest_edge:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        shortest_edge = size["shortest_edge"]

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            image = F.resize(
                image,
                resize_size,
                interpolation=interpolation,
                **kwargs,
            )
            # then crop to (shortest_edge, shortest_edge)
            return F.center_crop(
                image,
                (shortest_edge, shortest_edge),
                **kwargs,
            )
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return F.resize(
                image,
                (shortest_edge, shortest_edge),
                interpolation=interpolation,
                **kwargs,
            )

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: Dict[str, int],
        crop_pct: float,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: int,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, crop_pct=crop_pct, interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
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
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["ConvNextImageProcessorFast"]
