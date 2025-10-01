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
"""Fast Image processor class for ZoeDepth."""

from typing import (
    Optional,
    Union,
)

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import (
    BatchFeature,
)
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
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
    requires_backends,
)
from .image_processing_zoedepth import get_resize_output_image_size
from .modeling_zoedepth import ZoeDepthDepthEstimatorOutput


logger = logging.get_logger(__name__)


class ZoeDepthFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    keep_aspect_ratio (`bool`, *optional*, defaults to `True`):
        If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
        for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
        within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
        size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
        Can be overridden by `keep_aspect_ratio` in `preprocess`.
    ensure_multiple_of (`int`, *optional*, defaults to 32):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
        the height and width to the nearest multiple of this value.
        Works both with and without `keep_aspect_ratio` being set to `True`.
        Can be overridden by `ensure_multiple_of` in `preprocess`.
    """

    keep_aspect_ratio: Optional[bool]
    ensure_multiple_of: Optional[int]


@auto_docstring
class ZoeDepthImageProcessorFast(BaseImageProcessorFast):
    do_pad = True
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    size = {"height": 384, "width": 512}
    resample = PILImageResampling.BILINEAR
    keep_aspect_ratio = True
    ensure_multiple_of = 1 / 32
    valid_kwargs = ZoeDepthFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ZoeDepthFastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: "torch.Tensor",
        size: SizeDict,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        interpolation: Optional["F.InterpolationMode"] = None,
    ) -> "torch.Tensor":
        """
        Resize an image or batchd images to target size `(size["height"], size["width"])`. If `keep_aspect_ratio` is `True`, the image
        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is
        set, the image is resized to a size that is a multiple of this value.

        Args:
            images (`torch.Tensor`):
                Images to resize.
            size (`dict[str, int]`):
                Target size of the output image.
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
            ensure_multiple_of (`int`, *optional*, defaults to 1):
                The image is resized to a size that is a multiple of this value.
            interpolation (`F.InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size
                specified in `size`.
        """
        if not size.height or not size.width:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size}")
        output_size = get_resize_output_image_size(
            images,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
            input_data_format=ChannelDimension.FIRST,
        )
        height, width = output_size

        resized_images = torch.nn.functional.interpolate(
            images, (int(height), int(width)), mode=interpolation.value, align_corners=True
        )

        return resized_images

    def _pad_images(
        self,
        images: "torch.Tensor",
    ):
        """
        Args:
            image (`torch.Tensor`):
                Image to pad.
        """
        height, width = get_image_size(images, channel_dim=ChannelDimension.FIRST)

        pad_height = int(np.sqrt(height / 2) * 3)
        pad_width = int(np.sqrt(width / 2) * 3)

        return F.pad(images, padding=(pad_width, pad_height), padding_mode="reflect")

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        keep_aspect_ratio: Optional[bool],
        ensure_multiple_of: Optional[int],
        interpolation: Optional["F.InterpolationMode"],
        do_pad: bool,
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            if do_pad:
                stacked_images = self._pad_images(images=stacked_images)
            if do_resize:
                stacked_images = self.resize(
                    stacked_images, size, keep_aspect_ratio, ensure_multiple_of, interpolation
                )
            if do_normalize:
                stacked_images = self.normalize(stacked_images, image_mean, image_std)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        processed_images = torch.stack(resized_images, dim=0) if return_tensors else resized_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        outputs: "ZoeDepthDepthEstimatorOutput",
        source_sizes: Optional[Union[TensorType, list[tuple[int, int]], None]] = None,
        target_sizes: Optional[Union[TensorType, list[tuple[int, int]], None]] = None,
        outputs_flipped: Optional[Union["ZoeDepthDepthEstimatorOutput", None]] = None,
        do_remove_padding: Optional[Union[bool, None]] = None,
    ) -> list[dict[str, TensorType]]:
        """
        Converts the raw output of [`ZoeDepthDepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`ZoeDepthDepthEstimatorOutput`]):
                Raw outputs of the model.
            source_sizes (`TensorType` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the source size
                (height, width) of each image in the batch before preprocessing. This argument should be dealt as
                "required" unless the user passes `do_remove_padding=False` as input to this function.
            target_sizes (`TensorType` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            outputs_flipped ([`ZoeDepthDepthEstimatorOutput`], *optional*):
                Raw outputs of the model from flipped input (averaged out in the end).
            do_remove_padding (`bool`, *optional*):
                By default ZoeDepth adds padding equal to `int(√(height / 2) * 3)` (and similarly for width) to fix the
                boundary artifacts in the output depth map, so we need remove this padding during post_processing. The
                parameter exists here in case the user changed the image preprocessing to not include padding.

        Returns:
            `list[dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth

        if (outputs_flipped is not None) and (predicted_depth.shape != outputs_flipped.predicted_depth.shape):
            raise ValueError("Make sure that `outputs` and `outputs_flipped` have the same shape")

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        if do_remove_padding is None:
            do_remove_padding = self.do_pad

        if source_sizes is None and do_remove_padding:
            raise ValueError(
                "Either `source_sizes` should be passed in, or `do_remove_padding` should be set to False"
            )

        if (source_sizes is not None) and (len(predicted_depth) != len(source_sizes)):
            raise ValueError(
                "Make sure that you pass in as many source image sizes as the batch dimension of the logits"
            )

        if outputs_flipped is not None:
            predicted_depth = (predicted_depth + torch.flip(outputs_flipped.predicted_depth, dims=[-1])) / 2

        predicted_depth = predicted_depth.unsqueeze(1)

        # Zoe Depth model adds padding around the images to fix the boundary artifacts in the output depth map
        # The padding length is `int(np.sqrt(img_h/2) * fh)` for the height and similar for the width
        # fh (and fw respectively) are equal to '3' by default
        # Check [here](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L57)
        # for the original implementation.
        # In this section, we remove this padding to get the final depth image and depth prediction
        padding_factor_h = padding_factor_w = 3

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        source_sizes = [None] * len(predicted_depth) if source_sizes is None else source_sizes
        for depth, target_size, source_size in zip(predicted_depth, target_sizes, source_sizes):
            # depth.shape = [1, H, W]
            if source_size is not None:
                pad_h = pad_w = 0

                if do_remove_padding:
                    pad_h = int(np.sqrt(source_size[0] / 2) * padding_factor_h)
                    pad_w = int(np.sqrt(source_size[1] / 2) * padding_factor_w)

                depth = F.resize(
                    depth,
                    size=[source_size[0] + 2 * pad_h, source_size[1] + 2 * pad_w],
                    interpolation=F.InterpolationMode.BICUBIC,
                    antialias=False,
                )

                if pad_h > 0:
                    depth = depth[:, pad_h:-pad_h, :]
                if pad_w > 0:
                    depth = depth[:, :, pad_w:-pad_w]

            if target_size is not None:
                target_size = [target_size[0], target_size[1]]
                depth = F.resize(
                    depth,
                    size=target_size,
                    interpolation=F.InterpolationMode.BICUBIC,
                    antialias=False,
                )
            depth = depth.squeeze(0)
            # depth.shape = [H, W]
            results.append({"predicted_depth": depth})

        return results


__all__ = ["ZoeDepthImageProcessorFast"]
