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
"""Image processor class for ZoeDepth."""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Union

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available, requires_backends
from ...utils.import_utils import requires


if TYPE_CHECKING:
    from .modeling_zoedepth import ZoeDepthDepthEstimatorOutput

if is_torch_available():
    import torch
    from torch import nn

if is_torchvision_available():
    import torchvision.transforms.v2.functional as tvF


# Adapted from transformers.models.zoedepth.image_processing_zoedepth.ZoeDepthImageProcessorKwargs
class ZoeDepthImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    keep_aspect_ratio (`bool`, *optional*, defaults to `self.keep_aspect_ratio`):
        If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
        for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
        within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
        size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
        Can be overridden by `keep_aspect_ratio` in `preprocess`.
    ensure_multiple_of (`int`, *optional*, defaults to `self.ensure_multiple_of`):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
        the height and width to the nearest multiple of this value.
        Works both with and without `keep_aspect_ratio` being set to `True`.
        Can be overridden by `ensure_multiple_of` in `preprocess`.
    """

    keep_aspect_ratio: bool
    ensure_multiple_of: int


# Adapted from transformers.models.zoedepth.image_processing_zoedepth.get_resize_output_image_size
def get_resize_output_image_size(
    input_image: "torch.Tensor | np.ndarray",
    output_size: int | Iterable[int],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: str | ChannelDimension | None = None,
) -> tuple[int, int]:
    def constrain_to_multiple_of(val, multiple, min_val=0):
        x = (np.round(val / multiple) * multiple).astype(int)

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    input_height, input_width = get_image_size(input_image, input_data_format)
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

    return (new_height, new_width)


@auto_docstring
@requires(backends=("torch",))
class ZoeDepthImageProcessorPil(PilBackend):
    valid_kwargs = ZoeDepthImageProcessorKwargs
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

    def __init__(self, **kwargs: Unpack[ZoeDepthImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[ZoeDepthImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
    ) -> np.ndarray:
        """
        Resize an image to target size `(size.height, size.width)`. If `keep_aspect_ratio` is `True`, the image
        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is
        set, the image is resized to a size that is a multiple of this value.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Target size of the output image.
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
            ensure_multiple_of (`int`, *optional*, defaults to 1):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size
                specified in `size`.
        """
        if not size.height or not size.width:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size}")
        height, width = get_resize_output_image_size(
            image,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
            input_data_format=ChannelDimension.FIRST,
        )

        torch_image = torch.from_numpy(image).unsqueeze(0)
        # TODO support align_corners=True in image_transforms.resize
        requires_backends(self, "torch")
        resample_to_mode = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}
        mode = resample_to_mode[resample]
        resized_image = nn.functional.interpolate(
            torch_image, (int(height), int(width)), mode=mode, align_corners=True
        )
        resized_image = resized_image.squeeze().numpy()

        return resized_image

    def pad_image(
        self,
        image: np.ndarray,
    ):
        """
        Args:
            image (`np.ndarray`):
                Image to pad.
        """
        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        pad_height = int(np.sqrt(height / 2) * 3)
        pad_width = int(np.sqrt(width / 2) * 3)

        return np_pad(
            image,
            padding=((pad_height, pad_height), (pad_width, pad_width)),
            mode=PaddingMode.REFLECT,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        keep_aspect_ratio: bool | None,
        ensure_multiple_of: int | None,
        resample: PILImageResampling | None,
        do_pad: bool,
        do_rescale: bool,
        rescale_factor: float | None,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_pad:
                image = self.pad_image(image)
            if do_resize:
                image = self.resize(image, size, keep_aspect_ratio, ensure_multiple_of, resample)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        outputs: "ZoeDepthDepthEstimatorOutput",
        source_sizes: TensorType | list[tuple[int, int]] | None | None = None,
        target_sizes: TensorType | list[tuple[int, int]] | None | None = None,
        outputs_flipped: Union["ZoeDepthDepthEstimatorOutput", None] | None = None,
        do_remove_padding: bool | None | None = None,
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

                depth = tvF.resize(
                    depth,
                    size=[source_size[0] + 2 * pad_h, source_size[1] + 2 * pad_w],
                    interpolation=tvF.InterpolationMode.BICUBIC,
                    antialias=False,
                )

                if pad_h > 0:
                    depth = depth[:, pad_h:-pad_h, :]
                if pad_w > 0:
                    depth = depth[:, :, pad_w:-pad_w]

            if target_size is not None:
                target_size = [target_size[0], target_size[1]]
                depth = tvF.resize(
                    depth,
                    size=target_size,
                    interpolation=tvF.InterpolationMode.BICUBIC,
                    antialias=False,
                )
            depth = depth.squeeze(0)
            # depth.shape = [H, W]
            results.append({"predicted_depth": depth})

        return results


__all__ = ["ZoeDepthImageProcessorPil"]
