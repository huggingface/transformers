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
"""Image processor class for DPT."""

import math
from typing import TYPE_CHECKING, Union

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, logging, requires_backends
from ...utils.import_utils import requires
from .image_processing_dpt import DPTImageProcessorKwargs, get_resize_output_image_size


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

import torch
from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class DPTImageProcessorPil(PilBackend):
    """PIL backend for DPT with custom resize and pad."""

    valid_kwargs = DPTImageProcessorKwargs

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = False
    rescale_factor = 1 / 255
    ensure_multiple_of = 1
    keep_aspect_ratio = False

    def __init__(self, **kwargs: Unpack[DPTImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[DPTImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Handle extra inputs beyond images."""
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        data = {}
        data["pixel_values"] = self._preprocess(images, **images_kwargs)

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update({"do_normalize": False, "do_rescale": False})
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )

            processed_segmentation_maps = [
                processed_segmentation_map.squeeze(0).astype(np.int64)
                for processed_segmentation_map in processed_segmentation_maps
            ]
            data["labels"] = processed_segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def reduce_label(self, image: np.ndarray) -> np.ndarray:
        """Reduce label values by 1, replacing 0 with 255."""
        image[image == 0] = 255
        image = image - 1
        image[image == 254] = 255
        return image

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        ensure_multiple_of: int = 1,
        keep_aspect_ratio: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Resize with aspect ratio and multiple constraints."""
        if size.height is None or size.width is None:
            raise ValueError(f"Size must contain 'height' and 'width' keys. Got {size.keys()}")
        output_size = get_resize_output_image_size(
            image,
            output_size=(size.height, size.width),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
        )
        return super().resize(image, output_size, resample, **kwargs)

    def pad_image(self, image: np.ndarray, size_divisor: int = 1, **kwargs) -> np.ndarray:
        """Center pad image to be a multiple of size_divisor."""

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        height, width = image.shape[-2:]
        pad_size_left, pad_size_right = _get_pad(height, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)
        return np_pad(
            image,
            padding=((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)),
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_reduce_labels: bool,
        keep_aspect_ratio: bool,
        ensure_multiple_of: int,
        size_divisor: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Custom preprocessing for DPT."""
        processed_images = []
        for image in images:
            if do_reduce_labels:
                image = self.reduce_label(image)
            if do_resize:
                image = self.resize(
                    image, size, resample, ensure_multiple_of=ensure_multiple_of, keep_aspect_ratio=keep_aspect_ratio
                )
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            if do_pad and size_divisor is not None:
                image = self.pad_image(image, size_divisor)
            processed_images.append(image)
        return processed_images

    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None):
        """Converts the output of [`DPTForSemanticSegmentation`] into semantic segmentation maps."""
        requires_backends(self, "torch")
        logits = outputs.logits
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.numpy()
            semantic_segmentation = []
            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
        return semantic_segmentation

    def post_process_depth_estimation(
        self, outputs: "DepthEstimatorOutput", target_sizes: TensorType | list[tuple[int, int]] | None = None
    ) -> list[dict[str, TensorType]]:
        """Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions."""
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


__all__ = ["DPTImageProcessorPil"]
