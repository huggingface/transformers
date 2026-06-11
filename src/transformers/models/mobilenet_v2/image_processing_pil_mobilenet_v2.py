# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for MobileNetV2."""

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    pass

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires


# Adapted from transformers.models.mobilenet_v2.image_processing_mobilenet_v2.MobileNetV2ImageProcessorKwargs
class MobileNetV2ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    do_reduce_labels: bool


@auto_docstring
class MobileNetV2ImageProcessorPil(PilBackend):
    """PIL backend for MobileNetV2 with reduce_label support."""

    valid_kwargs = MobileNetV2ImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False

    def __init__(self, **kwargs: Unpack[MobileNetV2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[MobileNetV2ImageProcessorKwargs],
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
        device: str | None = None,
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
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    # Nearest interpolation is used for segmentation maps instead of BILINEAR.
                    "resample": PILImageResampling.NEAREST,
                }
            )

            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )

            # Squeeze channel dimension and convert to int64
            processed_segmentation_maps = [
                processed_segmentation_map.squeeze(0).astype(np.int64)
                for processed_segmentation_map in processed_segmentation_maps
            ]
            data["labels"] = processed_segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def reduce_label(self, image: np.ndarray) -> np.ndarray:
        """Reduce label values by 1, replacing 0 with 255."""
        # Avoid using underflow conversion
        image[image == 0] = 255
        image = image - 1
        image[image == 254] = 255
        return image

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_reduce_labels: bool = False,
        **kwargs,
    ) -> list[np.ndarray]:
        """Custom preprocessing for MobileNetV2."""
        processed_images = []
        for image in images:
            if do_reduce_labels:
                image = self.reduce_label(image)
            if do_resize:
                image = self.resize(image, size, resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return processed_images

    @requires(backends=("torch",))
    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None):
        """
        Converts the output of [`MobileNetV2ForSemanticSegmentation`] into semantic segmentation maps.
        """
        import torch

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


__all__ = ["MobileNetV2ImageProcessorPil"]
