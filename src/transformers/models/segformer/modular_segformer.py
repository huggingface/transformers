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
"""Fast Image processor class for Segformer."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from transformers.models.beit.image_processing_beit_fast import BeitFastImageProcessorKwargs, BeitImageProcessorFast

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
)


class SegformerFastImageProcessorKwargs(BeitFastImageProcessorKwargs):
    pass


class SegformerImageProcessorFast(BeitImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_reduce_labels = False
    do_center_crop = None
    crop_size = None

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput],
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[SegformerFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        batch_feature = self._preprocess(images, **images_kwargs)

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
                    "interpolation": F.InterpolationMode.NEAREST_EXACT,
                }
            )
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            ).pixel_values
            batch_feature["labels"] = processed_segmentation_maps.squeeze(1).to(torch.int64)

        return batch_feature

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: SizeDict,
        rescale_factor: float,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        disable_grouping: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:  # Return type can be list if return_tensors=None
        if do_reduce_labels:
            images = self.reduce_label(images)  # Apply reduction if needed

        # Group images by size for batched resizing
        resized_images = images
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                resized_stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
                resized_images_grouped[shape] = resized_stacked_images
            resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing (rescale/normalize)
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack images into a single tensor if return_tensors is set
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["SegformerImageProcessorFast"]
