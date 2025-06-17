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
"""Fast Image processor class for MobileViT."""

from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_torch_available


if is_torch_available():
    import torch


class MobileViTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Keyword arguments for MobileViTImageProcessorFast that extend the default ones
    to include channel flipping support.

    Args:
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR. This matches the behavior of the
            slow MobileViT image processor.
    """

    do_flip_channel_order: Optional[bool]


@auto_docstring
class MobileViTImageProcessorFast(BaseImageProcessorFast):
    # Default values verified against the slow MobileViTImageProcessor
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 224}
    crop_size = {"height": 256, "width": 256}
    default_to_square = False
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_flip_channel_order = True
    # MobileViT slow processor does NOT have normalization, so set to None
    do_normalize = None
    valid_kwargs = MobileViTFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[MobileViTFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self, images, **kwargs: Unpack[MobileViTFastImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["torch.nn.functional.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_flip_channel_order: bool,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, interpolation=interpolation
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
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            # Handle channel flipping (RGB to BGR conversion)
            if do_flip_channel_order:
                # Flip the channel dimension (channels are at dimension 1 for batched tensors)
                stacked_images = stacked_images.flip(1)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_images = (
            torch.stack(processed_images, dim=0) if return_tensors else processed_images
        )

        return BatchFeature(
            data={"pixel_values": processed_images}, tensor_type=return_tensors
        )

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        """
        Converts the output of [`MobileViTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileViTForSemanticSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # Import torch here to avoid errors if torch is not available
        if not is_torch_available():
            raise ImportError(
                "PyTorch is required for post-processing semantic segmentation outputs."
            )

        import torch
        import torch.nn.functional as F

        logits = outputs.logits

        # Resize logits if target sizes are provided
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            resized_logits = []
            for i in range(len(logits)):
                resized_logit = F.interpolate(
                    logits[i].unsqueeze(dim=0),
                    size=target_sizes[i],
                    mode="bilinear",
                    align_corners=False,
                )
                resized_logits.append(resized_logit[0])
            logits = torch.stack(resized_logits)

        semantic_segmentation = logits.argmax(dim=1)
        semantic_segmentation = [
            semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
        ]

        return semantic_segmentation


__all__ = ["MobileViTImageProcessorFast"]
