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

from typing import Optional, Union

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    is_torch_tensor,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from .image_processing_mobilevit import MobileVitImageProcessorKwargs


@auto_docstring
class MobileViTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 256, "width": 256}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = None
    do_convert_rgb = None
    do_flip_channel_order = True
    do_reduce_labels = False
    valid_kwargs = MobileVitImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[MobileVitImageProcessorKwargs]):
        super().__init__(**kwargs)

    # Copied from transformers.models.beit.image_processing_beit_fast.BeitImageProcessorFast.reduce_label
    def reduce_label(self, labels: list["torch.Tensor"]):
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype), label)
            labels[idx] = label

        return labels

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[MobileVitImageProcessorKwargs],
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
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[MobileVitImageProcessorKwargs],
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
                    "do_rescale": False,
                    "do_flip_channel_order": False,
                    # Nearest interpolation is used for segmentation maps instead of BILINEAR.
                    "interpolation": tvF.InterpolationMode.NEAREST_EXACT,
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
        do_resize: bool,
        size: SizeDict | None,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float | None,
        do_center_crop: bool,
        crop_size: SizeDict | None,
        do_flip_channel_order: bool,
        disable_grouping: bool,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []

        if do_reduce_labels:
            images = self.reduce_label(images)

        # Group images by shape for more efficient batch processing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        # Process each group of images with the same shape
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images

        # Reorder images to original sequence
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group again after resizing (in case resize produced different sizes)
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(image=stacked_images, size=crop_size)
            if do_rescale:
                stacked_images = self.rescale(image=stacked_images, scale=rescale_factor)
            if do_flip_channel_order:
                # For batched images, we need to handle them all at once
                if stacked_images.ndim > 3 and stacked_images.shape[1] >= 3:
                    # Flip RGB → BGR for batched images
                    flipped = stacked_images.clone()
                    flipped[:, 0:3] = stacked_images[:, [2, 1, 0], ...]
                    stacked_images = flipped

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack all processed images if return_tensors is specified

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_semantic_segmentation(self, outputs, target_sizes: list[tuple] | None = None):
        """
        Converts the output of [`MobileNetV2ForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileNetV2ForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            if is_torch_tensor(target_sizes):
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


__all__ = ["MobileViTImageProcessorFast"]
