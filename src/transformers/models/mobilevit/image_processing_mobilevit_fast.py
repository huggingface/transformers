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

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    is_torch_tensor,
    make_list_of_images,
    pil_torch_interpolation_mapping,
    validate_kwargs,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
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


class MobileVitFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):
        Whether to flip the color channels from RGB to BGR or vice versa.
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    do_flip_channel_order: Optional[bool]
    do_reduce_labels: Optional[bool]


@auto_docstring
class MobileViTImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
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
    valid_kwargs = MobileVitFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[MobileVitFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    # Copied from transformers.models.beit.image_processing_beit_fast.BeitImageProcessorFast.reduce_label
    def reduce_label(self, labels: list["torch.Tensor"]):
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype), label)
            labels[idx] = label

        return label

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        do_resize: bool,
        size: Optional[SizeDict],
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_center_crop: bool,
        crop_size: Optional[SizeDict],
        do_flip_channel_order: bool,
        disable_grouping: bool,
        return_tensors: Optional[Union[str, TensorType]],
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
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return processed_images

    def _preprocess_images(
        self,
        images,
        **kwargs,
    ):
        """Preprocesses images."""
        kwargs["do_reduce_labels"] = False
        processed_images = self._preprocess(images=images, **kwargs)
        return processed_images

    def _preprocess_segmentation_maps(
        self,
        segmentation_maps,
        **kwargs,
    ):
        """Preprocesses segmentation maps."""
        processed_segmentation_maps = []
        for segmentation_map in segmentation_maps:
            segmentation_map = self._process_image(
                segmentation_map, do_convert_rgb=False, input_data_format=ChannelDimension.FIRST
            )

            if segmentation_map.ndim == 2:
                segmentation_map = segmentation_map[None, ...]

            processed_segmentation_maps.append(segmentation_map)

        kwargs["do_rescale"] = False
        kwargs["do_flip_channel_order"] = False
        kwargs["interpolation"] = pil_torch_interpolation_mapping[PILImageResampling.NEAREST]
        processed_segmentation_maps = self._preprocess(images=processed_segmentation_maps, **kwargs)

        processed_segmentation_maps = processed_segmentation_maps.squeeze(1)

        processed_segmentation_maps = processed_segmentation_maps.to(torch.int64)
        return processed_segmentation_maps

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs: Unpack[MobileVitFastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        images = self._prepare_input_images(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Prepare segmentation maps
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(images=segmentation_maps, expected_ndims=2)

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        images = self._preprocess_images(
            images=images,
            **kwargs,
        )

        if segmentation_maps is not None:
            segmentation_maps = self._preprocess_segmentation_maps(
                segmentation_maps=segmentation_maps,
                **kwargs,
            )
            return BatchFeature(data={"pixel_values": images, "labels": segmentation_maps})

        return BatchFeature(data={"pixel_values": images})

    def post_process_semantic_segmentation(self, outputs, target_sizes: Optional[list[tuple]] = None):
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
