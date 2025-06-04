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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
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
    infer_channel_dimension_format,
    is_torch_tensor,
    pil_torch_interpolation_mapping,
    validate_kwargs,
    make_list_of_images,
    to_numpy_array,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
)
from ...utils.deprecation import deprecate_kwarg


class SegformerFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_reduce_labels: Optional[bool]


@add_start_docstrings(
    "Constructs a fast Segformer image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """,
)
class SegformerImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_reduce_labels = False
    valid_kwargs = SegformerFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[SegformerFastImageProcessorKwargs]):
        # Allow explicit setting of do_reduce_labels or use default
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to save support of deprecated `reduce_labels` in old configs
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in image_processor_dict:
            image_processor_dict["do_reduce_labels"] = image_processor_dict.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    def reduce_label(self, labels: list["torch.Tensor"]) -> list["torch.Tensor"]:
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype), label)
            labels[idx] = label

        return labels

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: SizeDict,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature: # Return type can be list if return_tensors=None
        if do_reduce_labels:
            images = self.reduce_label(images) # Apply reduction if needed

        # Group images by size for batched resizing
        resized_images = images
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                resized_stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
                resized_images_grouped[shape] = resized_stacked_images
            resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing (rescale/normalize)
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # Stack images into a single tensor if return_tensors is set
        if return_tensors:
            processed_images = torch.stack(processed_images, dim=0)

        return processed_images


    def _preprocess_segmentation_maps(
        self,
        segmentation_maps,
        **kwargs,
    ):
        """Preprocesses a single segmentation map."""
        processed_segmentation_maps = []
        added_dimension_list = []
        for segmentation_map in segmentation_maps:
            segmentation_map = to_numpy_array(segmentation_map)
            # Add an axis to the segmentation maps for transformations.
            if segmentation_map.ndim == 2:
                segmentation_map = segmentation_map[None, ...]
                added_dimension = True
                input_data_format = ChannelDimension.FIRST
            else:
                added_dimension = False
                if input_data_format is None:
                    input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

            processed_segmentation_maps.append(torch.tensor(segmentation_map))
            added_dimension_list.append(added_dimension)

        kwargs["do_normalize"] = False
        kwargs["do_rescale"] = False
        kwargs["input_data_format"] = ChannelDimension.FIRST
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[PILImageResampling.NEAREST] 
            if PILImageResampling.NEAREST in pil_torch_interpolation_mapping 
            else kwargs.get("interpolation")
        )

        processed_segmentation_maps = self._preprocess(
            images=processed_segmentation_maps,
            **kwargs
        )
        final_segmentation_maps = []
        is_batched = isinstance(processed_segmentation_maps, torch.Tensor)
        for idx, seg_map in enumerate(processed_segmentation_maps):
            current_map = seg_map if is_batched else processed_segmentation_maps[idx]
            if added_dimension_list[idx]:
                # Squeeze dim 1 if batched (B, C, H, W), dim 0 if not batched (C, H, W)
                squeeze_dim = 1 if is_batched and current_map.ndim > 2 else 0
                if current_map.ndim > squeeze_dim and current_map.shape[squeeze_dim] == 1:
                    current_map = current_map.squeeze(squeeze_dim)

            current_map = current_map.to(torch.int64)
            final_segmentation_maps.append(current_map)

        # Return stacked tensor or list, matching the output format of _preprocess
        if is_batched:
            return torch.stack(final_segmentation_maps, dim=0)
        else:
            return final_segmentation_maps

    def __call__(self, images, segmentation_maps=None, **kwargs):
        # Overrides the `__call__` method of the `Preprocessor` class such that the images and segmentation maps can both
        # be passed in as positional arguments.
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="4.41.0")
    @add_start_docstrings(
        "Preprocess an image or batch of images and optionally segmentation maps.",
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, # Inherit base docs
         """Optionally preprocesses segmentation maps alongside the images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess. Expects a single or batch of segmentation maps. For batch processing,
                the number of segmentation maps should match the number of images.
            do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255. Only applied if `segmentation_maps` are provided.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `torch.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
        """,
    )
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        **kwargs: Unpack[SegformerFastImageProcessorKwargs],
    ) -> BatchFeature:
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
        prepared_images = self._prepare_input_images(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Prepare segmentation maps
        prepared_segmentation_maps = None
        if segmentation_maps is not None:
            # Segmentation maps should not be converted to RGB
            prepared_segmentation_maps = make_list_of_images(
                images = segmentation_maps,
                expected_ndims = 2
            )
            if len(prepared_images) != len(prepared_segmentation_maps):
                 raise ValueError("Number of images and segmentation maps must match.")

        # Update kwargs that need further processing before being validated (e.g., size)
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs applicable to the core preprocessing steps
        self._validate_preprocess_kwargs(**kwargs)

        # Get interpolation mode for images
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        # Pop kwargs not needed by internal methods or handled differently
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")

        # Process images using _preprocess
        processed_images = self._preprocess(
            images=prepared_images,
            **kwargs,
        )
        data = {"pixel_values": processed_images}

        # Process segmentation maps using _preprocess_segmentation_maps
        if prepared_segmentation_maps is not None:
            processed_segmentation_maps = self._preprocess_segmentation_maps(
                segmentation_maps=prepared_segmentation_maps,
                **kwargs,
            )
            data["labels"] = processed_segmentation_maps

        return BatchFeature(data=data)

    def post_process_semantic_segmentation(self, outputs, target_sizes: Optional[List[Tuple]] = None):
        """
        Converts the output of [`SegformerForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`SegformerForSemanticSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # TODO: add support for other frameworks
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    f"Make sure that you pass in as many target sizes ({len(target_sizes)}) as "
                    f"the batch dimension of the logits ({len(logits)})"
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


__all__ = ["SegformerImageProcessorFast"]
