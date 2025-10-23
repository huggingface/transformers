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
"""Fast Image processor class for SAM2."""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...image_processing_utils import BatchFeature
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling, SizeDict
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor


class Sam2VideoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 1024, "width": 1024}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    model_input_names = ["pixel_values"]

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        size: SizeDict,
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        original_sizes = [video.shape[-2:] for video in videos]
        batch_feature = super()._preprocess(videos, size=size, return_tensors=return_tensors, **kwargs)
        batch_feature = BatchFeature(
            data={"original_sizes": original_sizes, **batch_feature.data},
            tensor_type=return_tensors,
        )
        return batch_feature

    def post_process_masks(
        self,
        masks,
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
        **kwargs,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarization and post-processing operations.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            max_hole_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a hole to fill.
            max_sprinkle_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a sprinkle to fill.
            apply_non_overlapping_constraints (`bool`, *optional*, defaults to `False`):
                Whether to apply non-overlapping constraints to the masks.

        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        # TODO: add connected components kernel for postprocessing
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise TypeError("Input masks should be a list of `torch.tensors` or a list of `np.ndarray`")
            interpolated_mask = F.interpolate(masks[i], original_size, mode="bilinear", align_corners=False)
            if apply_non_overlapping_constraints:
                interpolated_mask = self._apply_non_overlapping_constraints(interpolated_mask)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks


__all__ = ["Sam2VideoVideoProcessor"]
