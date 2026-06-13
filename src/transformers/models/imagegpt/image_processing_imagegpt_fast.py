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
"""Fast Image processor class for ImageGPT."""

from typing import Optional, Union

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


def squared_euclidean_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances between all pixels and clusters.

    Args:
        a: (N, 3) tensor of pixel RGB values
        b: (M, 3) tensor of cluster RGB values

    Returns:
        (N, M) tensor of squared distances
    """
    b = b.t()  # (3, M)
    a2 = torch.sum(a**2, dim=1)  # (N,)
    b2 = torch.sum(b**2, dim=0)  # (M,)
    ab = torch.matmul(a, b)  # (N, M)
    d = a2[:, None] - 2 * ab + b2[None, :]  # Squared Euclidean Distance: a^2 - 2ab + b^2
    return d  # (N, M) tensor of squared distances


def color_quantize_torch(x: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
    """
    Assign each pixel to its nearest color cluster.

    Args:
        x: (H*W, 3) tensor of flattened pixel RGB values
        clusters: (n_clusters, 3) tensor of cluster RGB values

    Returns:
        (H*W,) tensor of cluster indices
    """
    d = squared_euclidean_distance_torch(x, clusters)
    return torch.argmin(d, dim=1)


class ImageGPTFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    clusters (`np.ndarray` or `list[list[int]]` or `torch.Tensor`, *optional*):
        The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overridden by `clusters`
        in `preprocess`.
    do_color_quantize (`bool`, *optional*, defaults to `True`):
        Controls whether to apply color quantization to convert continuous pixel values to discrete cluster indices.
        When True, each pixel is assigned to its nearest color cluster, enabling ImageGPT's discrete token modeling.
    """

    clusters: Optional[Union[np.ndarray, list[list[int]], torch.Tensor]]
    do_color_quantize: Optional[bool]


@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["input_ids"]
    resample = PILImageResampling.BILINEAR
    do_color_quantize = True
    clusters = None
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_rescale = True
    do_normalize = True
    valid_kwargs = ImageGPTFastImageProcessorKwargs

    def __init__(
        self,
        clusters: Optional[Union[list, np.ndarray, torch.Tensor]] = None,  # keep as arg for backwards compatibility
        **kwargs: Unpack[ImageGPTFastImageProcessorKwargs],
    ):
        r"""
        clusters (`np.ndarray` or `list[list[int]]` or `torch.Tensor`, *optional*):
            The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overridden by `clusters`
            in `preprocess`.
        """
        clusters = torch.as_tensor(clusters, dtype=torch.float32) if clusters is not None else None
        super().__init__(clusters=clusters, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: dict[str, int],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: dict[str, int],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)

        # If color quantization is requested, perform it; otherwise return pixel values
        if do_color_quantize:
            # Prepare clusters
            if clusters is None:
                raise ValueError("Clusters must be provided for color quantization.")
            # Convert to torch tensor if needed (clusters might be passed as list/numpy)
            clusters_torch = (
                torch.as_tensor(clusters, dtype=torch.float32) if not isinstance(clusters, torch.Tensor) else clusters
            ).to(pixel_values[0].device, dtype=pixel_values[0].dtype)

            # Group images by shape for batch processing
            # We need to check if the pixel values are a tensor or a list of tensors
            grouped_images, grouped_images_index = group_images_by_shape(
                pixel_values, disable_grouping=disable_grouping
            )
            # Process each group
            input_ids_grouped = {}

            for shape, stacked_images in grouped_images.items():
                input_ids = color_quantize_torch(
                    stacked_images.permute(0, 2, 3, 1).reshape(-1, 3), clusters_torch
                )  # (B*H*W, C)
                input_ids_grouped[shape] = input_ids.reshape(stacked_images.shape[0], -1).reshape(
                    stacked_images.shape[0], -1
                )  # (B, H, W)

            input_ids = reorder_images(input_ids_grouped, grouped_images_index)

            return BatchFeature(
                data={"input_ids": torch.stack(input_ids, dim=0) if return_tensors else input_ids},
                tensor_type=return_tensors,
            )

        pixel_values = torch.stack(pixel_values, dim=0) if return_tensors else pixel_values
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def to_dict(self):
        # Convert torch tensors to lists for JSON serialization
        output = super().to_dict()
        if output.get("clusters") is not None and isinstance(output["clusters"], torch.Tensor):
            output["clusters"] = output["clusters"].tolist()

        return output


__all__ = ["ImageGPTImageProcessorFast"]
