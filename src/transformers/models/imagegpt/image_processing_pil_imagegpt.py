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
"""Image processor class for ImageGPT."""

from typing import Union

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
)


if is_torch_available():
    import torch


def squared_euclidean_distance(a, b):
    b = b.T
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=0)
    ab = np.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance(x, clusters)
    return np.argmin(d, axis=1)


# Adapted from transformers.models.imagegpt.image_processing_imagegpt.ImageGPTImageProcessorKwargs
class ImageGPTImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    clusters (`np.ndarray` or `list[list[int]]` or `torch.Tensor`, *optional*, defaults to `self.clusters`):
        The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overridden by `clusters`
        in `preprocess`.
    do_color_quantize (`bool`, *optional*, defaults to `self.do_color_quantize`):
        Controls whether to apply color quantization to convert continuous pixel values to discrete cluster indices.
        When True, each pixel is assigned to its nearest color cluster, enabling ImageGPT's discrete token modeling.
    """

    clusters: Union[np.ndarray, list[list[int]], "torch.Tensor"] | None
    do_color_quantize: bool


@auto_docstring
class ImageGPTImageProcessorPil(PilBackend):
    model_input_names = ["input_ids"]
    valid_kwargs = ImageGPTImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    do_color_quantize = True
    clusters = None
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_rescale = True
    do_normalize = True
    size = {"height": 256, "width": 256}
    do_resize = True

    def __init__(
        self,
        clusters: "list | np.ndarray | torch.Tensor | None" = None,  # keep as arg for backwards compatibility
        **kwargs: Unpack[ImageGPTImageProcessorKwargs],
    ):
        r"""
        clusters (`np.ndarray` or `list[list[int]]` or `torch.Tensor`, *optional*):
            The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overridden by `clusters`
            in `preprocess`.
        """
        if clusters is not None:
            clusters = np.array(clusters)
        super().__init__(clusters=clusters, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        do_color_quantize: bool | None = None,
        clusters: "list | np.ndarray | torch.Tensor | None" = None,
        **kwargs,
    ):
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        # If color quantization is requested, perform it; otherwise return pixel values
        if do_color_quantize:
            # Prepare clusters
            if clusters is None:
                raise ValueError("Clusters must be provided for color quantization.")
            # Convert to numpy array if needed
            clusters_np = np.array(clusters) if not isinstance(clusters, np.ndarray) else clusters

            # Stack channel-first images (B, C, H, W) and transpose to (B, H, W, C) for color quantization
            images_array = np.array(processed_images)
            images_hwc = images_array.transpose(0, 2, 3, 1)
            input_ids = color_quantize(images_hwc, clusters_np).reshape(
                images_array.shape[0], images_array.shape[2], images_array.shape[3]
            )

            # flatten to (batch_size, height*width)
            batch_size = input_ids.shape[0]
            input_ids = input_ids.reshape(batch_size, -1)

            # We need to convert back to a list to keep consistent behaviour across processors.
            input_ids = list(input_ids)
            return BatchFeature(data={"input_ids": input_ids}, tensor_type=return_tensors)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def to_dict(self):
        output = super().to_dict()
        if output.get("clusters") is not None and isinstance(output["clusters"], np.ndarray | torch.Tensor):
            output["clusters"] = output["clusters"].tolist()

        return output


__all__ = ["ImageGPTImageProcessorPil"]
