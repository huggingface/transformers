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

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import PILImageResampling
from ...utils import auto_docstring


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
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


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


@auto_docstring
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    """
    Constructs a fast ImageGPT image processor.

    This processor can be used to resize images to a smaller resolution (such as 32x32 or 64x64),
    normalize them and finally color quantize them to obtain sequences of "pixel values" (color clusters).
    """

    model_input_names = ["input_ids"]

    # Defaults largely aligned with the slow processor, except normalization which we do manually to [-1, 1]
    resample = PILImageResampling.BILINEAR
    size = {"height": 256, "width": 256}
    do_resize = True
    # We do NOT use the base normalization/rescale as ImageGPT expects (x/127.5 - 1)
    do_rescale = False
    do_normalize = False

    do_color_quantize = True
    clusters = None  # Must be set at instantiation

    def __init__(
        self,
        clusters: Optional[Union[list, np.ndarray]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store clusters as numpy for JSON serializability. Convert to torch in _preprocess when needed.
        if clusters is not None:
            self.clusters = np.asarray(clusters, dtype=np.float32)
        else:
            self.clusters = None
        # Default: follow ImageGPT behavior (normalize by default). We stash here and force base to skip.
        self._do_normalize_imagegpt = kwargs.get("do_normalize", True)

    def _further_process_kwargs(self, **kwargs):
        # Let the base process size/crop and other standard kwargs first
        kwargs = super()._further_process_kwargs(**kwargs)
        if "do_normalize" in kwargs and kwargs["do_normalize"] is not None:
            self._do_normalize_imagegpt = kwargs["do_normalize"]
        # Force base pipeline to skip its rescale/normalize validation and logic
        kwargs["do_rescale"] = False
        kwargs["do_normalize"] = False
        return kwargs

    def _preprocess(
        self,
        images,
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[list, np.ndarray, torch.Tensor]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        # Run standard fast pipeline (resize, crop, batching) without rescale/normalize
        base_batch = super()._preprocess(
            images, return_tensors=return_tensors, **kwargs
        )
        pixel_values = base_batch["pixel_values"]  # Tensor [B,C,H,W] or list of [C,H,W]

        # Apply ImageGPT normalization when requested: [-1, 1]
        do_normalize = getattr(self, "_do_normalize_imagegpt", True)
        if isinstance(pixel_values, torch.Tensor):
            normalized = pixel_values.to(dtype=torch.float32)
            if do_normalize:
                normalized = normalized / 127.5 - 1.0
        else:
            normalized = [img.to(dtype=torch.float32) for img in pixel_values]
            if do_normalize:
                normalized = [img / 127.5 - 1.0 for img in normalized]

        # If color quantization is requested, perform it; otherwise return pixel values
        do_color_quantize = (
            do_color_quantize
            if do_color_quantize is not None
            else self.do_color_quantize
        )
        if do_color_quantize:
            # Prepare clusters
            clusters = clusters if clusters is not None else self.clusters
            if clusters is None:
                raise ValueError("Clusters must be provided for color quantization.")
            clusters_torch = torch.as_tensor(clusters, dtype=torch.float32)

            # Helper for clarity: quantize a single image [C,H,W] -> [H*W]
            def _quantize_one_image(
                image_chw: torch.Tensor, clusters_ref: torch.Tensor
            ) -> torch.Tensor:
                device_clusters = clusters_ref.to(
                    image_chw.device, dtype=image_chw.dtype
                )
                img_hwc = image_chw.permute(1, 2, 0)
                pixels = img_hwc.reshape(-1, 3)
                return color_quantize_torch(pixels, device_clusters)

            images_list = list(normalized)

            ids_list = [_quantize_one_image(img, clusters_torch) for img in images_list]

            if return_tensors == "pt":
                input_ids = torch.stack(ids_list, dim=0)
                pixel_values_out = torch.stack(images_list, dim=0)
            else:
                input_ids = ids_list
                pixel_values_out = images_list

            from ...image_processing_utils import BatchFeature

            return BatchFeature(
                data={"input_ids": input_ids, "pixel_values": pixel_values_out},
                tensor_type=return_tensors,
            )

        # Otherwise, return pixel values (normalized or not depending on flag)
        base_batch["pixel_values"] = normalized
        return base_batch

    def to_dict(self):
        # Convert numpy arrays to lists for JSON serialization
        output = super().to_dict()
        if output.get("clusters") is not None and isinstance(
            output["clusters"], np.ndarray
        ):
            output["clusters"] = output["clusters"].tolist()
        # ImageGPT does not use base mean/std normalization; keep these None for parity with slow processor
        # output["image_mean"] = None
        # output["image_std"] = None
        # No rescaling in fast ImageGPT path

        # Need to set these valus to match with slow processor during testing
        output["rescale_factor"] = None
        output["do_rescale"] = None
        output["do_color_quantize"] = bool(getattr(self, "do_color_quantize", True))
        output.pop("_do_normalize_imagegpt", None)
        return output


__all__ = ["ImageGPTImageProcessorFast"]
