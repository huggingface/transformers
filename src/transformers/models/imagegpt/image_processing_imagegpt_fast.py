# coding=utf-8
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

"""Fast image processor for ImageGPT."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from ...image_processing_utils import BaseImageProcessorFast, register_for_auto_class
from ...image_utils import PILImageResampling
from ...utils import logging, TensorType

logger = logging.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Computes pairwise squared Euclidean distances between all pixels and clusters
def squared_euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b = b.T
    a2 = torch.sum(a**2, dim=1)
    b2 = torch.sum(b**2, dim=0)
    ab = torch.matmul(a, b)
    return a2[:, None] - 2 * ab + b2[None, :]


# Assigns each pixel to its nearest color cluster
def color_quantize(x: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance(x, clusters)
    return torch.argmin(d, dim=1)


@register_for_auto_class("AutoImageProcessor")
class ImageGPTImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["input_ids"]

    def __init__(
        self,
        clusters: Optional[Union[List[List[float]], np.ndarray]] = None,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        do_color_quantize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clusters = (
            torch.tensor(clusters, dtype=torch.float32).to(device)
            if clusters is not None
            else None
        )
        self.do_resize = do_resize
        self.size = size or {"height": 256, "width": 256}
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_color_quantize = do_color_quantize

    def _preprocess(
        self,
        images: List[torch.Tensor],
        do_resize: bool,
        size: Dict[str, int],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: Dict[str, int],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        if do_resize:
            images = [
                F.interpolate(img.unsqueeze(0), size=(size["height"], size["width"]), mode="bilinear", align_corners=False)[0]
                for img in images
            ]

        if do_normalize:
            images = [img * 2.0 - 1.0 for img in images]

        if self.do_color_quantize:
            if self.clusters is None:
                raise ValueError("Color quantization requires `clusters` to be set.")

            input_ids = []
            for img in images:
                img = img.permute(1, 2, 0).contiguous().view(-1, 3)
                ids = color_quantize(img, self.clusters.to(img.device))
                input_ids.append(ids)

            return {"input_ids": torch.stack(input_ids)}

        return {"pixel_values": torch.stack(images)}

    def to_dict(self):
        output = super().to_dict()
        output.update({
            "clusters": self.clusters.cpu().numpy().tolist() if self.clusters is not None else None,
            "do_resize": self.do_resize,
            "size": self.size,
            "resample": self.resample,
            "do_normalize": self.do_normalize,
            "do_color_quantize": self.do_color_quantize,
        })
        return output

    @classmethod
    def from_dict(cls, image_processor_dict):
        return cls(**image_processor_dict)


__all__ = ["ImageGPTImageProcessorFast"]
