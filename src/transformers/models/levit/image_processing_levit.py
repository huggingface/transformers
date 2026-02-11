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
"""Image processor class for LeViT."""

import numpy as np

from ...image_processing_utils import (
    BaseImageProcessor,
    PilBackend,
    TorchVisionBackend,
)
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...utils import auto_docstring, is_torch_available, is_torchvision_available, logging


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class LevitTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for LeViT with custom resize (shortest_edge * 256/224)."""

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize: shortest_edge is rescaled to int((256/224) * shortest_edge)."""
        if size.shortest_edge:
            shortest_edge = int((256 / 224) * size.shortest_edge)
            new_size_height, new_size_width = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=new_size_height, width=new_size_width)
        elif not size.height or not size.width:
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {list(size.keys())}."
            )
        return super().resize(image, size=size, resample=resample, **kwargs)


class LevitPilBackend(PilBackend):
    """PIL backend for LeViT with custom resize (shortest_edge * 256/224)."""

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize: shortest_edge is rescaled to int((256/224) * shortest_edge)."""
        if size.shortest_edge:
            shortest_edge = int((256 / 224) * size.shortest_edge)
            new_size_height, new_size_width = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=new_size_height, width=new_size_width)
        elif not size.height or not size.width:
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {list(size.keys())}."
            )
        return super().resize(
            image,
            size=size,
            resample=resample,
            **kwargs,
        )


@auto_docstring(custom_intro="Constructs a LeViT image processor.")
class LevitImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    _backend_classes = {
        "torchvision": LevitTorchVisionBackend,
        "pil": LevitPilBackend,
    }

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None


__all__ = ["LevitImageProcessor"]
