# Copyright 2026 the HuggingFace Team. All rights reserved.
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


import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import divide_to_patches
from ...image_utils import ImageInput, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


class TmlImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    """

    patch_size: int


@auto_docstring(custom_intro="Constructs a Tml image processor.")
class TmlImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = None
    default_to_square = False
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    patch_size = 40
    valid_kwargs = TmlImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[TmlImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[TmlImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        per_image_patches: list[torch.Tensor] = []
        num_patches: list[int] = []
        for image in images:
            height, width = image.shape[-2:]
            image_patches = divide_to_patches(image, patch_size)
            image_patches = torch.stack(image_patches, dim=0)
            image_patches = image_patches[..., None].repeat(1, 1, 1, 1, 2)
            per_image_patches.append(image_patches)
            num_patches.append(image_patches.shape[0])

        data = {
            "vision_patches_bthwc": torch.cat(per_image_patches, dim=0),
            "num_patches": torch.stack(num_patches, dim=0),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["TmlImageProcessor"]
