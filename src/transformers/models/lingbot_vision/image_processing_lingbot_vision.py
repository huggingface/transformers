# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for LingBot-Vision."""

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
    SizeDict,
    torch_pil_interpolation_mapping,
)
from ...utils import auto_docstring


@auto_docstring
class LingbotVisionImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # The reference implementation resizes PIL images before converting them to tensors. Tensor interpolation
        # differs measurably, so preserve that path for PIL-originated CPU images while retaining torchvision for others.
        if image.dtype != torch.uint8 or image.device.type != "cpu" or size.height is None or size.width is None:
            return super().resize(image=image, size=size, resample=resample, **kwargs)

        if not isinstance(resample, (PILImageResampling, int)):
            resample = torch_pil_interpolation_mapping.get(resample, PILImageResampling.BILINEAR)
        resample = resample if resample is not None else PILImageResampling.BILINEAR

        is_unbatched = image.ndim == 3
        images = image.unsqueeze(0) if is_unbatched else image
        resized_images = [
            tvF.pil_to_tensor(tvF.to_pil_image(single_image).resize((size.width, size.height), resample=resample))
            for single_image in images
        ]
        resized_images = torch.stack(resized_images)
        return resized_images.squeeze(0) if is_unbatched else resized_images

    def rescale_and_normalize(
        self,
        images: torch.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> torch.Tensor:
        # Match the reference arithmetic order: convert to float, rescale, then normalize.
        if do_rescale:
            images = self.rescale(images.to(dtype=torch.float32), rescale_factor)
        if do_normalize:
            images = self.normalize(images, image_mean, image_std)
        return images


__all__ = ["LingbotVisionImageProcessor"]
