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
"""Image processor class for GLPN."""

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, logging, requires_backends
from ...utils.import_utils import requires
from .image_processing_glpn import GLPNImageProcessorKwargs


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

logger = logging.get_logger(__name__)


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class GLPNImageProcessorPil(PilBackend):
    """PIL backend for GLPN with size_divisor resize."""

    valid_kwargs = GLPNImageProcessorKwargs

    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    resample = PILImageResampling.BILINEAR
    size_divisor = 32

    def __init__(self, **kwargs: Unpack[GLPNImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # pop `do_resize` to not raise an error as `size` is not used (we use size_divisor)
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[GLPNImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        size_divisor: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Resize so height and width are rounded down to the closest multiple of size_divisor."""
        height, width = image.shape[-2:]
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        return super().resize(image, SizeDict(height=new_h, width=new_w), resample, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        return_tensors: str | TensorType | None,
        size_divisor: int = 32,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for GLPN."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, size_divisor=size_divisor)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self, outputs: "DepthEstimatorOutput", target_sizes: TensorType | list[tuple[int, int]] | None = None
    ) -> list[dict[str, TensorType]]:
        """
        Convert raw model outputs to final depth predictions.
        Only supports PyTorch.
        """
        requires_backends(self, "torch")
        predicted_depth = outputs.predicted_depth
        if target_sizes is not None and len(predicted_depth) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )
        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = depth[None, None, ...]
                depth = torch.nn.functional.interpolate(depth, size=target_size, mode="bicubic", align_corners=False)
                depth = depth.squeeze(0).squeeze(0)
            results.append({"predicted_depth": depth})
        return results


__all__ = ["GLPNImageProcessorPil"]
