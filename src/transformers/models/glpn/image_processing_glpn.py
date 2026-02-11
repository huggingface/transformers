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

from ...image_processing_backends import PilBackend, TorchVisionBackend
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging, requires_backends


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class GLPNImageProcessorKwargs(ImagesKwargs, total=False):
    """
    size_divisor (`int`, *optional*, defaults to 32):
        When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
        multiple of `size_divisor`.
    """

    size_divisor: int


class GLPNTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for GLPN with size_divisor resize."""

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        size_divisor: int = 32,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize so height and width are rounded down to the closest multiple of size_divisor."""
        height, width = image.shape[-2:]
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        return super().resize(
            image,
            SizeDict(height=new_h, width=new_w),
            resample=resample,
            **kwargs,
        )

    def preprocess(
        self,
        images: list["torch.Tensor"],
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
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        size_divisor: int = 32,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for GLPN."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample, size_divisor=size_divisor)
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class GLPNPilBackend(PilBackend):
    """PIL backend for GLPN with size_divisor resize."""

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

    def preprocess(
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
        disable_grouping: bool | None,
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


@auto_docstring(custom_intro="Constructs a GLPN image processor.")
class GLPNImageProcessor(BaseImageProcessor):
    valid_kwargs = GLPNImageProcessorKwargs

    _backend_classes = {
        "torchvision": GLPNTorchVisionBackend,
        "pil": GLPNPilBackend,
    }

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

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: TensorType | list[tuple[int, int]] | None = None,
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


__all__ = ["GLPNImageProcessor", "GLPNImageProcessorKwargs"]
