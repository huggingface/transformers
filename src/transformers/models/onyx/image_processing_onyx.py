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
from __future__ import annotations

import itertools
import math

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD


logger = logging.get_logger(__name__)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_tokens: int,
) -> tuple[int, int]:
    """Pick the integer (H, W) grid closest to the aspect ratio under the token cap.

    Mirrors ``OnyxVisionEncoder._compute_grid_size`` so the processor needs no
    torch model import. Returns ``(target_h, target_w)``.
    """
    i_nph = height / patch_size
    i_npw = width / patch_size
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )
    candidates = [(nph, npw) for nph, npw in candidates if nph >= 1 and npw >= 1 and nph * npw <= max_tokens]
    if not candidates:
        candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
    nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - height / width))
    return nph * patch_size, npw * patch_size


class OnyxImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    TODO:
    """

    patch_size: int
    temporal_patch_size: int
    max_image_tokens: int
    downsample_factor: int


@auto_docstring(custom_intro="Constructs an Onyx image processor.")
class OnyxImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    patch_size = 14
    temporal_patch_size = 2
    downsample_factor = 2
    max_image_tokens = 4096

    valid_kwargs = OnyxImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[OnyxImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Onyx uses aspect_ratio_preserving_resize driven by patch_size,
        # not the standard `size` parameter. Temporarily disable do_resize so
        # the base validation doesn't raise an error
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        image: torch.Tensor,
        patch_size: int,
        max_tokens: int,
        resample: tvF.InterpolationMode,
    ) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_tokens=max_tokens,
        )

        if target_height == height and target_width == width:
            return image

        return tvF.resize(
            image,
            size=[target_height, target_width],
            interpolation=resample,
            antialias=True,
        )

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[OnyxImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int,
        temporal_patch_size: int,
        max_image_tokens: int,
        downsample_factor: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                stacked_images = self.aspect_ratio_preserving_resize(
                    image=stacked_images,
                    patch_size=patch_size * downsample_factor,
                    max_tokens=max_image_tokens,
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 6, 3, 2, 5, 7)
            flatten_patches = patches.reshape(
                batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )


__all__ = ["OnyxImageProcessor"]
