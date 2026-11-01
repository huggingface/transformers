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
"""DALI image processor class for Qwen2-VL."""

import warnings
from collections.abc import Iterable

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
import torch

from ...image_processing_backends import DaliBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType
from ...utils.import_utils import requires
from .image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs, smart_resize


@requires(backends=("torchvision",))
class Qwen2VLImageProcessorDali(DaliBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Qwen2VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    patchify = Qwen2VLImageProcessor.patchify
    get_number_of_image_patches = Qwen2VLImageProcessor.get_number_of_image_patches

    def __init__(self, **kwargs: Unpack[Qwen2VLImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else size
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if (max_pixels := kwargs.pop("max_pixels", None)) is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None or max_pixels is not None:
            warnings.warn(
                "Passing `min_pixels` and `max_pixels` to a processor call is deprecated and will be removed in v5.23. "
                "Pass in `size={'longest_edge': xxx, 'shortest_edge': xxx} to override the target size.`",
                FutureWarning,
            )

            size_dict = dict(size) if isinstance(size, (dict, SizeDict)) else {}
            if min_pixels is not None:
                size_dict["shortest_edge"] = min_pixels
            if max_pixels is not None:
                size_dict["longest_edge"] = max_pixels
            size = SizeDict(**size_dict)
        return super()._standardize_kwargs(size=size, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        device: "torch.device | None" = None,
        **kwargs,
    ) -> BatchFeature:
        with self.eval_context():
            batch = self.decode(images)
            if do_resize:
                targets = [
                    smart_resize(
                        height,
                        width,
                        factor=patch_size * merge_size,
                        min_pixels=size.shortest_edge,
                        max_pixels=size.longest_edge,
                    )
                    for height, width, _ in batch.shape
                ]
                # Per-sample sizes must be a batch of 1D arrays, a list of [height, width] pairs is rejected
                batch = ndd.resize(
                    batch,
                    size=ndd.batch([np.array(target, dtype=np.float32) for target in targets]),
                    interp_type=self.interpolation(resample),
                    antialias=True,
                )
            batch = self.rescale_and_normalize(batch, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            grouped_images, grouped_images_index = group_images_by_shape(
                [tensor.torch() for tensor in batch.tensors], disable_grouping=disable_grouping
            )
            processed_images_grouped = {}
            processed_grids = {}
            for shape, stacked_images in grouped_images.items():
                patches, grid_h, grid_w = self.patchify(
                    stacked_images,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    temporal_patch_size=temporal_patch_size,
                )
                processed_images_grouped[shape] = patches
                processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

            processed_images = reorder_images(processed_images_grouped, grouped_images_index)
            processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
            pixel_values = torch.cat(processed_images, dim=0).to(device if device is not None else "cpu")

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": torch.tensor(processed_grids_ordered, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )


__all__ = ["Qwen2VLImageProcessorDali"]
