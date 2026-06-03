# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Video processor for MiniMax M3 VL."""

import torch
from torchvision.transforms import InterpolationMode

from ...feature_extraction_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos
from .image_processing_minimax_m3_vl_fast import smart_resize


class MiniMaxM3VLVideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    total_pixels: int
    min_frames: int
    max_frames: int
    fps: float | int


class MiniMaxM3VLVideoProcessor(BaseVideoProcessor):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    do_sample_frames = False
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = 4 * 28 * 28
    max_pixels = 768 * 28 * 28
    total_pixels = int(64000 * 28 * 28 * 0.9)
    fps = 1.0
    min_frames = 4
    max_frames = 768
    valid_kwargs = MiniMaxM3VLVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLVideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped, grouped_idx = group_videos_by_shape(videos)
        resized_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked in grouped.items():
            bs, nf, c, h, w = stacked.shape
            rh, rw = (h, w)
            if do_resize:
                rh, rw = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
                stacked = stacked.view(bs * nf, c, h, w)
                stacked = self.resize(stacked, size=SizeDict(height=rh, width=rw), resample=resample)
                stacked = stacked.view(bs, nf, c, rh, rw)
            resized_grouped[shape] = stacked
        resized = reorder_videos(resized_grouped, grouped_idx)

        grouped, grouped_idx = group_videos_by_shape(resized)
        processed_grouped = {}
        grids = {}
        for shape, stacked in grouped.items():
            rh, rw = stacked.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat([patches, repeats], dim=1)
            bs, grid_t, c = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = rh // patch_size, rw // patch_size
            patches = patches.view(
                bs,
                grid_t,
                temporal_patch_size,
                c,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flat = patches.reshape(bs, grid_t * grid_h * grid_w, c * temporal_patch_size * patch_size * patch_size)
            processed_grouped[shape] = flat
            grids[shape] = [[grid_t, grid_h, grid_w]] * bs

        processed = reorder_videos(processed_grouped, grouped_idx)
        grids = reorder_videos(grids, grouped_idx)
        pixel_values_videos = torch.cat(processed, dim=0)
        video_grid_thw = torch.tensor(grids, dtype=torch.long)
        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["MiniMaxM3VLVideoProcessor"]
