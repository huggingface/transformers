# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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
"""Video processor for Cosmos3 Edge."""

import math

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torchvision_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 32,
    min_pixels: int = 64 * 64,
    max_pixels: int = 24 * 1024 * 1024,
) -> tuple[int, int]:
    """Resize video frames while keeping the packed patch grid valid for Cosmos3 Edge."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    if min_pixels <= 0 or max_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            "`min_pixels` and `max_pixels` must be positive and `max_pixels` must be greater than or equal to "
            f"`min_pixels`, got min_pixels={min_pixels}, max_pixels={max_pixels}."
        )

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    resized_frames = math.ceil(num_frames / temporal_factor) * temporal_factor

    if resized_frames * resized_height * resized_width > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_frames * resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    return resized_height, resized_width


class Cosmos3EdgeVideoProcessorKwargs(VideosKwargs, total=False):
    """Keyword arguments for [`Cosmos3EdgeVideoProcessor`]."""

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


@add_start_docstrings(
    "Constructs a video processor that dynamically resizes and packs Cosmos3 Edge video frames.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            Spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            Temporal patch size of the vision encoder. Cosmos3 Edge processes each sampled frame independently.
        merge_size (`int`, *optional*, defaults to 2):
            Spatial merge size applied by the vision projector.
    """,
)
class Cosmos3EdgeVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 64 * 64, "longest_edge": 24 * 1024 * 1024}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = Cosmos3EdgeVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeVideoProcessorKwargs]):
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else dict(size)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Uniformly sample frames with the checkpoint's default two frames per second policy."""
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        fps = self.fps if fps is None else fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Cosmos3 Edge samples video frames using fps, but input video metadata did not provide an fps. "
                    "Defaulting to fps=24. Pass `video_metadata` for accurate timestamps."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        return np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.reshape(batch_size * num_frames, channels, height, width)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.reshape(
                    batch_size, num_frames, channels, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos

        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}

        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            patch_group_size = patch_size * merge_size
            if resized_height % patch_group_size or resized_width % patch_group_size:
                raise ValueError(
                    "Video frames must have dimensions divisible by `patch_size * merge_size`, got "
                    f"height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                    f"merge_size={merge_size}."
                )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            batch_size, grid_t, channels = stacked_videos.shape[:3]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size

            patches = stacked_videos.reshape(
                batch_size,
                grid_t,
                channels,
                grid_height,
                patch_size,
                grid_width,
                patch_size,
            )
            patches = patches.permute(0, 1, 3, 5, 4, 6, 2)
            processed_videos_grouped[shape] = patches.reshape(
                batch_size, grid_t * grid_height * grid_width, channels * patch_size * patch_size
            )
            processed_grids[shape] = [[grid_t, grid_height, grid_width]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        """Return the number of pre-projector vision patches for a video size."""
        videos_kwargs = videos_kwargs or {}
        size = videos_kwargs.get("size", self.size)
        if isinstance(size, SizeDict):
            min_pixels, max_pixels = size.shortest_edge, size.longest_edge
        else:
            min_pixels, max_pixels = size["shortest_edge"], size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        resized_height, resized_width = smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_t = math.ceil(num_frames / temporal_patch_size)
        return grid_t * (resized_height // patch_size) * (resized_width // patch_size)


__all__ = ["Cosmos3EdgeVideoProcessor"]
