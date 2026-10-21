# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Video processor for MiMo V2 multimodal models."""

import math

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension, PILImageResampling, SizeDict
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoInput, VideoMetadata
from .image_processing_mimo_v2 import _patchify_frames, _preprocess_frames


class MiMoV2VideoProcessorKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of consecutive frames in a temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor along each axis.
    temporal_compression_ratio (`int`, *optional*, defaults to 1):
        Number of temporal grids represented by one timestamp/token group.
    total_max_pixels (`int`, *optional*, defaults to `16384 * 28 * 28`):
        Total pixel budget, measured after temporal compression.
    min_frames (`int`, *optional*, defaults to 8):
        Minimum frame count when sampling at a target frame rate.
    max_frames (`int`, *optional*, defaults to 256):
        Maximum frame count when sampling at a target frame rate.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    temporal_compression_ratio: int
    total_max_pixels: int
    min_frames: int
    max_frames: int


@auto_docstring(
    custom_intro="Constructs a video processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2VideoProcessor(BaseVideoProcessor):
    size = {"shortest_edge": 4 * 28 * 28, "longest_edge": 4096 * 28 * 28}
    default_to_square = False
    resample = PILImageResampling.BILINEAR
    do_resize = True
    do_rescale = True
    do_normalize = True
    # RGB conversion is handled by BaseVideoProcessor._prepare_input_videos before _preprocess.
    # trf-ignore: TRF016
    do_convert_rgb = True
    do_sample_frames = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    temporal_compression_ratio = 1
    total_max_pixels = 16384 * 28 * 28
    fps = 2
    num_frames: int | None = None
    min_frames = 8
    max_frames = 256
    valid_kwargs = MiMoV2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiMoV2VideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        videos = [
            np.ascontiguousarray(video)
            if isinstance(video, np.ndarray) and any(stride < 0 for stride in video.strides)
            else video
            for video in videos
        ]
        return super()._prepare_input_videos(
            videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

    @auto_docstring
    def preprocess(self, videos: VideoInput, **kwargs: Unpack[MiMoV2VideoProcessorKwargs]) -> BatchFeature:
        # A sampling override takes precedence over the mutually exclusive instance default.
        if kwargs.get("num_frames") is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        elif kwargs.get("fps") is not None and "num_frames" not in kwargs:
            kwargs["num_frames"] = None
        elif self.num_frames is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        return super().preprocess(videos, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: float | None = None,
        temporal_patch_size: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample rounded, endpoint-exclusive frame indices for the MiMo V2 series.

        `num_frames` and `fps` are mutually exclusive. The sampled count is rounded up to a multiple of
        `temporal_patch_size`, retaining repeated indices when the video is shorter than one temporal patch.
        """
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive.")
        total_num_frames = metadata.total_num_frames
        if total_num_frames <= 0:
            raise ValueError("Video must contain at least one frame.")
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be positive.")
        if num_frames is None and fps is None:
            num_frames = self.num_frames
        if num_frames is not None:
            if not math.isfinite(num_frames) or num_frames <= 0:
                raise ValueError("num_frames must be finite and positive.")
        else:
            fps = self.fps if fps is None else fps
            if any(rate is None or not math.isfinite(rate) or rate <= 0 for rate in (metadata.fps, fps)):
                raise ValueError("A finite positive video_metadata.fps and sampling fps are required.")
            min_frames = self.min_frames if min_frames is None else min_frames
            max_frames = self.max_frames if max_frames is None else max_frames
            if not 1 <= min_frames <= max_frames:
                raise ValueError("Require 1 <= min_frames <= max_frames.")
            num_frames = max(min(math.ceil(total_num_frames / metadata.fps * fps), max_frames), min_frames)
        num_frames = math.ceil(min(num_frames, total_num_frames) / temporal_patch_size) * temporal_patch_size
        indices = np.linspace(0, total_num_frames, num_frames + 1)[:-1].round().astype(np.int64)
        return np.clip(indices, 0, total_num_frames - 1)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        temporal_compression_ratio: int,
        total_max_pixels: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        if min(temporal_patch_size, temporal_compression_ratio, total_max_pixels) <= 0:
            raise ValueError("Temporal factors and total_max_pixels must be positive.")
        pixel_values = []
        video_grid_thw = []
        for video in videos:
            if len(video) == 0:
                raise ValueError("Video must contain at least one frame.")
            max_pixels = max(
                size.shortest_edge, min(size.longest_edge, total_max_pixels * temporal_factor // len(video))
            )
            video_size = SizeDict(shortest_edge=size.shortest_edge, longest_edge=max_pixels)
            if pad_length := -len(video) % temporal_factor:
                video = torch.cat((video, video[-1:].expand(pad_length, -1, -1, -1)))
            frames = _preprocess_frames(
                video,
                size=video_size,
                resample=resample,
                do_resize=do_resize,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                merge_size=merge_size,
            )
            patches, grid_thw = _patchify_frames(
                frames, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size
            )
            pixel_values.append(patches)
            video_grid_thw.append(grid_thw)
        return BatchFeature(
            {"pixel_values_videos": torch.cat(pixel_values), "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["MiMoV2VideoProcessor"]
