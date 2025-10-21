# coding=utf-8
# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_vision_available,
    logging,
)
from ...utils.import_utils import requires
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    infer_channel_dimension_format,
    reorder_videos,
)
from .image_processing_ernie4_5_vl import smart_resize


if is_vision_available():
    from PIL import ImageDraw, ImageFont
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image


logger = logging.get_logger(__name__)


# TODO: how do we move this
FONT_PATH = os.path.join(Path(__file__).parent.absolute(), "Roboto-Regular.ttf")
if not os.path.exists(FONT_PATH):
    ttf = requests.get("https://paddlenlp.bj.bcebos.com/vision-language-models/materials/Roboto-Regular.ttf")
    open(FONT_PATH, "wb").write(ttf.content)


class Ernie4_5_VLVideoProcessorInitKwargs(VideosKwargs, total=False):
    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


@add_start_docstrings(
    "Constructs a fast Qwen2-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        min_pixels (`int`, *optional*, defaults to `299 * 28 * 28`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `1196 * 28 * 28`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        min_frames (`int`, *optional*, defaults to 16):
            The minimum number of frames that can be sampled.
        max_frames (`int`, *optional*, defaults to 180):
            The maximum number of frames that can be sampled.
    """,
)
@requires(backends=("torchvision",))
class Ernie4_5_VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 299 * 28 * 28, "longest_edge": 1196 * 28 * 28}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    min_pixels = 299 * 28 * 28
    max_pixels = 1196 * 28 * 28
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 16
    max_frames = 180
    do_sample_frames = True
    valid_kwargs = Ernie4_5_VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Ernie4_5_VLVideoProcessorInitKwargs]):
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        if temporal_patch_size is None or temporal_patch_size != 2:
            raise ValueError("`Ernie 4.5 VL` only supports a temporal patch size of 2")

        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = self.size if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ):
        min_frames = min_frames if min_frames is not None else self.min_frames
        max_frames = max_frames if max_frames is not None else self.max_frames
        total_num_frames = metadata.total_num_frames

        max_frames = min(max_frames, total_num_frames)
        num_frames = min(max(total_num_frames, min_frames), max_frames)

        indices = torch.arange(0, total_num_frames, total_num_frames / num_frames).int()
        # test = np.linspace(start=0, stop=total_num_frames, num=num_frames + 1).astype(int)[:-1]

        return indices

    def _convert_timestamp(self, time_stamp_in_seconds):
        """Convert to `time: hr:min:sec` format"""
        hours = 0
        while time_stamp_in_seconds >= 3600:
            hours += 1
            time_stamp_in_seconds -= 3600
        mins = 0
        while time_stamp_in_seconds >= 60:
            mins += 1
            time_stamp_in_seconds -= 60
        return f"time: {int(hours):02d}:{int(mins):02d}:{time_stamp_in_seconds:05.02f}"

    def _render_image_with_timestamp(self, image: torch.Tensor, timestamp: str):
        """
        TODO: I have not found a function in torchvision to do this
        """
        image = to_pil_image(image)

        font_size = int(min(*image.size) * 0.1)
        outline_size = int(font_size * 0.1)
        font = ImageFont.truetype(FONT_PATH, font_size)

        # Draw a black timestamp with a white border
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, 0),
            timestamp,
            font=font,
            fill=(0, 0, 0),
            stroke_width=outline_size,
            stroke_fill=(255, 255, 255),
        )

        return pil_to_tensor(image)

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional[str] = None,
        video_metadata: Optional[list[VideoMetadata]] = None,
    ) -> list["torch.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        processed_videos = []
        for video, metadata in zip(videos, video_metadata):
            # Check for attributes that are necessary to draw timestamps on frames
            if metadata is None:
                raise ValueError("Need video metadata to process videos in Ernie 4.5 VL")
            elif metadata.fps is None:
                metadata.fps = self.fps
                logger.warning_once(
                    f"Could not infer the fps of a video, defaulting to {self.fps}. "
                    "This likely leads to unexpected behavior, so make sure to properly load videos."
                )

            # `make_batched_videos` always returns a 4D array per video
            if isinstance(video, np.ndarray):
                # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
                video = torch.from_numpy(video).contiguous()

            # Infer the channel dimension format if not provided
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(video)

            if input_data_format == ChannelDimension.LAST:
                video = video.permute(0, 3, 1, 2).contiguous()

            # TODO: check for correctness
            # See `timestamp_converting` and `render_single_image_with_timestamp` in og
            for idx, frame in enumerate(video):
                video[idx] = self._render_image_with_timestamp(
                    frame, self._convert_timestamp(metadata.timestamps[idx])
                )

            # last frame is copied if uneven (mitigating issues for temporal patch size)
            if video.shape[0] % 2 != 0:
                video = torch.cat((video, video[-1].detach().clone()[None, ...]), dim=0)

            if device is not None:
                video = video.to(device)

            processed_videos.append(video)
        return processed_videos

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)

            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # [batch, grid_t, grid_h/merge, grid_w/merge, merge, merge, channel, patch, patch]
            patches = patches.permute(0, 1, 3, 6, 4, 7, 2, 5, 8)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["Ernie4_5_VLVideoProcessor"]
