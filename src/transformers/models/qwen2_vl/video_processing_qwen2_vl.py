# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""video processor class for Qwen2-VL."""

import math

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


# Copied from transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLVideoProcessorInitKwargs(VideosKwargs, total=False):
    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


@auto_docstring
class Qwen2VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 28 * 28, "longest_edge": 28 * 28 * 768}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = Qwen2VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Qwen2VLVideoProcessorInitKwargs]):
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = kwargs.pop("size", None)
        size = self.size if size is None else size
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if (max_pixels := kwargs.pop("max_pixels", None)) is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
        return super()._standardize_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformly. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            temporal_patch_size (`int`, *optional*):
                The temporal patch size of the vision encoder. Number of sampled frames will be rounded to be divisible by frame factor.
            min_frames (`int`, *optional*):
                The minimum number of frames that can be sampled.
            max_frames (`int`, *optional*):
                The maximum number of frames that can be sampled.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        num_frames = num_frames if num_frames is not None else self.num_frames
        fps = fps if fps is not None else self.fps
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        min_frames = min_frames if min_frames is not None else self.min_frames
        max_frames = max_frames if max_frames is not None else self.max_frames
        total_num_frames = metadata.total_num_frames

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is not None:
            num_frames = round(num_frames / temporal_patch_size) * temporal_patch_size
        elif fps is not None:
            if metadata is None or metadata.fps is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            max_frames = math.floor(min(max_frames, total_num_frames) / temporal_patch_size) * temporal_patch_size
            num_frames = total_num_frames / metadata.fps * fps
            num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
            num_frames = math.floor(num_frames / temporal_patch_size) * temporal_patch_size

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The inferred `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
                "Decrease `num_frames` or `fps` for sampling."
            )

        if num_frames is not None:
            indices = torch.arange(0, total_num_frames, total_num_frames / num_frames).int()
        else:
            indices = torch.arange(0, total_num_frames).int()

        return indices

    def resize(
        self,
        videos: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input video aspect ratio."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")

        height, width = videos.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return super().resize(
            image=videos,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def patchify(
        self,
        videos: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int]:
        "Patchifies each video into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape

        # Check that videos have `num_frames` divisible by `temporal_patch_size`
        if pad := -num_frames % temporal_patch_size:
            repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
            videos = torch.cat((videos, repeats), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        return flatten_patches, grid_t, grid_h, grid_w

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        """
        A utility that returns number of video patches a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video.
            width (`int`):
                Width of the input video.
            videos_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the video processor.
        Returns:
            `Tuple(int, int)`: Number of placeholder tokens required and number of patches per image.
        """
        min_pixels = videos_kwargs.get("min_pixels", None) or self.size["shortest_edge"]
        max_pixels = videos_kwargs.get("max_pixels", None) or self.size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", None) or self.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.merge_size
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", None) or self.temporal_patch_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_t = num_frames // temporal_patch_size
        return grid_t * grid_h * grid_w


__all__ = ["Qwen2VLVideoProcessor"]
