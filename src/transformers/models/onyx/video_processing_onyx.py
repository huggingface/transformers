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


import torch
import torchvision.transforms.v2.functional as tvF

from ...feature_extraction_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    get_image_size,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from .image_processing_onyx import get_aspect_ratio_preserving_size


class OnyxVideoProcessorInitKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    TODO:
    """

    patch_size: int
    temporal_patch_size: int
    max_video_frame_tokens: int
    downsample_factor: int


@auto_docstring
class OnyxVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 14
    temporal_patch_size = 2
    downsample_factor = 2
    max_video_frame_tokens = 144
    num_frames = 96
    fps = 2.0
    do_sample_frames = True

    valid_kwargs = OnyxVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[OnyxVideoProcessorInitKwargs]):
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

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: int | None = None,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            temporal_patch_size (`int`, *optional*):
                The temporal patch size of the vision encoder. Number of sampled frames will be rounded to be divisible by frame factor.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.

        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        total_num_frames = metadata.total_num_frames
        num_frames = min(int(total_num_frames * fps / metadata.fps), num_frames, total_num_frames)
        num_frames = max(temporal_patch_size, (num_frames // temporal_patch_size) * temporal_patch_size)
        num_frames = min(num_frames, total_num_frames)
        indices = torch.linspace(0, total_num_frames - 1, num_frames).long()
        return indices

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        do_convert_rgb: bool,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int,
        temporal_patch_size: int,
        max_video_frame_tokens: int,
        downsample_factor: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.aspect_ratio_preserving_resize(
                    image=stacked_videos,
                    patch_size=patch_size * downsample_factor,
                    max_tokens=max_video_frame_tokens,
                    resample=resample,
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

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            T = patches.shape[1]
            if pad := -T % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)

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
            patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                temporal_patch_size * channel * patch_size * patch_size,
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


__all__ = ["OnyxVideoProcessor"]
