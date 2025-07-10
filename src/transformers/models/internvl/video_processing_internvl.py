# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Video processor class for InternVL."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)
from ...utils.import_utils import requires
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


if is_torch_available():
    import torch

if is_vision_available():
    from ...image_utils import PILImageResampling


class InternVLVideoProcessorInitKwargs(VideosKwargs):
    initial_shift: Union[bool, float, int]


@requires(backends=("torchvision",))
class InternVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    initial_shift = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = InternVLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[InternVLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def sample_frames(
        self,
        video: "torch.Tensor",
        metadata: Optional[Union[VideoMetadata, dict]] = None,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        initial_shift: Optional[Union[bool, float, int]] = None,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoMetadata`, *optional*):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
            initial_shift (`bool`, `float` or `int`, defaults to `self.initial_shift`):
                The initial shift to apply when sampling frames. If `True`, the shift is set so that frames are sampled from the middle of the video.

        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        num_frames = num_frames if num_frames is not None else self.num_frames
        initial_shift = initial_shift if initial_shift is not None else self.initial_shift
        total_num_frames = video.shape[0]

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata is None:
                raise ValueError(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Please pass in `VideoMetadata` object or use a fixed `num_frames` per input video"
                )
            num_frames = int(total_num_frames / metadata["fps"] * fps)

        if initial_shift is True:
            initial_shift = total_num_frames / num_frames / 2

        if num_frames > total_num_frames:
            raise ValueError(
                f"Video can't be sampled. The `num_frames={num_frames}` exceeds `total_num_frames={total_num_frames}`. "
            )

        indices = torch.arange(initial_shift, total_num_frames, total_num_frames / num_frames).int()
        video = video[indices].contiguous()
        return video

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        video_metadata: Union[list[VideoMetadata], list[dict]],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        size_divisor: Optional[int],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        do_pad: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_sample_frames: Optional[bool] = None,
        fps: Optional[Union[int, float]] = None,
        num_frames: Optional[int] = None,
        initial_shift: Optional[Union[bool, float, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional["torch.Tensor"] = None,
    ) -> BatchFeature:
        if do_sample_frames:
            # Sample video frames
            videos = [
                self.sample_frames(video, metadata, fps=fps, num_frames=num_frames, initial_shift=initial_shift)
                for video, metadata in zip(videos, video_metadata)
            ]

        # We need to sample frames first before moving to device, if `do_sample_frames=True`. Otherwise
        # moving the whole video incurs high GPU mem usage for long videos
        if device is not None:
            videos = [video.to(device) for video in videos]

        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(
                    stacked_videos, size=size, size_divisor=size_divisor, interpolation=interpolation
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = torch.stack(processed_videos, dim=0) if return_tensors else processed_videos

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)


__all__ = ["InternVLVideoProcessor"]
