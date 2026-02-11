# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling
from ...processing_utils import Unpack, VideosKwargs
from ...video_processing_utils import BaseVideoProcessor, VideoMetadata
from ...video_utils import VideoInput


class PeVideoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        if num_frames:
            total_frames = metadata.total_num_frames
            num_frames = num_frames if num_frames is not None else self.num_frames
            assert num_frames is not None, "`num_frames` must be specified if `fixed_len_video == True`"
            frame_idxs = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
            return torch.tensor(frame_idxs)
        else:
            return super().sample_frames(metadata, num_frames, fps, **kwargs)

    def _preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        # Always set `return_tensors` to `None` since it won't pad variable length videos
        # We'll handle this after we call the parent' method
        return_tensors = kwargs.pop("return_tensors", None)
        result = super()._preprocess(videos, **kwargs)
        pixels = result.pixel_values_videos
        data = {"pixel_values_videos": pixels}
        if return_tensors:
            lengths = torch.tensor([video.size(0) for video in pixels])
            pixels = torch.nn.utils.rnn.pad_sequence(pixels, batch_first=True, padding_value=0.0)
            data["pixel_values_videos"] = pixels
            if lengths.unique().size(0) > 1:
                mask = torch.arange(lengths.max())[None] < lengths[:, None]
                data["padding_mask_videos"] = mask
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["PeVideoVideoProcessor"]
