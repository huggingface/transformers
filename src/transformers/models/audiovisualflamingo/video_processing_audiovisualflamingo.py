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
"""Video processor class for AudioVisualFlamingo."""

import numpy as np

from ...image_utils import PILImageResampling
from ...utils import auto_docstring
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata


@auto_docstring
class AudioVisualFlamingoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True
    num_frames = 128

    def sample_frames(
        self, metadata: VideoMetadata, num_frames: int | None = None, fps: float | None = None, **kwargs
    ) -> np.ndarray:
        # As in training, `num_frames` frames are sampled evenly from the first to the last one, repeating frames of
        # videos shorter than `num_frames`. A fixed number of frames lets videos of different lengths be batched.
        if fps is not None:
            raise ValueError("AudioVisualFlamingo samples a fixed number of frames: use `num_frames` instead of `fps`")
        num_frames = num_frames or self.num_frames
        return np.round(np.linspace(0, metadata.total_num_frames - 1, num_frames)).astype(int)


__all__ = ["AudioVisualFlamingoVideoProcessor"]
