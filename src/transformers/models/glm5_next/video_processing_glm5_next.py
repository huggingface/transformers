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
"""Video processor class for GLM-5-Next.

GLM-5-Next reuses the GLMga image processor and the GLMga video
``_preprocess`` (resize / patchify / grid) unchanged -- only the frame
sampling differs from GLMga, so this class inherits :class:`GlmgaVideoProcessor`
and overrides :meth:`sample_frames` to match the GLM-5-Next reference.
"""

import math

import numpy as np

from ...video_utils import VideoMetadata
from ..glmga.video_processing_glmga import GlmgaVideoProcessor


class Glm5NextVideoProcessor(GlmgaVideoProcessor):
    max_duration = 2400
    dynamic_fps_thresholds = [(30, 3), (300, 1), (2400, 0.5)]

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ):
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was provided which is required when sampling in Glm5Next. "
                "Please pass in `VideoMetadata` object or set `do_sample_frames=False`"
            )

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1
        effective_duration = min(duration, self.max_duration)

        target_fps = fps
        if target_fps is None:
            target_fps = next(
                candidate_fps
                for boundary, candidate_fps in self.dynamic_fps_thresholds
                if effective_duration <= boundary
            )
        target_fps *= self.temporal_patch_size

        extract_t = int(effective_duration * target_fps)
        extract_t = min(extract_t, self.max_frames)

        duration_per_frame = 1 / metadata.fps
        timestamps = [i * duration_per_frame for i in range(total_frames)]
        max_second = int(duration)

        if total_frames < extract_t:
            frame_indices = [math.floor(_i * total_frames / extract_t) for _i in range(extract_t)]
        else:
            frame_indices = []
            current_second = 0
            inv_fps = 1 / target_fps
            for frame_index in range(total_frames):
                if timestamps[frame_index] >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_second:
                        break

        if len(frame_indices) < extract_t:
            if len(frame_indices) == 0:
                start, end = 0, max(total_frames - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        return np.array(uniq)


__all__ = ["Glm5NextVideoProcessor"]
