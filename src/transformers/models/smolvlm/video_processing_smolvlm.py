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

from typing import List, Tuple

import numpy as np
from PIL import Image

# Make sure these are imported from your library
from ...image_utils import load_video
from ...utils import is_decord_available, logging


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames " "from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"


def smolvlm_sample_indices_fn(metadata, max_frames, target_fps, skip_secs=0):
    """
    Example sampling function which:
      - Uses `max_frames` (if provided) or calculates it from `fps` and metadata.
      - Applies a basic center-skip if fewer frames than available, otherwise
        optionally skips `skip_secs` from both the start and end.
      - Uniformly samples the desired number of frames between the start and end indices.

    Args:
        max_frames (`int`):
            Maximum number of frames to sample.
        target_fps (`int`):
            Target frames to sample per second.
        metadata (`dict`):
            Contains video metadata such as "n_frames" and "video_fps".
        skip_secs (`float`, *optional*, defaults to 1.0):
            Number of seconds to skip from the start and end if the video is long enough.

    Returns:
        numpy.ndarray:
            An array of unique frame indices to sample.
    """

    total_num_frames = metadata.get("total_num_frames", 0)
    if total_num_frames <= 0:
        raise ValueError(f"Invalid total_num_frames={total_num_frames} in metadata.")

    native_fps = metadata.get("fps", 30.0) or 30.0
    duration_seconds = metadata.get("duration", 0)

    if duration_seconds <= 0:
        raise ValueError("fInvalid duration_seconds={duration_seconds} in metadata.")

    # Step 1) Estimate how many frames we'd sample at `target_fps`, fallback if target_fps <= 0
    estimated_frames = int(round(target_fps * duration_seconds))

    # Step 2) desired_frames
    desired_frames = min(estimated_frames, max_frames)
    if desired_frames < 1:
        desired_frames = 1

    # Step 3) center skip logic
    start_idx = 0
    end_idx = total_num_frames - 1

    if skip_secs > 0 and (duration_seconds - 2 * skip_secs) > (max_frames * target_fps):
        start_idx = int(skip_secs * native_fps)
        end_idx = int(total_num_frames - skip_secs * native_fps)

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total_num_frames - 1)
    if start_idx >= end_idx:
        start_idx, end_idx = 0, total_num_frames - 1

    indices = np.linspace(start_idx, end_idx, desired_frames, dtype=int)
    indices = np.unique(indices)

    return indices



def load_smolvlm_video(
    path: str,
    max_frames: int = 64,
    target_fps: float = 1.0,
    skip_secs: float = 1.0,
    backend: str = "decord"
) -> Tuple[List[Image.Image], List[str], float]:
    """
    Loads a video from `path`, replicating the old skip-secs logic by
    using the new `load_video` + a custom `sample_indices_fn`.
    
    1) We gather metadata with `get_video_details`
       (this is optional here, but often useful for logs or verifying fps).
    2) We call `load_video` with `sample_indices_fn` that
       implements skipping logic + uniform sampling.
    3) Convert the returned array (N, C, H, W) to a list of PIL images in RGB.
    4) Build timestamps (MM:SS) based on the final sampled indices.

    Returns:
        frames (List[Image.Image]): Decoded frames in RGB format.
        timestamps (List[str]): Timestamps (MM:SS) for each frame.
        duration_seconds (float): The total video duration in seconds.
    """
    if backend == "decord" and not is_decord_available():
        logger.info("Decord not available, defaulting to OpenCV.")
        backend = "opencv"

    # Wrap our skip-logic sampler in a partial, so we can pass skip_secs, etc.
    # We'll let load_video fill in (num_frames, fps, metadata, ...) automatically.
    def sample_indices_fn_func(metadata, **fn_kwargs):
        return smolvlm_sample_indices_fn(metadata, max_frames=max_frames, target_fps=target_fps,  skip_secs=skip_secs, **fn_kwargs)
    sample_indices_fn = sample_indices_fn_func
    
    video_array, metadata = load_video(
        video=path,
        backend=backend,
        sample_indices_fn=sample_indices_fn
    )

    # Construct final frames & timestamps
    # Decide if we need to do color conversion
    sampled_indices = metadata.get("frame_indices", list(range(video_array.shape[0])))
    fps = metadata.get("fps")

    frames, timestamps = [], []
    for i, frame_idx in enumerate(sampled_indices):
        # Convert to PIL.Image (RGB)
        pil_frame = Image.fromarray(frame, mode="RGB")
        frames.append(pil_frame)

        # Build timestamps
        sec = frame_idx / fps
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamps.append(f"{mm:02d}:{ss:02d}")

    return frames, timestamps, duration_seconds
