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
from ...image_utils import get_video_details, load_video
from ...utils import is_decord_available, logging


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames "
    "from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"

# Some backends produce BGR (OpenCV), others produce RGB (Decord, PyAV, etc.)
BACKEND_CHANNEL_ORDERS = {
    "opencv": "bgr",
    "decord": "rgb",
    "pyav": "rgb",
    "torchvision": "rgb",
}

def load_smolvlm_video(
    path: str,
    max_frames: int = 64,
    sampling_fps: float = 1.0,
    skip_secs: float = 1.0,
    backend: str = "decord",
) -> Tuple[List[Image.Image], List[str], float]:
    """
    Loads a video from `path` by first gathering metadata via `get_video_details`
    and then computing frame indices based on the old skip-secs logic.
    Finally, it calls the updated `load_video` (which returns a numpy array)
    and converts that array into a list of PIL images (RGB).

    Returns:
        frames (List[Image.Image]): Decoded frames in RGB format.
        timestamps (List[str]): Timestamps (MM:SS) for each frame.
        duration_seconds (float): The total video duration in seconds.
    """
    if backend == "decord" and not is_decord_available():
        logger.info("Decord not available, defaulting to OpenCV.")
        backend = "opencv"

    # 1) Gather metadata
    n_frames, fps, duration_seconds = get_video_details(path, backend=backend)
    if fps <= 0:
        fps = 30.0  # fallback if needed

    # 2) Estimate how many frames we'd sample at `sampling_fps`
    estimated_frames = int(round(sampling_fps * duration_seconds)) if sampling_fps > 0 else max_frames
    desired_frames = min(estimated_frames, max_frames)
    if desired_frames < 1:
        desired_frames = 1

    # 3) Compute skip logic
    start_idx = 0
    end_idx = n_frames - 1

    if desired_frames < max_frames:
        leftover = n_frames - desired_frames
        start_idx = leftover // 2
        end_idx = n_frames - (leftover - start_idx)
    elif skip_secs > 0 and (duration_seconds - 2 * skip_secs) > (max_frames * sampling_fps):
        start_idx = int(skip_secs * fps)
        end_idx = int(n_frames - skip_secs * fps)

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, n_frames - 1)
    if start_idx >= end_idx:
        start_idx, end_idx = 0, n_frames - 1

    frames_idx = np.linspace(start_idx, end_idx, desired_frames, dtype=int)
    frames_idx = np.unique(frames_idx).tolist()

    # 4) Decode frames with the updated load_video (returns a numpy array: (N, H, W, C))
    frames_array = load_video(
        video=path,
        num_frames=None,
        fps=None,
        frame_indicies=frames_idx,
        backend=backend,
    )

    # 5) Convert frames to PIL (RGB) + build timestamps
    channel_order = BACKEND_CHANNEL_ORDERS.get(backend, "rgb")
    frames, timestamps = [], []
    for idx, frame_np in zip(frames_idx, frames_array):
        if channel_order == "bgr":
            # Convert BGR -> RGB if needed
            frame_np = frame_np[..., ::-1]
        pil_frame = Image.fromarray(frame_np, mode="RGB")
        frames.append(pil_frame)

        sec = idx / fps
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamps.append(f"{mm:02d}:{ss:02d}")

    return frames, timestamps, duration_seconds
