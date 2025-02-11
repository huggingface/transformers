# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ...utils import logging, is_cv2_available, is_decord_available
from typing import TYPE_CHECKING, Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import numpy as np

logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
# DEFAULT_VIDEO_INTRO = "Here are some frames sampled from a video:"
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames "
    "from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"


if is_decord_available():
    import decord
    decord.bridge.set_bridge("torch")
    from decord import VideoReader
    
elif is_cv2_available():
    import cv2
    
else:
    logger.warn("Both cv2 and decord are not available. Video loading not suppo")


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")
    
def is_str(val) -> bool:
    return isinstance(val, str)

def load_video(
    path: str,
    max_frames: int = 64,
    target_fps: float = 1.0,
    skip_secs: float = 1.0
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads a video from `path` using decord, sampling up to `max_frames` frames.
    After deduplicating indices (e.g., to handle rounding collisions), each frame
    is decoded into a PIL Image (in RGB mode). Timestamps are generated in "MM:SS" format
    based on the frame index over `native_fps`.

    Args:
        path (str): Path to the video file (e.g., MP4).
        max_frames (int): Hard cap on how many frames we ever pick in total.
        target_fps (float): Target approximate sampling rate in frames per second.
        skip_secs (float): Number of seconds to skip at the beginning and end if 
            the video is sufficiently long ((duration - 2*skip_secs) > max_frames * target_fps).
    
    Returns:
        Tuple[List[Image.Image], List[str]]:
          - A list of PIL.Image objects corresponding to each selected frame.
          - A list of parallel timestamps ("MM:SS" strings), one per selected frame.
    """
    try:
        vr = VideoReader(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open video '{path}': {e}")

    total_frames = len(vr)
    if total_frames == 0:
        raise RuntimeError(f"Video '{path}' has 0 frames.")

    # Fallback to 30 if native_fps is None or zero
    native_fps = vr.get_avg_fps() or 30.0
    duration_seconds = total_frames / native_fps

    # Estimate how many frames we'd get if we sample at `target_fps`.
    estimated_frames = int(round(target_fps * duration_seconds)) if target_fps > 0 else max_frames
    desired_frames = min(estimated_frames, max_frames)
    if desired_frames < 1:
        desired_frames = 1

    start_idx = 0
    end_idx = total_frames - 1

    # Centered skip if we want fewer frames than max_frames
    if desired_frames < max_frames:
        leftover = total_frames - desired_frames
        start_idx = leftover // 2
        end_idx = total_frames - (leftover - start_idx)
    # Otherwise, if video is long enough, skip a bit from start and end
    elif skip_secs > 0 and (duration_seconds - 2 * skip_secs) > (max_frames * target_fps):
        start_idx = int(skip_secs * native_fps)
        end_idx = int(total_frames - skip_secs * native_fps)

    # Ensure valid start / end
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, total_frames - 1)
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = total_frames - 1

    # Uniformly sample the desired number of frames from [start_idx..end_idx]
    frames_idx = np.linspace(start_idx, end_idx, desired_frames, dtype=int)
    frames_idx = np.unique(frames_idx).tolist()

    # Read frames from decord
    try:
        frames_tensor = vr.get_batch(frames_idx).cpu().numpy()  # (N, H, W, C)
    except Exception as e:
        raise RuntimeError(f"Failed to read frames from '{path}': {e}")

    # Convert to PIL Images
    frames_out = [Image.fromarray(arr).convert("RGB") for arr in frames_tensor]

    # Build timestamps (MM:SS) for each selected frame index
    timestamps = []
    for idx in frames_idx:
        sec = idx / native_fps
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamps.append(f"{mm:02d}:{ss:02d}")

    return frames_out, timestamps, duration_seconds



def load_video_cv2(
    path: str,
    max_frames: int = 64,
    target_fps: float = 1.0,
    skip_secs: float = 1.0
) -> Tuple[List[Image.Image], List[str], float]:
    """
    Loads a video from `path` using OpenCV, sampling up to `max_frames` frames
    at approximately `target_fps`. Returns:
      - A list of PIL.Image objects (RGB) for each frame,
      - A list of "MM:SS" timestamps per frame,
      - The video duration in seconds.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        raise RuntimeError(f"Video '{path}' has 0 frames.")

    duration_seconds = total_frames / native_fps
    estimated_frames = int(round(target_fps * duration_seconds)) if target_fps > 0 else max_frames
    desired_frames = min(estimated_frames, max_frames)
    if desired_frames < 1:
        desired_frames = 1

    start_idx = 0
    end_idx = total_frames - 1

    # If we want fewer than max_frames, center the skip
    if desired_frames < max_frames:
        leftover = total_frames - desired_frames
        start_idx = leftover // 2
        end_idx = total_frames - (leftover - start_idx)
    # Otherwise skip some from start/end if feasible
    elif skip_secs > 0 and (duration_seconds - 2 * skip_secs) > (max_frames * target_fps):
        start_idx = int(skip_secs * native_fps)
        end_idx = int(total_frames - skip_secs * native_fps)

    start_idx = max(0, start_idx)
    end_idx = min(total_frames - 1, end_idx)
    if start_idx >= end_idx:
        start_idx, end_idx = 0, total_frames - 1

    frames_idx = np.linspace(start_idx, end_idx, desired_frames, dtype=int)
    frames_idx = np.unique(frames_idx).tolist()

    frames_out = []
    timestamps = []

    for idx in frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_out.append(Image.fromarray(frame_rgb).convert("RGB"))

        sec = idx / native_fps
        mm = int(sec // 60)
        ss = int(sec % 60)
        timestamps.append(f"{mm:02d}:{ss:02d}")

    cap.release()
    return frames_out, timestamps, duration_seconds
    

def load_video_from_disk_or_url(path_or_url: str, sampling_fps: int = 1, max_frames: int = 48):
    """
    Minimal example of loading a video or frames from a URL/local path.
    Returns: (frames, timestamps, duration_seconds).
    This can be replaced by a more robust version with decord or ffmpeg, etc.
    """
    if is_url(path_or_url):
        ## load video
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "temp_video.mp4")
            
            # e.g. use requests
            resp = requests.get(path_or_url, stream=True)
            resp.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 2) Actually load frames from the local temp file
            if is_decord_available():
                frames, timestamps, duration_seconds = load_video(
                    filename, max_frames=max_frames, target_fps=sampling_fps
                )
            elif is_cv2_available:
                frames, timestamps, duration_seconds = load_video_cv2(
                    filename, max_frames=max_frames, target_fps=sampling_fps
                )
            else:
                raise RuntimeError("No valid video runtime. please install either opencv or decord")
    
    else:
        if is_decord_available():
            frames, timestamps, duration_seconds = load_video(
                path_or_url, max_frames=max_frames, target_fps=sampling_fps
            )
        elif is_cv2_available:
            frames, timestamps, duration_seconds = load_video_cv2(
                path_or_url, max_frames=max_frames, target_fps=sampling_fps
            )
        else:
            raise RuntimeError("No valid video runtime. please install either opencv or decord")
            
    return frames, timestamps, duration_seconds