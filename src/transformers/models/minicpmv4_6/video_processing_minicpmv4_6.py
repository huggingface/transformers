# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Video processor for MiniCPM-V 4.6.

MiniCPM-V treats video as a sequence of images: frames are extracted and
optionally sub-second frames are stacked into composite images via
:func:`concat_images`.  The resulting PIL images are then processed by
:class:`MiniCPMV4_6ImageProcessor` through the normal image pipeline.

This module provides :class:`MiniCPMV4_6VideoProcessor` (inherits from
:class:`BaseVideoProcessor`) which encapsulates all video-specific logic
(frame sampling, stacking, ffmpeg / decord backends) that was previously
inlined in ``processing_minicpmv4_6.py``.
"""

import math

from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torch_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class MiniCPMV4_6VideoProcessorKwargs(VideosKwargs, total=False):
    r"""
    max_frames (`int`, *optional*, defaults to 128):
        Maximum number of main frames to sample per video.
    stack_frames (`int`, *optional*, defaults to 1):
        Sub-frames per second to stack.  ``1`` disables stacking.
    use_ffmpeg (`bool`, *optional*, defaults to `False`):
        Use ffmpeg/ffprobe CLI instead of decord for frame extraction.
    """

    max_frames: int
    stack_frames: int
    use_ffmpeg: bool


# ---------------------------------------------------------------------------
# Frame-level helpers
# ---------------------------------------------------------------------------


def _uniform_sample(lst, n):
    """Uniformly sample *n* items from *lst*."""
    if len(lst) <= n:
        return lst
    import numpy as np

    idxs = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in idxs]


def concat_images(images, bg_color=(255, 255, 255), cell_size=None, line_color=(0, 0, 0), line_width=6):
    """Concatenate PIL images into a grid layout.

    Picks the most square-like (rows, cols) arrangement automatically.
    Each cell is sized to the largest input image (or *cell_size* if given),
    with images centered and rescaled to fit.
    """
    import numpy as np
    from PIL import Image

    n = len(images)
    if n == 0:
        raise ValueError("images is empty")
    if n == 1:
        return images[0]

    if cell_size is None:
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
    else:
        cell_w, cell_h = cell_size

    def _canvas_ratio(r, c):
        cW = c * cell_w + (c - 1) * line_width
        cH = r * cell_h + (r - 1) * line_width
        return cW / max(1, cH)

    def _pick_best(candidates):
        ratios = [abs(_canvas_ratio(r, c) - 1.0) for r, c in candidates]
        return candidates[int(np.argmin(ratios))]

    if n == 4:
        rows, cols = 2, 2
    elif n == 3:
        rows, cols = _pick_best([(1, 3), (3, 1)])
    elif n == 2:
        candidates = [(1, 2), (2, 1)]
        ratios = [abs(_canvas_ratio(r, c) - 1.0) for r, c in candidates]
        if ratios[0] == ratios[1]:
            avg_ar = np.mean([im.width / max(1, im.height) for im in images])
            rows, cols = (1, 2) if avg_ar >= 1.0 else (2, 1)
        else:
            rows, cols = candidates[int(np.argmin(ratios))]
    else:
        rows, cols = 1, n

    W = cols * cell_w + (cols - 1) * line_width
    H = rows * cell_h + (rows - 1) * line_width
    canvas = Image.new("RGB", (W, H), line_color)

    for i, im in enumerate(images[: rows * cols]):
        r, c = divmod(i, cols)
        s = min(cell_w / im.width, cell_h / im.height)
        nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
        try:
            im_r = im.resize((nw, nh), Image.Resampling.BICUBIC)
        except AttributeError:
            im_r = im.resize((nw, nh), Image.BICUBIC)
        bg = Image.new("RGB", (cell_w, cell_h), bg_color)
        bg.paste(im_r, ((cell_w - nw) // 2, (cell_h - nh) // 2))
        canvas.paste(bg, (c * (cell_w + line_width), r * (cell_h + line_width)))

    return canvas


def _group_stacked_by_second(sub_frames, sub_ts, num_seconds):
    """Group sub-frames by second and concat each group into a composite."""
    stacked = []
    cursor = 0
    for sec in range(num_seconds):
        group = []
        while cursor < len(sub_ts) and sub_ts[cursor] < sec + 1:
            group.append(sub_frames[cursor])
            cursor += 1
        stacked.append(concat_images(group) if group else None)
    return stacked


# ---------------------------------------------------------------------------
# Frame extraction (decord / ffmpeg backends)
# ---------------------------------------------------------------------------


def extract_frames(video_source, max_frames=128, stack_frames=1, use_ffmpeg=False):
    """Extract video frames, optionally with sub-second stacked frames.

    Returns ``(main_frames, stacked_frames)``.

    *main_frames*: list of ``PIL.Image`` at ~1 fps (uniformly sampled to
    *max_frames* for long videos).

    *stacked_frames*: ``None`` when *stack_frames* <= 1.  Otherwise a list
    of length ``num_seconds`` where each element is a composite
    ``PIL.Image`` (sub-frames concatenated via :func:`concat_images`) or
    ``None`` if no sub-frames exist for that second.

    Supports local paths and ``http(s)://`` URLs.  Set *use_ffmpeg* to use
    the ffmpeg/ffprobe CLI instead of ``decord`` (the default).
    """
    import os
    import tempfile

    temp_video = None
    if isinstance(video_source, str) and video_source.startswith(("http://", "https://")):
        import requests

        resp = requests.get(video_source, timeout=120)
        resp.raise_for_status()
        fd, temp_video = tempfile.mkstemp(suffix=".mp4")
        os.write(fd, resp.content)
        os.close(fd)
        video_source = temp_video

    try:
        if use_ffmpeg:
            return _extract_frames_ffmpeg(video_source, max_frames, stack_frames)
        else:
            return _extract_frames_decord(video_source, max_frames, stack_frames)
    finally:
        if temp_video is not None:
            os.unlink(temp_video)


def _extract_frames_ffmpeg(video_source, max_frames, stack_frames):
    import os
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    probe_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_source,
        ],
        capture_output=True,
        text=True,
    )
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_source}: {probe_result.stderr.strip()}")
    dur = float(probe_result.stdout.strip())
    num_seconds = math.ceil(dur)

    main_dir = tempfile.mkdtemp()
    try:
        extract_fps = 10 if dur > max_frames else 1
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_source,
                "-vf",
                f"fps={extract_fps}",
                os.path.join(main_dir, "frame_%06d.jpg"),
            ],
            capture_output=True,
            check=True,
        )
        frame_files = sorted(f for f in os.listdir(main_dir) if f.endswith(".jpg"))
        if dur > max_frames:
            sampled_indices = _uniform_sample(list(range(len(frame_files))), max_frames)
            frame_files = [frame_files[i] for i in sampled_indices]
        main_frames = [Image.open(os.path.join(main_dir, f)).convert("RGB") for f in frame_files]
    finally:
        shutil.rmtree(main_dir, ignore_errors=True)

    stacked = None
    if stack_frames > 1:
        stack_dir = tempfile.mkdtemp()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_source,
                    "-vf",
                    f"fps={stack_frames}",
                    os.path.join(stack_dir, "frame_%06d.jpg"),
                ],
                capture_output=True,
                check=True,
            )
            stack_files = sorted(f for f in os.listdir(stack_dir) if f.endswith(".jpg"))
            valid_indices, sub_ts = [], []
            for i in range(len(stack_files)):
                if i % stack_frames != 0:
                    ts = i / stack_frames
                    if ts < dur:
                        sub_ts.append(ts)
                        valid_indices.append(i)
            max_stack = max_frames * (stack_frames - 1)
            if len(valid_indices) > max_stack:
                sampled = _uniform_sample(list(zip(valid_indices, sub_ts)), max_stack)
                valid_indices = [x[0] for x in sampled]
                sub_ts = [x[1] for x in sampled]
            sub_frames = [Image.open(os.path.join(stack_dir, stack_files[i])).convert("RGB") for i in valid_indices]
            stacked = _group_stacked_by_second(sub_frames, sub_ts, num_seconds)
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)

    return main_frames, stacked


def _extract_frames_decord(video_source, max_frames, stack_frames):
    from PIL import Image

    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError(
            "Video processing requires `decord`. Install with: pip install decord  (or use use_ffmpeg=True)"
        )
    vr = VideoReader(str(video_source), ctx=cpu(0))
    total, avg_fps = len(vr), vr.get_avg_fps()
    dur = total / avg_fps
    num_seconds = math.ceil(dur)

    is_long = dur > max_frames
    if is_long:
        timestamps = [round(i * 0.1, 1) for i in range(int(dur / 0.1))]
        indices = [min(int(ts * avg_fps), total - 1) for ts in timestamps]
        indices = _uniform_sample(indices, max_frames)
    else:
        indices = [int(i * avg_fps) for i in range(num_seconds)]
    main_frames = [Image.fromarray(f.astype("uint8")).convert("RGB") for f in vr.get_batch(indices).asnumpy()]

    stacked = None
    if stack_frames > 1:
        sub_ts = []
        for sec in range(num_seconds):
            for j in range(1, stack_frames):
                ts = sec + j / stack_frames
                if ts < dur:
                    sub_ts.append(ts)
        sub_indices = [min(int(ts * avg_fps), total - 1) for ts in sub_ts]
        max_stack = max_frames * (stack_frames - 1)
        if len(sub_indices) > max_stack:
            idx = _uniform_sample(list(range(len(sub_indices))), max_stack)
            sub_indices = [sub_indices[j] for j in idx]
            sub_ts = [sub_ts[j] for j in idx]
        sub_frames = [Image.fromarray(f.astype("uint8")).convert("RGB") for f in vr.get_batch(sub_indices).asnumpy()]
        stacked = _group_stacked_by_second(sub_frames, sub_ts, num_seconds)

    return main_frames, stacked


# ---------------------------------------------------------------------------
# VideoProcessor class
# ---------------------------------------------------------------------------


@add_start_docstrings(
    "Constructs a MiniCPM-V 4.6 video processor.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        max_frames (`int`, *optional*, defaults to 128):
            Maximum number of main frames to sample per video.
        stack_frames (`int`, *optional*, defaults to 1):
            Sub-frames per second to stack into composite images.  ``1`` means
            no stacking.
        use_ffmpeg (`bool`, *optional*, defaults to `False`):
            Use ffmpeg/ffprobe CLI instead of decord for frame extraction.
    """,
)
class MiniCPMV4_6VideoProcessor(BaseVideoProcessor):
    """MiniCPM-V 4.6 video processor.

    Unlike models that process video as 3-D temporal tensors, MiniCPM-V
    converts each video into a sequence of PIL images (with optional
    sub-second frame stacking) that are then handled by the image processor.

    The main entry point for the Processor is :meth:`extract_frames`, which
    is called during ``apply_chat_template`` to expand ``{"type": "video"}``
    content blocks into ``{"type": "image"}`` blocks.
    """

    resample = PILImageResampling.BICUBIC
    do_resize = False
    do_rescale = False
    do_normalize = False
    do_convert_rgb = True
    do_sample_frames = False
    max_frames = 128
    stack_frames = 1
    use_ffmpeg = False
    valid_kwargs = MiniCPMV4_6VideoProcessorKwargs
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_6VideoProcessorKwargs]):
        super().__init__(**kwargs)

    def extract_frames(
        self,
        video_source,
        max_frames: int | None = None,
        stack_frames: int | None = None,
        use_ffmpeg: bool | None = None,
    ):
        """Extract frames from a video, returning PIL images.

        This is the primary interface called by
        :class:`MiniCPMV4_6Processor` during ``apply_chat_template``.

        Args:
            video_source: Local file path or ``http(s)://`` URL.
            max_frames: Override ``self.max_frames``.
            stack_frames: Override ``self.stack_frames``.
            use_ffmpeg: Override ``self.use_ffmpeg``.

        Returns:
            ``(main_frames, stacked_frames)`` where *main_frames* is a list
            of PIL images and *stacked_frames* is ``None`` or a list of
            composite PIL images (one per second of video).
        """
        max_frames = max_frames if max_frames is not None else self.max_frames
        stack_frames = stack_frames if stack_frames is not None else self.stack_frames
        use_ffmpeg = use_ffmpeg if use_ffmpeg is not None else self.use_ffmpeg
        return extract_frames(video_source, max_frames=max_frames, stack_frames=stack_frames, use_ffmpeg=use_ffmpeg)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool,
        do_resize: bool,
        size,
        resample,
        do_center_crop: bool,
        crop_size,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Pass-through: actual image transforms are done by the image processor.

        When videos are processed through the standard ``preprocess`` pipeline
        (rather than via ``extract_frames``), this simply bundles the decoded
        frames into a :class:`BatchFeature` without additional transforms,
        since MiniCPM-V delegates all spatial processing to the image processor.
        """
        import torch

        all_frames = []
        for video in videos:
            if video.dim() == 4:
                all_frames.extend(list(video))
            else:
                all_frames.append(video)
        return BatchFeature(
            data={"pixel_values": torch.stack(all_frames) if all_frames else torch.empty(0)},
            tensor_type=return_tensors,
        )


__all__ = ["MiniCPMV4_6VideoProcessor"]
