# Copyright 2026 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
"""Train-consistent native video preprocessing for Dots3-Note Omni."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import numpy as np
from PIL import Image

from ...utils import requires_backends
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


_ALIGN = 28
_MIN_FRAMES = 4
_PF_FLOOR = 128
_PF_CEIL = 1024
_FPS_CAP = 1.0
_FPS_MIN = 0.2
_FRAME_OVERHEAD = 15
_BUDGET_OVERHEAD = 2240
_INTERLEAVE_MIN_SECONDS = 1.0
_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_SAMPLES_PER_TOKEN = 1_280
_AUDIO_CHUNK_SECONDS = 30


@dataclass(frozen=True)
class Dots3NoteOmniVideoPart:
    kind: Literal["text", "image", "audio"]
    value: str | Image.Image | np.ndarray


class Dots3NoteOmniVideoProcessor(Qwen2VLVideoProcessor):
    """Video processor providing the native Dots3 timestamped image/audio expansion."""

    size = {"shortest_edge": 56 * 56, "longest_edge": (36 * 28) ** 2}
    temporal_patch_size = 1

    def preprocess_native(self, video, **kwargs) -> list[Dots3NoteOmniVideoPart]:
        return preprocess_dots3_note_video(video, **kwargs)


def _token_length(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _compute_target_size(orig_height: int, orig_width: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    if height * width > max_pixels:
        scale = math.sqrt(orig_height * orig_width / max_pixels)
        height = max(_ALIGN, math.floor(orig_height / scale / _ALIGN) * _ALIGN)
        width = max(_ALIGN, math.floor(orig_width / scale / _ALIGN) * _ALIGN)
    elif height * width < min_pixels:
        scale = math.sqrt(min_pixels / max(1, orig_height * orig_width))
        height = math.ceil(orig_height * scale / _ALIGN) * _ALIGN
        width = math.ceil(orig_width * scale / _ALIGN) * _ALIGN
        if height * width > max_pixels:
            scale = math.sqrt(height * width / max_pixels)
            height = max(_ALIGN, math.floor(height / scale / _ALIGN) * _ALIGN)
            width = max(_ALIGN, math.floor(width / scale / _ALIGN) * _ALIGN)
    return int(height), int(width)


def _real_patches_at(orig_height: int, orig_width: int, patch_cap: int) -> int:
    height, width = _compute_target_size(
        orig_height,
        orig_width,
        _PF_FLOOR * _ALIGN * _ALIGN,
        max(_PF_FLOOR, patch_cap) * _ALIGN * _ALIGN,
    )
    return (height // _ALIGN) * (width // _ALIGN)


def _frame_hard_cap(sequence_length: int) -> int:
    required = max(1, (sequence_length - _BUDGET_OVERHEAD) // (_PF_FLOOR + _FRAME_OVERHEAD))
    if required <= 1024:
        return 1024
    cap = 1
    while cap < required:
        cap <<= 1
    return cap


def _solve_degrade(
    visual_budget: int,
    duration: float,
    orig_height: int,
    orig_width: int,
    orig_fps: float,
    sequence_length: int,
) -> tuple[int, int]:
    aligned_height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    aligned_width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    original_patch_cap = (aligned_height // _ALIGN) * (aligned_width // _ALIGN)
    fps_cap = min(_FPS_CAP, max(orig_fps, 1e-6))
    patch_cap = min(_PF_CEIL, max(original_patch_cap, _PF_FLOOR))
    frame_cap = _frame_hard_cap(sequence_length)

    def usage(scale: float) -> tuple[int, int, int]:
        fps = _FPS_MIN + scale * (fps_cap - _FPS_MIN)
        candidate_patch_cap = _PF_FLOOR + scale * (patch_cap - _PF_FLOOR)
        num_frames = max(_MIN_FRAMES, min(int(round(duration * fps)), frame_cap))
        patches = _real_patches_at(orig_height, orig_width, int(round(candidate_patch_cap)))
        return num_frames * (patches + _FRAME_OVERHEAD), int(round(candidate_patch_cap)), num_frames

    if usage(1.0)[0] <= visual_budget:
        _, candidate_patch_cap, num_frames = usage(1.0)
        return num_frames, candidate_patch_cap

    floor_cost = _real_patches_at(orig_height, orig_width, _PF_FLOOR) + _FRAME_OVERHEAD
    if usage(0.0)[0] > visual_budget:
        return max(_MIN_FRAMES, min(visual_budget // floor_cost, frame_cap)), _PF_FLOOR

    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if usage(middle)[0] <= visual_budget:
            low = middle
        else:
            high = middle
    _, candidate_patch_cap, num_frames = usage(low)
    return num_frames, candidate_patch_cap


def _audio_tokens(duration: float, sample_rate: int) -> int:
    if duration <= 0:
        return 0
    total_samples = int(duration * sample_rate)
    chunk_samples = _AUDIO_CHUNK_SECONDS * sample_rate
    count = 0
    for start in range(0, total_samples, chunk_samples):
        count += math.ceil(min(chunk_samples, total_samples - start) / _AUDIO_SAMPLES_PER_TOKEN)
    return count + 2


def _decode_audio(video_bytes: bytes, sample_rate: int) -> tuple[np.ndarray | None, float]:
    requires_backends(_decode_audio, ["torchcodec"])
    from torchcodec.decoders import AudioDecoder

    try:
        samples = AudioDecoder(video_bytes, sample_rate=sample_rate, num_channels=1).get_all_samples()
    except Exception:
        return None, 0.0
    waveform = samples.data
    if waveform is None or waveform.numel() == 0:
        return None, 0.0
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    pcm = (np.clip(waveform.cpu().numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm, float(pcm.shape[0]) / sample_rate


def _open_video(video_bytes: bytes):
    requires_backends(_open_video, ["torchcodec"])
    from torchcodec.decoders import VideoDecoder

    try:
        return VideoDecoder(video_bytes, dimension_order="NHWC", num_ffmpeg_threads=1, seek_mode="approximate")
    except TypeError:
        return VideoDecoder(video_bytes, dimension_order="NHWC", num_ffmpeg_threads=1)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _decode_frames(
    video_bytes: bytes,
    visual_budget: int,
    sequence_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    decoder = _open_video(video_bytes)
    metadata = decoder.metadata
    duration = float(metadata.duration_seconds or 0)
    orig_height = int(metadata.height)
    orig_width = int(metadata.width)
    total_frames = int(getattr(metadata, "num_frames", 0) or 0)
    orig_fps = float(getattr(metadata, "average_fps", 0) or 0) or 25.0
    if duration <= 0 or orig_height <= 0 or orig_width <= 0:
        raise ValueError(f"Invalid video metadata: duration={duration}, height={orig_height}, width={orig_width}")
    if total_frames <= 0:
        total_frames = max(1, int(duration * orig_fps))

    num_frames, patch_cap = _solve_degrade(visual_budget, duration, orig_height, orig_width, orig_fps, sequence_length)
    aligned_height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    aligned_width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    original_patch_cap = (aligned_height // _ALIGN) * (aligned_width // _ALIGN)
    target_height, target_width = _compute_target_size(
        orig_height,
        orig_width,
        _PF_FLOOR * _ALIGN * _ALIGN,
        min(patch_cap, original_patch_cap) * _ALIGN * _ALIGN,
    )
    num_frames = max(_MIN_FRAMES, min(num_frames, total_frames))
    step = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
    indices = sorted({max(0, min(int(round(index * step)), total_frames - 1)) for index in range(num_frames)})

    try:
        decoded = decoder.get_frames_at(indices=indices).data
    except (IndexError, RuntimeError):
        safe_indices = [index for index in indices if index < total_frames]
        while safe_indices and safe_indices[-1] > 0:
            try:
                decoded = decoder.get_frames_at(indices=safe_indices).data
                break
            except (IndexError, RuntimeError):
                safe_indices = safe_indices[:-1]
        else:
            raise

    actual_fps = round(len(decoded) / max(duration, 1e-6), 4)
    frames = []
    for frame_number, frame in enumerate(decoded):
        array = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
        image = Image.fromarray(array)
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        # SGLang's train flattener recomputes timestamps from the sampled-frame index
        # and the rounded effective FPS instead of preserving source-frame timestamps.
        timestamp = round(frame_number / actual_fps, 3)
        frames.append((timestamp, _jpeg_roundtrip(image, jpeg_quality)))
    return frames, duration


def _prepare_decoded_frames(
    video,
    visual_budget: int,
    sequence_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    metadata = None
    if isinstance(video, tuple):
        video, metadata = video
    if hasattr(video, "detach"):
        frames = video.detach().cpu().numpy()
    elif isinstance(video, (list, tuple)):
        frames = np.stack(
            [frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame) for frame in video]
        )
    else:
        frames = np.asarray(video)
    if frames.ndim == 4 and frames.shape[1] in (3, 4) and frames.shape[-1] not in (3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise TypeError("Decoded Dots3 video must have shape (frames, height, width, channels)")
    if not np.issubdtype(frames.dtype, np.integer):
        frames = np.clip(frames * 255.0 if frames.max(initial=0) <= 1.0 else frames, 0, 255).astype(np.uint8)
    fps = float((metadata or {}).get("fps", 1.0)) if metadata else 1.0
    fps = max(fps, 1e-6)
    duration = len(frames) / fps
    orig_height, orig_width = frames.shape[1:3]
    num_frames, patch_cap = _solve_degrade(visual_budget, duration, orig_height, orig_width, fps, sequence_length)
    num_frames = min(max(1, num_frames), len(frames))
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    target_height, target_width = _compute_target_size(
        orig_height, orig_width, _PF_FLOOR * _ALIGN * _ALIGN, patch_cap * _ALIGN * _ALIGN
    )
    selected_indices = sorted(set(indices.tolist()))
    actual_fps = round(len(selected_indices) / max(duration, 1e-6), 4)
    output = []
    for frame_number, index in enumerate(selected_indices):
        image = Image.fromarray(frames[index]).convert("RGB")
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        output.append((round(frame_number / actual_fps, 3), _jpeg_roundtrip(image, jpeg_quality)))
    return output, duration


def _format_timestamp(seconds: float) -> str:
    centiseconds = int(round(max(seconds, 0.0) * 100))
    hours = centiseconds // 360_000
    minutes = (centiseconds // 6_000) % 60
    secs = (centiseconds // 100) % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds % 100:02d}"


def _group_bounds(num_frames: int, duration: float, mode: str, rng: random.Random) -> list[int]:
    if num_frames <= 1 or duration <= 0:
        return [0, num_frames]
    max_groups = min(num_frames, max(1, int(duration // _INTERLEAVE_MIN_SECONDS)))
    if mode == "whole" or max_groups <= 1:
        groups = 1
    elif mode == "eval30":
        groups = round(math.sqrt(max_groups))
    elif mode == "eval_ek":
        groups = round((max_groups - 1) / math.log(max_groups))
    elif mode == "logk":
        groups = round(math.exp(rng.uniform(0.0, math.log(max_groups))))
    else:
        raise ValueError(f"Unsupported video k_mode: {mode}")
    groups = max(1, min(max_groups, groups))
    if groups == 1:
        return [0, num_frames]
    if mode == "logk":
        cuts = sorted(rng.sample(range(1, num_frames), groups - 1))
    else:
        cuts = sorted(
            {
                round(index * num_frames / groups)
                for index in range(1, groups)
                if 0 < round(index * num_frames / groups) < num_frames
            }
        )
    return [0, *cuts, num_frames]


def _read_video_bytes(video) -> bytes | None:
    if isinstance(video, (bytes, bytearray)):
        return bytes(video)
    if isinstance(video, Path):
        return video.read_bytes()
    if not isinstance(video, str):
        return None
    if video.startswith(("http://", "https://")):
        with urlopen(video, timeout=30) as response:
            return response.read()
    if video.startswith("data:"):
        return base64.b64decode(video.split(",", 1)[1])
    path = Path(video)
    if path.is_file():
        return path.read_bytes()
    try:
        return base64.b64decode(video, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("video string must be a path, URL, data URI, or base64 payload") from error


def preprocess_dots3_note_video(
    video,
    *,
    tokenizer,
    question: str = "",
    sequence_length: int = 131_072,
    output_reserve: int | None = None,
    audio_cap: float = 1.0,
    audio_sample_rate: int = _AUDIO_SAMPLE_RATE,
    k_mode: str = "eval_ek",
    max_new_tokens: int = 0,
    jpeg_quality: int = 85,
) -> list[Dots3NoteOmniVideoPart]:
    """Expand one video into SGLang-compatible timestamped image/audio parts."""
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    configured_reserve = sequence_length // 4 if output_reserve is None else output_reserve
    effective_reserve = max(configured_reserve, max_new_tokens)
    if effective_reserve >= sequence_length:
        raise ValueError("output_reserve/max_new_tokens must leave room for video input")
    if audio_cap < 0:
        raise ValueError(f"audio_cap must be non-negative, got {audio_cap}")
    if audio_sample_rate <= 0:
        raise ValueError(f"audio_sample_rate must be positive, got {audio_sample_rate}")
    if k_mode not in {"logk", "eval30", "eval_ek", "whole"}:
        raise ValueError(f"Unsupported video k_mode: {k_mode}")

    input_length = sequence_length - effective_reserve
    video_bytes = _read_video_bytes(video)
    video_duration_hint = (
        float(_open_video(video_bytes).metadata.duration_seconds or 0) if video_bytes is not None else 0.0
    )
    pcm = None
    audio_duration = 0.0
    if video_bytes is not None and audio_cap > 0:
        pcm, audio_duration = _decode_audio(video_bytes, audio_sample_rate)

    audio_token_count = _audio_tokens(audio_duration, audio_sample_rate) if pcm is not None else 0
    precheck_frame_bound = max(1, int(audio_duration * _FPS_CAP))
    precheck_groups = min(precheck_frame_bound, max(1, int(audio_duration // _INTERLEAVE_MIN_SECONDS)))
    precheck_audio_tokens = audio_token_count + 3 * precheck_groups if pcm is not None else 0
    minimum_visual_tokens = _MIN_FRAMES * (_PF_FLOOR + _FRAME_OVERHEAD)
    if (
        audio_token_count > audio_cap * input_length
        or precheck_audio_tokens + minimum_visual_tokens + _BUDGET_OVERHEAD > input_length
    ):
        pcm = None
        audio_duration = 0.0

    frame_upper_bound = max(1, int(video_duration_hint * _FPS_CAP))
    max_groups = min(frame_upper_bound, max(1, int(audio_duration // _INTERLEAVE_MIN_SECONDS)))
    reserved_audio_tokens = audio_token_count + 3 * max_groups if pcm is not None else 0

    overhead = (
        _token_length(tokenizer, "<|system|>You are a helpful assistant.<|endofsystem|>\n")
        + 2
        + _token_length(tokenizer, "<video_0>")
        + 64
    )
    visual_budget = max(_PF_FLOOR + _FRAME_OVERHEAD, input_length - overhead - reserved_audio_tokens)
    if video_bytes is not None:
        frames, _ = _decode_frames(video_bytes, visual_budget, input_length, jpeg_quality)
    else:
        frames, _ = _prepare_decoded_frames(video, visual_budget, input_length, jpeg_quality)

    if pcm is None:
        output = []
        for timestamp, image in frames:
            output.append(Dots3NoteOmniVideoPart("text", f"<{_format_timestamp(timestamp)}>"))
            output.append(Dots3NoteOmniVideoPart("image", image))
        return output

    video_id = hashlib.sha1(video_bytes).hexdigest()
    record_key = hashlib.sha1(f"{video_id}|{question}".encode()).hexdigest()
    seed = hashlib.sha1(f"42|flatten|{record_key}".encode()).hexdigest()
    rng = random.Random(int(seed[:8], 16))
    bounds = _group_bounds(len(frames), audio_duration, k_mode, rng)
    output = []
    for group in range(len(bounds) - 1):
        start, end = bounds[group], bounds[group + 1]
        if end <= start:
            continue
        start_time = 0.0 if group == 0 else frames[start][0]
        end_time = audio_duration if group == len(bounds) - 2 else frames[end][0]
        if end_time <= start_time:
            end_time = start_time + audio_duration / max(1, len(bounds) - 1)
        for timestamp, image in frames[start:end]:
            output.append(Dots3NoteOmniVideoPart("text", f"<{_format_timestamp(timestamp)}>"))
            output.append(Dots3NoteOmniVideoPart("image", image))
        sample_start = max(0, int(round(start_time * audio_sample_rate)))
        sample_end = min(len(pcm), int(round(end_time * audio_sample_rate)))
        if sample_end > sample_start:
            waveform = np.ascontiguousarray(pcm[sample_start:sample_end].astype(np.float32) / 32768.0)
            output.append(Dots3NoteOmniVideoPart("audio", waveform))
    return output


__all__ = ["Dots3NoteOmniVideoProcessor"]
