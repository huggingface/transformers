# coding=utf-8
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
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from ...feature_extraction_utils import BatchFeature
from ...video_processing_utils import BaseVideoProcessor
from .image_processing_onyx import _grid_size


try:
    import torchcodec
except Exception:
    torchcodec = None


class OnyxVideoProcessor(BaseVideoProcessor):
    """Onyx video preprocessing behind the standard HF ``AutoVideoProcessor`` API.

    Wraps torchcodec decoding (training-faithful), uniform frame sampling to a
    whole multiple of ``patch_temporal``, real per-group PTS timestamps, and
    ``patch_temporal`` frame-grouping (frames cat on the channel axis ->
    ``[patch_temporal * 3, H, W]``; the encoder detects video by channel count).

    Overrides ``preprocess`` instead of the BaseVideoProcessor fast pipeline
    because Onyx consumes a LIST of variable-size group tensors and needs real
    per-group PTS, neither of which the stacked fast path models.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 14,
        downsample_factor: int = 2,
        patch_temporal: int = 2,
        max_video_frame_tokens: int = 144,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        video_num_frames: int = 96,
        video_sampling_fps: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.patch_temporal = patch_temporal
        self.max_video_frame_tokens = max_video_frame_tokens
        self.image_mean = image_mean
        self.image_std = image_std
        self.video_num_frames = video_num_frames
        self.video_sampling_fps = video_sampling_fps

    def _to_norm_tensor(self, image: Image.Image) -> torch.Tensor:
        return T.functional.normalize(
            T.functional.to_tensor(image),
            [self.image_mean] * 3,
            [self.image_std] * 3,
        )

    def decode_video(self, video_path: str) -> tuple[list[Image.Image], list[float]]:
        """Sample frames + per-group PTS with torchcodec (the training decode path).

        ``timestamps[g]`` is the ACTUAL decoded PTS of the first frame in temporal
        group ``g``; ``len(frames)`` is a whole multiple of ``patch_temporal``.
        """
        if torchcodec is None:
            raise RuntimeError(
                "torchcodec is required for video decoding (it matches the training decode path)."
            )
        pt = self.patch_temporal
        reader = torchcodec.decoders.VideoDecoder(video_path)
        total = len(reader)
        assert reader.metadata.average_fps is not None, f"Video has no FPS metadata: {video_path}"
        fps = reader.metadata.average_fps
        assert self.video_sampling_fps and self.video_sampling_fps > 0, (
            f"video_sampling_fps must be positive, got {self.video_sampling_fps}"
        )
        n = min(int(total * self.video_sampling_fps / fps), self.video_num_frames, total)
        n = max(pt, (n // pt) * pt)
        n = min(n, total)
        if n < pt:
            raise ValueError(
                f"Video has only {total} decodable frame(s) but needs at least "
                f"{pt} (one temporal patch): {video_path}"
            )
        indices = torch.linspace(0, total - 1, n).long().tolist()
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for j, i in enumerate(indices):
            fr = reader[i]
            frames.append(Image.fromarray(fr.data.permute(1, 2, 0).numpy()).convert("RGB"))
            if j % pt == 0:
                pts = getattr(fr, "pts_seconds", None)
                timestamps.append(float(pts) if pts is not None else i / fps)
        return frames, timestamps

    def compute_video_frame_size(self, img_w: int, img_h: int) -> tuple[int, int, int]:
        ph = self.patch_size * self.downsample_factor
        return _grid_size(img_w, img_h, ph, self.max_video_frame_tokens)

    def _group_frames(self, frames: list[Image.Image]) -> tuple[list[torch.Tensor], int, int]:
        pt = self.patch_temporal
        if len(frames) < pt or len(frames) % pt != 0:
            raise ValueError(
                f"video frame count {len(frames)} must be a positive multiple of patch_temporal={pt}"
            )
        first = frames[0].convert("RGB")
        target_h, target_w, n_tokens = self.compute_video_frame_size(first.width, first.height)
        groups: list[torch.Tensor] = []
        for i in range(0, len(frames), pt):
            grp = [
                self._to_norm_tensor(
                    frames[i + j].convert("RGB").resize((target_w, target_h), Image.LANCZOS)
                )
                for j in range(pt)
            ]
            groups.append(torch.cat(grp, dim=0))
        return groups, len(groups), n_tokens

    @staticmethod
    def _normalize_videos(videos) -> list:
        if isinstance(videos, (str, Path)):
            return [videos]
        if videos and not isinstance(videos[0], (list, tuple, str, Path)):
            return [videos]
        return list(videos or [])

    def preprocess_one(
        self,
        video: str | Path | list[Image.Image],
        timestamps: list[float] | None = None,
    ) -> tuple[list[torch.Tensor], int, int, list[float]]:
        """Decode (if a path) + group ONE video.

        Returns ``(group_tensors, n_groups, tokens_per_group, group_timestamps)``.
        """
        if isinstance(video, (str, Path)):
            frames, ts = self.decode_video(str(video))
        else:
            frames = [f.convert("RGB") for f in video]
            ts = list(timestamps) if timestamps is not None else []
        groups, n_groups, tokens_per_group = self._group_frames(frames)
        return groups, n_groups, tokens_per_group, ts

    def preprocess(
        self,
        videos,
        video_timestamps: list[list[float]] | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """Preprocess video file path(s) or pre-decoded frame list(s)."""
        videos = self._normalize_videos(videos)
        pixel_values: list[torch.Tensor] = []
        num_groups: list[int] = []
        tokens_per_group: list[int] = []
        out_ts: list[list[float]] = []
        for idx, v in enumerate(videos):
            ts_in = video_timestamps[idx] if video_timestamps else None
            groups, ng, tpg, ts = self.preprocess_one(v, ts_in)
            pixel_values += groups
            num_groups.append(ng)
            tokens_per_group.append(tpg)
            out_ts.append(ts)
        batch = BatchFeature(
            data={
                "video_num_groups": num_groups,
                "video_tokens_per_group": tokens_per_group,
                "video_timestamps": out_ts,
            },
            tensor_type=None,
        )
        batch["pixel_values"] = pixel_values
        return batch


__all__ = ["OnyxVideoProcessor"]
