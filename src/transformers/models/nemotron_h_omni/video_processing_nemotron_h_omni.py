# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Video processor class for NemotronH Omni."""

import math

from ...image_processing_base import BatchFeature
from ...processing_utils import Unpack, VideosKwargs
from ...utils import auto_docstring, is_torch_available
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos


if is_torch_available():
    import torch


class NemotronH_Omni_Reasoning_V3VideoProcessorInitKwargs(VideosKwargs, total=False):
    norm_mean: list[float] | None
    norm_std: list[float] | None
    patch_size: int
    downsample_ratio: float
    video_target_num_patches: int
    video_maintain_aspect_ratio: bool


@auto_docstring
class NemotronH_Omni_Reasoning_V3VideoProcessor(BaseVideoProcessor):
    r"""
    Frames are resized to a single aspect-preserving tile whose patch grid sits near
    `video_target_num_patches`, then normalized with `norm_mean` / `norm_std`. This is the video
    counterpart of [`NemotronH_Omni_Reasoning_V3ImageProcessor`], which uses a per-image dynamic
    resolution budget instead.

    Args:
        norm_mean (`list[float]`, *optional*):
            Per-channel mean used to normalize the frames.
        norm_std (`list[float]`, *optional*):
            Per-channel standard deviation used to normalize the frames.
        patch_size (`int`, *optional*, defaults to 16):
            Side length, in pixels, of one vision-tower patch.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            Pixel-shuffle spatial downsample ratio; its reciprocal is the patch-grid divisor.
        video_target_num_patches (`int`, *optional*, defaults to 1024):
            Patch-grid budget each frame is resized towards.
        video_maintain_aspect_ratio (`bool`, *optional*, defaults to `True`):
            Whether to preserve the frame aspect ratio when choosing the tile; a square tile is
            used otherwise.
    """

    valid_kwargs = NemotronH_Omni_Reasoning_V3VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    norm_mean = None
    norm_std = None
    patch_size = 16
    downsample_ratio = 0.5
    video_target_num_patches = 1024
    video_maintain_aspect_ratio = True

    def __init__(self, **kwargs: Unpack[NemotronH_Omni_Reasoning_V3VideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    @property
    def _downsample_factor(self) -> int:
        """Integer reduction factor for pixel_shuffle (downsample_ratio = 0.5 -> factor 2)."""
        return int(round(1.0 / self.downsample_ratio))

    def _compute_target_patches(self, height: int, width: int) -> tuple[int, int]:
        """Choose an aspect-preserving `(w_patches, h_patches)` tile near `video_target_num_patches`."""
        target = self.video_target_num_patches
        divisor = self._downsample_factor
        if self.video_maintain_aspect_ratio:
            aspect_wh = width / max(height, 1)
            ph = max(round(math.sqrt(target / aspect_wh)), 1)
            pw = max(round(math.sqrt(target * aspect_wh)), 1)
            if divisor > 1:
                rem_h = ph % divisor
                rem_w = pw % divisor
                ph_up = ph + (divisor - rem_h if rem_h else 0)
                ph_down = ph - rem_h
                pw_up = pw + (divisor - rem_w if rem_w else 0)
                pw_down = pw - rem_w
                if ph_up * pw_up <= target:
                    ph, pw = ph_up, pw_up
                else:
                    ph = max(divisor, ph_down)
                    pw = max(divisor, pw_down)
        else:
            side = int(math.sqrt(target))
            side = max(divisor, (side // divisor) * divisor)
            ph = pw = side
        return pw, ph

    def _preprocess(self, videos: list["torch.Tensor"], return_tensors=None, **kwargs):
        norm_mean = torch.tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.tensor(self.norm_std).view(1, 3, 1, 1)
        divisor = self._downsample_factor

        # Frames of one video share a size, so the whole clip resizes in a single batched call.
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = stacked_videos.shape[-2:]
            wp, hp = self._compute_target_patches(height, width)
            target_h, target_w = hp * self.patch_size, wp * self.patch_size

            # `group_videos_by_shape` stacks to (num_videos, num_frames, C, H, W); `interpolate` needs the
            # frames flattened into one batch, so the whole group resizes in one call.
            num_videos, num_frames = stacked_videos.shape[:2]
            frames = stacked_videos.flatten(0, 1).to(dtype=torch.float32)
            if frames.shape[-2] != target_h or frames.shape[-1] != target_w:
                # Antialiased bicubic interpolation via `torch.nn.functional.interpolate`. PIL's
                # bicubic uses a different kernel (and no antialiasing), producing pixel values that
                # amplify through the ViT / mamba stack and diverge past the first few tokens.
                frames = torch.nn.functional.interpolate(
                    frames, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
                )
            frames = (frames / 255.0 - norm_mean) / norm_std
            resized_grouped[shape] = frames.view(num_videos, num_frames, *frames.shape[1:])
        resized_videos = reorder_videos(resized_grouped, grouped_videos_index)

        # One tile per frame; every frame of a clip shares its size and so its token count.
        num_tokens = []
        for video in videos:
            wp, hp = self._compute_target_patches(*video.shape[-2:])
            num_tokens.extend([(wp * hp) // (divisor**2)] * video.shape[0])

        pixel_values_videos = torch.cat(resized_videos, dim=0)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "num_patches": [1] * pixel_values_videos.shape[0],
                "num_tokens": num_tokens,
            },
            tensor_type=return_tensors,
        )


__all__ = ["NemotronH_Omni_Reasoning_V3VideoProcessor"]
