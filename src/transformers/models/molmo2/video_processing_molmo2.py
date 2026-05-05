# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Video processor class for Molmo2"""

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring, logging
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata
from .image_processing_molmo2 import arange_for_pooling, batch_pixels_to_patches


logger = logging.get_logger(__name__)


class Molmo2VideoProcessorKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Side length in pixels of each ViT patch for video frames.
    pooling_size (`list[int]`, *optional*):
        `[pool_h, pool_w]` pooling window applied to video patch features.
    max_fps (`int`, *optional*):
        Maximum sampling rate in frames per second for short videos.
    """

    patch_size: int | None
    pooling_size: list[int] | None
    max_fps: int | None


@auto_docstring
class Molmo2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"height": 378, "width": 378}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 14
    pooling_size = [3, 3]
    num_frames = 64
    do_sample_frames = True
    max_fps = 2
    valid_kwargs = Molmo2VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_token_pooling", "video_grids"]

    def __init__(self, **kwargs: Unpack[Molmo2VideoProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (self.size.get("height", None) is None or self.size.get("width", None) is None):
            raise ValueError("size must contain 'height' and 'width' keys.")

    def _standardize_kwargs(
        self,
        size: SizeDict | None = None,
        **kwargs,
    ) -> dict:
        if size is not None and ("height" not in size or "width" not in size):
            raise ValueError("size must contain 'height' and 'width' keys.")

        return super()._standardize_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        max_fps: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Uniform sampling that always includes the last frame. When `max_fps` is set,
        samples at that rate if the video is short enough; otherwise falls back to
        uniform sampling of `num_frames` frames.

        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            max_fps (`int`, *optional*):
                Maximum frames per second to sample. Defaults to `self.max_fps`.
        """
        num_frames = num_frames if num_frames is not None else self.num_frames
        max_fps = max_fps if max_fps is not None else self.max_fps
        total_num_frames = metadata.total_num_frames

        if total_num_frames <= 2:
            return np.arange(total_num_frames).astype(int)

        if max_fps is not None and metadata.fps is not None:
            duration = total_num_frames / metadata.fps
            if duration <= (num_frames - 1) / max_fps:
                # Short video: sample at max_fps and include last frame
                float_indices = np.arange(0.0, stop=total_num_frames - 1, step=float(metadata.fps / max_fps))
                if np.round(float_indices[-1]) != total_num_frames - 1:
                    float_indices = np.concatenate([float_indices, [total_num_frames - 1]], axis=0)
                indices = np.round(float_indices).astype(int)
                if len(indices) > num_frames:
                    raise ValueError(f"Sampled {len(indices)} frames but max is {num_frames}.")
                return indices

        # Uniform fallback: evenly spaced including last frame
        indices = np.linspace(
            0,
            total_num_frames - 1,
            num=min(num_frames, total_num_frames),
            endpoint=True,
        ).astype(int)
        return indices

    def _build_frame_patches(
        self,
        frame_chw: torch.Tensor,
        base_image_input_size: list[int],
        resample: PILImageResampling,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_h: int,
        image_pooling_w: int,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        chw_resized = self.resize(
            frame_chw,
            size=SizeDict(height=base_image_input_size[0], width=base_image_input_size[1]),
            resample=resample,
            antialias=False,
        )
        chw_float = self.rescale(chw_resized.float(), scale=1.0 / 255.0)
        chw_normalized = self.normalize(chw_float, mean=image_mean, std=image_std)
        hwc = chw_normalized.permute(1, 2, 0).unsqueeze(0)  # → [1, H, W, C]

        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        resize_idx = torch.arange(crop_patch_w * crop_patch_h, dtype=torch.int32).reshape(crop_patch_h, crop_patch_w)

        pooling_idx = arange_for_pooling(resize_idx, image_pooling_h, image_pooling_w)
        h, w = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, image_pooling_h * image_pooling_w])

        return [h, w], batch_pixels_to_patches(hwc, image_patch_size), pooling_idx

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        size: SizeDict | None = None,
        resample: PILImageResampling | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        pooling_size: list[int] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        base_image_input_size = [size.height, size.width]
        image_pooling_h, image_pooling_w = pooling_size

        batch_grids = []
        batch_crops = []
        batch_pooled_patches_idx = []

        for video in videos:
            all_crops = []
            pooled_patches_idx = []

            for frame_chw in video:
                image_grid, crops, pooled_idx = self._build_frame_patches(
                    frame_chw,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_h,
                    image_pooling_w,
                )
                offset = sum(c.shape[0] * c.shape[1] for c in all_crops) if all_crops else 0
                pooled_idx_with_offset = torch.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
                pooled_patches_idx.append(pooled_idx_with_offset)
                all_crops.append(crops)

            video_grid = torch.tensor([len(video), image_grid[0], image_grid[1]], dtype=torch.int64)
            all_crops_tensor = torch.cat(all_crops, 0)
            pooled_patches_idx_tensor = torch.cat(pooled_patches_idx, 0)

            batch_grids.append(video_grid)
            batch_crops.append(all_crops_tensor)
            batch_pooled_patches_idx.append(pooled_patches_idx_tensor)

        video_grids = torch.stack(batch_grids, 0)
        pixel_values_videos = torch.cat(batch_crops, 0)
        video_token_pooling = torch.cat(batch_pooled_patches_idx, 0)

        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_token_pooling": video_token_pooling,
            "video_grids": video_grids,
        }

        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["Molmo2VideoProcessor"]
