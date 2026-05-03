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
import torchvision.transforms

from ...image_processing_utils import BatchFeature
from ...image_transforms import normalize
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring, logging
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata


logger = logging.get_logger(__name__)


def resize_image(
    image: np.ndarray,
    desired_output_size: list[int],
    resample: PILImageResampling,
) -> np.ndarray:
    """Resize an image or video and rescale to [0, 1] float32."""
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])

    resized = torchvision.transforms.Resize(desired_output_size, resample, antialias=False)(image)
    resized = torch.clip(resized, 0, 255).to(torch.uint8)
    resized = resized.to(torch.float32) / 255.0

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()

    return resized


def build_resized_image(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image(
        image,
        base_image_input_size,
        resample,
    )
    resized = normalize(resized, image_mean, image_std, input_data_format=ChannelDimension.LAST)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w * crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resize_idx


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches * w_patches, patch_size * patch_size * c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr, [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]], mode="constant", constant_values=-1
    )
    h, w = idx_arr.shape[0] // pool_h, idx_arr.shape[1] // pool_w
    return idx_arr.reshape(h, pool_h, w, pool_w).transpose(0, 2, 1, 3).reshape(h, w, pool_h * pool_w)


def image_to_patches_and_grids(
    image: np.ndarray,
    base_image_input_size: list[int],
    resample: PILImageResampling,
    image_mean: list[float],
    image_std: list[float],
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_grids, the shape of each image after pooling
    :return crops, the image crops to processes with the ViT
    :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                patches in `crops` to pool for that token, masked with -1
    """
    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)

    pooling_w = image_pooling_w
    pooling_h = image_pooling_h

    resized, resize_idx = build_resized_image(
        image,
        base_image_input_size,
        resample,
        image_mean,
        image_std,
        image_patch_size,
    )
    pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
    h, w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])
    image_grid = [h, w]
    return (
        image_grid,
        batch_pixels_to_patches(resized, image_patch_size),
        pooling_idx,
    )


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
            # Convert from torch (T, C, H, W) to numpy (T, H, W, C)
            if isinstance(video, torch.Tensor):
                video = video.permute(0, 2, 3, 1).numpy()

            all_crops = []
            pooled_patches_idx = []

            for frame in video:
                image_grid, crops, pooled_idx = image_to_patches_and_grids(
                    frame,
                    base_image_input_size,
                    resample,
                    image_mean,
                    image_std,
                    patch_size,
                    image_pooling_w,
                    image_pooling_h,
                )
                offset = sum(np.prod(x.shape[:2]) for x in all_crops)
                pooled_idx_with_offset = np.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
                pooled_patches_idx.append(pooled_idx_with_offset)
                all_crops.append(crops)

            video_grid = np.array([len(video), image_grid[0], image_grid[1]])
            all_crops = np.concatenate(all_crops, 0)
            pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)

            batch_grids.append(video_grid)
            batch_crops.append(all_crops)
            batch_pooled_patches_idx.append(pooled_patches_idx)

        video_grids = np.stack(batch_grids, 0)
        pixel_values_videos = np.concatenate(batch_crops, 0)
        video_token_pooling = np.concatenate(batch_pooled_patches_idx, 0)

        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_token_pooling": video_token_pooling,
            "video_grids": video_grids,
        }

        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["Molmo2VideoProcessor"]
