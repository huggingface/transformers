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

import math

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, VideosKwargs
from ...utils import TensorType
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from ..glm46v.configuration_glm46v import Glm46VConfig
from ..glm46v.image_processing_glm46v import Glm46VImageProcessor, smart_resize
from ..glm46v.image_processing_pil_glm46v import Glm46VImageProcessorPil
from ..glm46v.video_processing_glm46v import Glm46VVideoProcessor


# Glmga reuses GLM-4.6V's modeling and processor as-is; only the config and the
# image/video processors differ. The model and processor are wired to the glm46v
# classes through the auto-mappings, so no modeling/processing classes live here.
class GlmgaConfig(Glm46VConfig):
    r"""
    image_start_token_id (`int`, *optional*, defaults to 151339):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 151340):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 151361):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 151362):
        The video end token index to encode the end of video.

    ```python
    >>> from transformers import AutoModelForImageTextToText, GlmgaConfig

    >>> # Initializing a Glmga style configuration
    >>> configuration = GlmgaConfig()

    >>> # Initializing a model (reusing the GLM-4.6V implementation) from that configuration
    >>> model = AutoModelForImageTextToText.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glmga"


class GlmgaImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    patch_expand_factor (`int`, *optional*, defaults to 1):
        The patch_expand_factor of the vision encoder to llm encoder.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int


class GlmgaImageProcessor(Glm46VImageProcessor):
    patch_expand_factor = 1

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:  # (B, C, H, W)
                patches = patches.unsqueeze(1)  # (B, T=1, C, H, W)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1
                )
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, t_len, channel = patches.shape[:3]
            grid_t = t_len // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # (B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )


class GlmgaImageProcessorPil(Glm46VImageProcessorPil):
    patch_expand_factor = 1

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_expand_factor: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images one by one for PIL backend.
        """
        processed_images = []
        processed_grids = []

        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )

            # Rescale and normalize
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            # Ensure float32 for patch processing
            image_array = np.asarray(image, dtype=np.float32)
            if image_array.ndim == 3:  # (C, H, W)
                image_array = np.expand_dims(image_array, axis=0)  # (1, C, H, W)
            if image_array.ndim == 4:  # (B, C, H, W)
                image_array = np.expand_dims(image_array, axis=1)  # (B, T=1, C, H, W)

            resized_height, resized_width = image_array.shape[-2:]

            if image_array.shape[1] % temporal_patch_size != 0:
                repeats = np.repeat(
                    image_array[:, -1:],
                    temporal_patch_size - (image_array.shape[1] % temporal_patch_size),
                    axis=1,
                )
                image_array = np.concatenate([image_array, repeats], axis=1)

            batch_size, t_len, channel = image_array.shape[:3]
            grid_t = t_len // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = image_array.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # (B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
            patches = np.transpose(patches, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9))

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            # Remove batch dimension and append: shape is (seq_len, hidden_dim)
            processed_images.append(flatten_patches.squeeze(0))
            processed_grids.append([grid_t, grid_h, grid_w])

        # Concatenate all images along sequence dimension: (total_seq_len, hidden_dim)
        pixel_values = np.concatenate(processed_images, axis=0)
        image_grid_thw = np.array(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )


class GlmgaVideoProcessorInitKwargs(VideosKwargs, total=False):
    max_image_size: dict[str, int]
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    max_frames: int


class GlmgaVideoProcessor(Glm46VVideoProcessor):
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 2 * 55790}
    max_image_size = {"longest_edge": 28 * 28 * 2 * 55790}
    fps = 2
    patch_expand_factor = 1
    max_frames = 640

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ):
        """
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            np.ndarray:
                Indices to sample video frames.
        """
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was provided which is required when sampling in Glmga. "
                "Please pass in `VideoMetadata` object or set `do_sample_frames=False`"
            )

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1

        target_fps = fps if fps is not None else self.fps

        extract_t = int(duration * target_fps)
        extract_t = min(extract_t, self.max_frames)

        duration_per_frame = 1 / metadata.fps
        timestamps = [i * duration_per_frame for i in range(total_frames)]

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
                    if current_second >= duration - inv_fps:
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

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_expand_factor: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = [
    "GlmgaConfig",
    "GlmgaImageProcessor",
    "GlmgaImageProcessorPil",
    "GlmgaVideoProcessor",
]
