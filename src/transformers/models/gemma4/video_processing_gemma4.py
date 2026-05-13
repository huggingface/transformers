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


import torch

from ...image_processing_utils import BatchFeature
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoInput
from .image_processing_gemma4 import _SUPPORTED_SOFT_TOKENS, get_aspect_ratio_preserving_size


if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


class Gemma4VideoProcessorKwargs(VideosKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per video frame.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    """

    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


def convert_video_to_patches(video: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 4D tensor video of shape (num_frames, num_channels, height, width) into 3D tensor of patches of shape
    (num_frames, num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_frames, num_channels, height, width = video.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    patched_video = video.reshape(
        num_frames, num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_video = patched_video.permute(0, 2, 4, 3, 5, 1)
    patched_video = patched_video.reshape(num_frames, num_patches_height * num_patches_width, -1)
    return patched_video


def pad_to_max_patches(
    video: "torch.Tensor", positions: "torch.Tensor", target_length: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the video along to max number of patches
    """
    current_length = video.shape[1]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0, 0, padding_length, 0, 0]
        pos_padding = (0, 0, 0, padding_length, 0, 0)
        video = torch.nn.functional.pad(video, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return video, positions


@add_start_docstrings(
    "Constructs a Gemma4 video processor that samples frames from videos for use with the Gemma4 model.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
)
class Gemma4VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    num_frames = 32
    do_sample_frames = True
    patch_size = 16
    max_soft_tokens = 70
    pooling_kernel_size = 3
    valid_kwargs = Gemma4VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_position_ids"]

    def __init__(self, **kwargs: Unpack[Gemma4VideoProcessorKwargs]):
        super().__init__(**kwargs)

        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}.")

    def _validate_preprocess_kwargs(self, **kwargs):
        # Gemma4 uses aspect_ratio_preserving_resize driven by patch_size,
        # max_soft_tokens, and pooling_kernel_size — not the standard `size`
        # parameter. Temporarily disable do_resize so the base validation
        # doesn't require `size` to be set.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        video: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: F.InterpolationMode,
    ) -> torch.Tensor:
        height, width = video.shape[-2], video.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )

        if target_height == height and target_width == width:
            return video

        return F.resize(
            video,
            size=[target_height, target_width],
            interpolation=resample,
            antialias=True,
        )

    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[Gemma4VideoProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(videos, **kwargs)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        resample: "F.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")

        max_patches = max_soft_tokens * pooling_kernel_size**2

        pixel_values = []
        position_ids = []
        num_soft_tokens_per_video = []
        num_frames = 1

        for video in videos:
            if do_resize:
                video = self.aspect_ratio_preserving_resize(
                    video=video,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            video = self.rescale_and_normalize(video, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            num_frames = video.shape[0]
            patch_height = video.shape[-2] // patch_size
            patch_width = video.shape[-1] // patch_size
            patches = convert_video_to_patches(video, patch_size)
            num_soft_tokens_per_video.append(patches.shape[1] // pooling_kernel_size**2)

            device = video.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            real_positions = stacked_grid.reshape(patches.shape[1], 2)
            real_positions = real_positions[None, ...].repeat(num_frames, 1, 1)

            patches, positions = pad_to_max_patches(patches, real_positions, max_patches)
            pixel_values.append(patches)
            position_ids.append(positions)

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values, dim=0)  # (num_videos, num_frames, max_patches, patch_pixels)
        position_ids = torch.stack(position_ids, dim=0)  # (num_videos, num_frames, max_patches, 2)

        data = {
            "pixel_values_videos": pixel_values,
            "video_position_ids": position_ids,
            "num_soft_tokens_per_video": num_soft_tokens_per_video,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Gemma4VideoProcessor"]
