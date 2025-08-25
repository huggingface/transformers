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
"""Video processor class for VideoPrism."""

import numpy as np
import torch
from PIL import Image

from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import is_torchvision_available, is_torchvision_v2_available, is_vision_available
from ...utils.import_utils import requires
from ...video_processing_utils import (
    BaseVideoProcessor,
)


if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torchvision_available():
    # from .image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class VideoPrismFastVideoProcessorInitKwargs(VideosKwargs): ...


@requires(backends=("torchvision",))
class VideoPrismVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR  # PILImageResampling.LANCZOS # PIL.Image.Resampling.LANCZOS
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 288, "width": 288}
    rescale_factor = 1 / 255
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = False
    do_convert_rgb = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = VideoPrismFastVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[VideoPrismFastVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        video: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an video to `(size["height"], size["width"])`.
        Args:
            video (`torch.Tensor`):
                Video to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output video.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the video e.g. `InterpolationMode.BICUBIC`.
        Returns:
            `torch.Tensor`: The resized video.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        print(interpolation)
        print(video.shape)
        if interpolation == F.InterpolationMode.LANCZOS:
            # Resize each frame individually
            video = video.squeeze(0)  # Shape becomes [16, 3, 360, 640]

            resized_frames = []
            for frame in video.squeeze(0):  # Remove batch dimension (shape: [16, 3, 360, 640])
                # Permute dimensions to (height, width, channels)
                frame_np = frame.permute(1, 2, 0).numpy()  # Convert to (360, 640, 3)
                if frame_np.ndim != 3 or frame_np.shape[-1] not in [1, 3, 4]:
                    raise ValueError(f"Invalid frame shape for PIL conversion: {frame_np.shape}")

                # Convert to PIL Image and resize
                pil_frame = Image.fromarray(frame_np)  # Convert each frame to PIL Image
                resized_frame = pil_frame.resize((size.width, size.height), resample=Image.LANCZOS)  # Resize h and w
                resized_frames.append(np.array(resized_frame))  # Convert back to NumPy array

            # Stack resized frames and convert to tensor
            inputs = np.stack(resized_frames, axis=0)  # Shape: (16, size.height, size.width, channels)
            video = torch.from_numpy(inputs).permute(0, 3, 1, 2)  # Convert to (frames, channels, height, width)

            # Add batch dimension back to conform to BTCHW format
            video = video.unsqueeze(0)  # Shape becomes [1, 16, 3, size.height, size.width]
            print(video.shape)
            return video
        else:
            # raise ValueError("Unsupported interpolation mode.")
            super().resize(
                video,
                size,
                interpolation,
                antialias,
                **kwargs,
            )


__all__ = ["VideoPrismVideoProcessor"]
