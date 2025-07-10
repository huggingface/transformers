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


from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import add_start_docstrings, is_torch_available


if is_torch_available():
    import torch


@add_start_docstrings(
    "Constructs a fast InstructBLIPVideo image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class InstructBlipVideoImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    do_resize = True
    size = {"height": 384, "width": 384}
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True

    def __call__(self, images=None, return_tensors="pt", input_data_format=None, **kwargs):
        # 1) 4D array-like video (not a torch.Tensor): (frames, H, W, C) or (frames, C, H, W)
        if hasattr(images, "ndim") and images.ndim == 4 and not isinstance(images, torch.Tensor):
            frames = images.shape[0]
            # (frames, H, W, C) → (frames, C, H, W)
            if images.shape[-1] not in (1, 3, 4):
                images = images.transpose(0, 3, 1, 2)
            flat_frames = [images[i] for i in range(frames)]
            bf = super().__call__(
                flat_frames,
                return_tensors=return_tensors,
                input_data_format=input_data_format,
                **kwargs,
            )
            pv = bf["pixel_values"]  # (frames, C, H, W)
            pv = pv.unsqueeze(0)  # (1, frames, C, H, W)
            return BatchFeature(data={"pixel_values": pv}, tensor_type=return_tensors)

        # 2) Batched videos: list of 4D array-like or torch.Tensor each (frames, C, H, W)
        if isinstance(images, list) and len(images) > 0 and hasattr(images[0], "ndim") and images[0].ndim == 4:
            batch_size = len(images)
            frames = images[0].shape[0]
            # Flatten all frames
            flat_frames = []
            for video in images:
                for frame in video:
                    flat_frames.append(frame)
            bf = super().__call__(
                flat_frames,
                return_tensors=return_tensors,
                input_data_format=input_data_format,
                **kwargs,
            )
            pv = bf["pixel_values"]  # (batch_size*frames, C, H, W)
            pv = pv.view(batch_size, frames, pv.size(1), pv.size(2), pv.size(3))  # (batch_size, frames, C, H, W)
            return BatchFeature(data={"pixel_values": pv}, tensor_type=return_tensors)

        # 3) Fallback: default fast processor behavior
        return super().__call__(
            images,
            return_tensors=return_tensors,
            input_data_format=input_data_format,
            **kwargs,
        )


__all__ = ["InstructBlipVideoImageProcessorFast"]
