# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""PIL Image processor class for GLM-4.1V."""

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.glm4v.image_processing_glm4v.smart_resize
def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Glm4vImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int


@auto_docstring
class Glm4vImageProcessorPil(PilBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm4vImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm4vImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        factor: int,
        temporal_factor: int,
        **kwargs,
    ) -> np.ndarray:
        """Resize dynamically based on input image aspect ratio."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")

        height, width = image.shape[-2:]
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return super().resize(
            image=image,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def patchify(
        self,
        image: np.ndarray,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[np.ndarray, int, int]:
        """Patchifies each image into flat layout of shape (`seq_len`, `patch_dim`) so we can concat dynamically shaped pixels."""
        # Ensure float32 for patch processing
        image = np.asarray(image, dtype=np.float32)
        channel, resized_height, resized_width = image.shape
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = image.reshape(
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # (gh, gw, mh, mw, C, ph, pw)
        patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))

        # expand temporal_patch_size as a broadcast (zero-copy)
        patches = np.broadcast_to(
            patches[:, :, :, :, :, None, :, :],
            (*patches.shape[:5], temporal_patch_size, *patches.shape[5:]),
        )

        flatten_patches = patches.reshape(
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
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
            if do_resize:
                image = self.resize(
                    image,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                    temporal_factor=temporal_patch_size,
                )

            # Rescale and normalize
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches, grid_h, grid_w = self.patchify(
                image,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            # Remove batch dimension and append: shape is (seq_len, hidden_dim)
            processed_images.append(patches)
            processed_grids.append([1, grid_h, grid_w])

        # Concatenate all images along sequence dimension: (total_seq_len, hidden_dim)
        pixel_values = np.concatenate(processed_images, axis=0)
        image_grid_thw = np.array(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = images_kwargs.get("size", self.size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge,
            max_pixels=size["longest_edge"] if isinstance(size, dict) else size.longest_edge,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Glm4vImageProcessorPil"]
