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
from ...utils.import_utils import requires
from .image_processing_glm4v import smart_resize


logger = logging.get_logger(__name__)


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


@requires(backends=("vision", "torch", "torchvision"))
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

    def __init__(self, **kwargs: Unpack[Glm4vImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None:
            if not self.size.shortest_edge or not self.size.longest_edge:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm4vImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _standardize_kwargs(self, **kwargs) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return kwargs

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
                    factor=patch_size * merge_size,
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

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
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
        if images_kwargs is not None:
            patch_size = images_kwargs.get("patch_size", self.patch_size)
            merge_size = images_kwargs.get("merge_size", self.merge_size)
            size = images_kwargs.get("size", {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000})
        else:
            patch_size = self.patch_size
            merge_size = self.merge_size
            size = self.size

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
