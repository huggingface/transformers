# coding=utf-8
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
"""Fast Image processor class for GLM-4.1V."""

from typing import Optional, Union

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from .image_processing_glm4v import smart_resize


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


class Glm4vFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


@auto_docstring
class Glm4vImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm4vFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm4vFastImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.
        """

        processed_images = []
        processed_grids = []

        all_target_sizes = []
        for image in images:
            height, width = image.shape[-2:]
            resized_height, resized_width = smart_resize(
                num_frames=temporal_patch_size,
                height=height,
                width=width,
                temporal_factor=temporal_patch_size,
                factor=patch_size * merge_size,
                min_pixels=size.shortest_edge,
                max_pixels=size.longest_edge,
            )
            all_target_sizes.append((resized_height, resized_width))

        target_height = max([s[0] for s in all_target_sizes])
        target_width = max([s[1] for s in all_target_sizes])

        for image in images:
            if do_resize:
                image = self.resize(
                    image,
                    size=SizeDict(height=target_height, width=target_width),
                    interpolation=interpolation,
                )

            image = self.rescale_and_normalize(
                image.unsqueeze(0), do_rescale, rescale_factor, do_normalize, image_mean, image_std
            ).squeeze(0)

            patches = image.unsqueeze(0)
            if patches.shape[0] % temporal_patch_size != 0:
                repeats = patches[-1:].repeat(temporal_patch_size - (patches.shape[0] % temporal_patch_size), 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=0)
            channel = patches.shape[1]
            grid_t = patches.shape[0] // temporal_patch_size
            grid_h, grid_w = target_height // patch_size, target_width // patch_size
            patches = patches.view(
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
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_images.append(flatten_patches)
            processed_grids.append([grid_t, grid_h, grid_w])

        pixel_values = torch.stack(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Glm4vFastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)


__all__ = ["Glm4vImageProcessorFast"]
