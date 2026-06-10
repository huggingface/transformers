# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""
Processor class for PerceptionLM.
"""

from ...image_utils import get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class PerceptionLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class PerceptionLMProcessor(ProcessorMixin):
    valid_processor_kwargs = PerceptionLMProcessorKwargs

    def __init__(
        self,
        video_processor=None,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        chat_template=None,
        pooling_ratio=2,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        pooling_ratio (`int`, *optional*, defaults to 2):
            Pooling ratio for vision tokens. If not 1, 2D adaptive pooling is applied over projected vision tokens.
        """
        self.patch_size = patch_size
        self.pooling_ratio = pooling_ratio
        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.image_token_id
        self.video_token_id = tokenizer.video_token_id
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        media = image_inputs["pixel_values"][image_idx]
        height, width = get_image_size(to_numpy_array(media))
        num_tiles = media.shape[0]
        num_tokens = (
            (height // self.patch_size // self.pooling_ratio)
            * (width // self.patch_size // self.pooling_ratio)
            * num_tiles
        )
        return self.image_token * num_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        media = video_inputs["pixel_values_videos"][video_idx]
        height, width = get_image_size(to_numpy_array(media))
        num_tiles = media.shape[0]
        num_tokens = (
            (height // self.patch_size // self.pooling_ratio)
            * (width // self.patch_size // self.pooling_ratio)
            * num_tiles
        )
        return self.video_token * num_tokens

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = PerceptionLMProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            tile_size = images_kwargs.get("tile_size", None) or self.image_processor.tile_size
            vision_input_type = images_kwargs.get("vision_input_type", None) or self.image_processor.vision_input_type

            num_image_tokens = []
            num_image_patches = []
            for height, width in image_sizes:
                if vision_input_type == "thumb+tile":
                    aspect_ratio = self.image_processor._fit_image_to_canvas(
                        img_width=width, img_height=height, tile_size=tile_size
                    )
                    if aspect_ratio is None:
                        aspect_ratio = self.image_processor._find_closest_aspect_ratio(
                            img_width=width, img_height=height, tile_size=tile_size
                        )
                    num_tiles = aspect_ratio[0] * aspect_ratio[1] + 1  # base image and tiles
                else:
                    num_tiles = 1

                num_image_tokens.append(
                    (tile_size // self.patch_size // self.pooling_ratio)
                    * (tile_size // self.patch_size // self.pooling_ratio)
                    * num_tiles
                )
                num_image_patches.append(num_tiles)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)


__all__ = ["PerceptionLMProcessor"]
