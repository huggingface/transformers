# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from ...processing_utils import BatchFeature, MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
    }


@auto_docstring
class AyaVisionProcessor(ProcessorMixin):
    valid_processor_kwargs = AyaVisionProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 28,
        img_size: int = 364,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        downsample_factor: int = 1,
        start_of_img_token="<|START_OF_IMG|>",
        end_of_img_token="<|END_OF_IMG|>",
        img_patch_token="<|IMG_PATCH|>",
        img_line_break_token="<|IMG_LINE_BREAK|>",
        tile_token="TILE",
        tile_global_token="TILE_GLOBAL",
        chat_template=None,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 28):
            The size of image patches for tokenization.
        img_size (`int`, *optional*, defaults to 364):
            The size of the image to be tokenized. This should correspond to the size given to the image processor.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The token to be used to represent an image in the text.
        downsample_factor (`int`, *optional*, defaults to 1):
            The factor by which to scale the patch size.
        start_of_img_token (`str`, *optional*, defaults to `"<|START_OF_IMG|>"`):
            The token to be used to represent the start of an image in the text.
        end_of_img_token (`str`, *optional*, defaults to `"<|END_OF_IMG|>"`):
            The token to be used to represent the end of an image in the text.
        img_patch_token (`str`, *optional*, defaults to `"<|IMG_PATCH|>"`):
            The token to be used to represent an image patch in the text.
        img_line_break_token (`str`, *optional*, defaults to `"<|IMG_LINE_BREAK|>"`):
            The token to be used to represent a line break in the text.
        tile_token (`str`, *optional*, defaults to `"TILE"`):
            The token to be used to represent an image patch in the text.
        tile_global_token (`str`, *optional*, defaults to `"TILE_GLOBAL"`):
            The token to be used to represent the cover image in the text.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token = image_token
        self.patch_size = patch_size * downsample_factor
        self.img_size = img_size

        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token
        self.tile_token = tile_token
        self.tile_global_token = tile_global_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.img_patch_token)

    @property
    def image_token_ids(self) -> list[int]:
        return self.tokenizer.convert_tokens_to_ids(
            [
                self.img_patch_token,
                self.tile_token,
                self.tile_global_token,
                self.start_of_img_token,
                self.end_of_img_token,
            ]
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_patches = image_inputs["num_patches"][image_idx]
        img_patches_per_tile = (self.img_size // self.patch_size) ** 2
        img_string = f"{self.start_of_img_token}"
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{self.tile_token}_{idx}" + f"{self.img_patch_token}" * img_patches_per_tile

        img_string += f"{self.tile_global_token}" + f"{self.img_patch_token}" * img_patches_per_tile
        img_string += f"{self.end_of_img_token}"
        return img_string

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        """
        Checks that number of special tokens in text and processed text is same. The count can be different
        if tokenized text was truncated, leading to issues in model code.
        """
        # Aya visino uses `img_patch_token` instead of image token`
        token_str = self.img_patch_token
        token_id = self.image_token_id
        if token_str is not None and token_id is not None:
            ids_count = [list(ids).count(token_id) for ids in text_inputs["input_ids"]]
            text_count = [sample.count(token_str) for sample in text]

            if ids_count != text_count:
                raise ValueError(
                    f"Mismatch in `image` token count between text and `input_ids`. Got ids={ids_count} and text={text_count}. "
                    "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                )

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
            images_kwargs = AyaVisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            token_per_patch = (self.img_size // self.patch_size) ** 2
            num_image_tokens = [
                token_per_patch + 3 + sum(token_per_patch + 1 for _ in range(1, num_patches))
                for num_patches in num_image_patches
            ]  # Add +3 and +1 for BOI/EOI and image tile tokens
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_patches"]


__all__ = ["AyaVisionProcessor"]
