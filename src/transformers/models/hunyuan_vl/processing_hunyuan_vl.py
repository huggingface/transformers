# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
import re

from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class HunYuanVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
            "return_mm_token_type_ids": True,
        },
    }


@auto_docstring
class HunYuanVLProcessor(ProcessorMixin):
    r"""
    HunYuanVL processor that wraps an image processor and a tokenizer for image-text-to-text generation.

    The processor expands every `<image>` placeholder in the prompts into a span of placeholder tokens whose length is
    inferred from the corresponding `image_grid_thw`. The model uses those expanded placeholders and image grids to
    build HunYuanVL's multimodal RoPE position ids at runtime.
    """

    valid_processor_kwargs = HunYuanVLProcessorKwargs

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, cat_extra_token: bool = True, **kwargs
    ):
        r"""
        cat_extra_token (`bool`, *optional*, defaults to `True`):
            Whether to account for the two extra tokens that HunYuanVL inserts around each image span when computing
            the expanded image token sequence.
        """
        self.tokenizer = tokenizer

        # HunYuan-style tokenizers expose the special image tokens via attributes; preserve a useful error message
        # if a caller forgot to register them.
        for attr in (
            "image_token",
            "image_token_id",
            "image_start_token",
            "image_start_token_id",
            "image_end_token",
            "image_end_token_id",
            "pad_token",
            "pad_token_id",
        ):
            if getattr(tokenizer, attr, None) is None:
                raise ValueError(
                    f"Tokenizer is missing required attribute '{attr}'. "
                    "Add the corresponding mapping to `extra_special_tokens` in `tokenizer_config.json` or set the "
                    "attribute manually before constructing the processor."
                )

        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.image_start_token
        self.image_start_token_id = tokenizer.image_start_token_id
        self.image_end_token = tokenizer.image_end_token
        self.image_end_token_id = tokenizer.image_end_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.cat_extra_token = cat_extra_token
        chat_template = chat_template if chat_template is not None else getattr(tokenizer, "chat_template", None)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        _, grid_h, grid_w = (int(value) for value in image_inputs["image_grid_thw"][image_idx])
        spatial_patch_size = self.image_processor.spatial_patch_size
        merge_size = self.image_processor.merge_size
        patch_h = grid_h // merge_size // spatial_patch_size
        patch_w = grid_w // merge_size // spatial_patch_size
        num_image_tokens = patch_h * (patch_w + 1) + (2 if self.cat_extra_token else 0)
        return self.image_token * num_image_tokens

    @staticmethod
    def _has_wrappers(prompt: str, token_start: int, start_token: str, token: str, end_token: str) -> bool:
        "Used to check and verify that the users applied chat template correctly, otherwise quality will degrade!"
        start_index = token_start - len(start_token)
        end_index = token_start + len(token)
        return (
            start_index >= 0
            and prompt[start_index:token_start] == start_token
            and prompt[end_index : end_index + len(end_token)] == end_token
        )

    def validate_inputs(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[HunYuanVLProcessorKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)

        if images is not None and text is not None:
            if any(not isinstance(prompt, str) for prompt in text):
                raise ValueError(
                    "`HunYuanVLProcessor` expects string prompts when multimodal inputs are provided so that multimodal "
                    "placeholder tokens can be expanded before tokenization."
                )

            num_image_tokens = sum(prompt.count(self.image_token) for prompt in text)
            num_images = len(make_flat_list_of_images(images))
            if num_image_tokens != num_images:
                raise ValueError(
                    f"Number of {self.image_token} tokens in text ({num_image_tokens}) does not match the number of "
                    f"images ({num_images})."
                )

            for prompt in text:
                for match in re.finditer(re.escape(self.image_token), prompt):
                    if not self._has_wrappers(
                        prompt, match.start(), self.image_start_token, self.image_token, self.image_end_token
                    ):
                        raise ValueError(
                            f"HunYuanVL image placeholders must be formatted as "
                            f"{self.image_start_token}{self.image_token}{self.image_end_token}. "
                            "Please format prompts with the processor chat template or include the image start/end "
                            "tokens explicitly."
                        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Compute the number of placeholder tokens needed for the given list of image sizes."""
        vision_data: dict = {}
        if image_sizes is not None:
            images_kwargs = HunYuanVLProcessorKwargs._defaults.get("images_kwargs", {}).copy()
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size

            num_image_patches_size = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [
                patch_hw[0] // merge_size * (patch_hw[1] // merge_size + 1) + (2 if self.cat_extra_token else 0)
                for patch_hw in num_image_patches_size
            ]
            num_image_patches = [(patch_hw[0] * patch_hw[1]) for patch_hw in num_image_patches_size]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]


__all__ = ["HunYuanVLProcessor"]
