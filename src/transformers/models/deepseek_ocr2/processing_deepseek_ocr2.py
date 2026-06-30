# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Processor class for DeepSeek-OCR-2.
"""

import math

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class DeepseekOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
            "min_patches": 2,
            "max_patches": 6,
        },
    }


@auto_docstring
class DeepseekOcr2Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size=16,
        downsample_ratio=4,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to `16`):
            The patch size used by the vision encoder (SAM ViT patch embedding size).
        downsample_ratio (`int`, *optional*, defaults to `4`):
            The downsampling ratio applied after the vision encoder.
        """
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        num_crops_list: list[int],
    ) -> list[str]:
        """
        Expand each `<image>` placeholder in the text to the correct number of image tokens.

        Args:
            text (`list[str]`):
                List of text strings, each potentially containing `<image>` placeholders.
            num_crops_list (`list[int]`):
                Number of crops for each image, consumed in order as `<image>` placeholders
                are encountered across all text samples.

        Returns:
            `list[str]`: Text with expanded image token placeholders.
        """
        size = self.image_processor.size["height"]
        tile_size = self.image_processor.tile_size

        num_queries_global = math.ceil(size / self.patch_size / self.downsample_ratio)
        global_tokens = num_queries_global * num_queries_global

        num_queries_local = math.ceil(tile_size / self.patch_size / self.downsample_ratio)
        local_tokens = num_queries_local * num_queries_local

        crop_index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_tokens = global_tokens + local_tokens * num_crops_list[crop_index] + 1
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_tokens, 1)
                crop_index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)
        return text

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[DeepseekOcr2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Global view pixel values. Returned when `images` is not `None`.
            - **pixel_values_local** -- Local patch pixel values. Returned when `images` is not `None`.
        """
        if images is None:
            raise ValueError("`images` are expected as arguments to a `DeepseekOcr2Processor` instance.")
        if text is None:
            raise ValueError("`text` is required for `DeepseekOcr2Processor`. Example: `'<image>\\nFree OCR.'`")

        output_kwargs = self._merge_kwargs(
            DeepseekOcr2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        text = text.copy()  # below lines change text in-place

        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        num_crops_list = image_inputs["num_local_patches"]
        text = self._expand_image_tokens(text, num_crops_list)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )


__all__ = ["DeepseekOcr2Processor"]
