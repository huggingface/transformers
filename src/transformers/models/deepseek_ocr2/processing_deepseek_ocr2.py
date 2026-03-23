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
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class DeepseekOcr2ImagesKwargs(ImagesKwargs, total=False):
    """
    crop_to_patches (`bool`, *optional*):
        Whether to crop the image into local patches.
    min_patches (`int`, *optional*):
        The minimum number of patches to extract from the image for the local view.
    max_patches (`int`, *optional*):
        The maximum number of patches to extract from the image for the local view.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int


class DeepseekOcr2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: DeepseekOcr2ImagesKwargs
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

    def _get_num_multimodal_tokens(self, num_crops: int) -> int:
        """
        Calculate the total number of image tokens for a given number of crops.

        The total is composed of:
        - Global tokens: (ceil(size / patch_size / downsample_ratio))^2
        - Local tokens per crop: (ceil(tile_size / patch_size / downsample_ratio))^2
        - 1 separator token

        Args:
            num_crops (`int`):
                The number of local patches the image was divided into.

        Returns:
            `int`: Total number of image tokens.
        """
        size = self.image_processor.size["height"]
        tile_size = self.image_processor.tile_size

        num_queries_global = math.ceil(size / self.patch_size / self.downsample_ratio)
        global_tokens = num_queries_global * num_queries_global

        num_queries_local = math.ceil(tile_size / self.patch_size / self.downsample_ratio)
        local_tokens = num_queries_local * num_queries_local

        total = global_tokens + local_tokens * num_crops + 1  # +1 for separator
        return total

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
        crop_index = 0
        processed_text = []
        for sample in text:
            parts = sample.split(self.image_token)
            # N occurrences of image_token produce N+1 parts
            expanded = parts[0]
            for part in parts[1:]:
                if crop_index >= len(num_crops_list):
                    raise ValueError(
                        f"Number of `{self.image_token}` tokens in text exceeds the number of images provided. "
                        f"Found more placeholders than the {len(num_crops_list)} images given."
                    )
                num_crops = num_crops_list[crop_index]
                num_tokens = self._get_num_multimodal_tokens(num_crops)
                expanded += self.image_token * num_tokens + part
                crop_index += 1
            processed_text.append(expanded)
        return processed_text

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
        if text is None and images is None:
            raise ValueError("You must provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            DeepseekOcr2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            # Get number of local patches per image from pixel_values_local
            num_crops_list = [len(patches) for patches in image_inputs["pixel_values_local"]]
            text = self._expand_image_tokens(text, num_crops_list)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["DeepseekOcr2Processor"]
