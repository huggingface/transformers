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


from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class Cohere2VisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
    }


@auto_docstring
class Cohere2VisionProcessor(ProcessorMixin):
    valid_processor_kwargs = Cohere2VisionProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.patch_size = self.image_processor.patch_size
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token = tokenizer.image_token
        self.img_line_break_token = tokenizer.img_line_break_token
        self.image_token_id = tokenizer.image_token_id

    @property
    def image_token_ids(self) -> list[int]:
        return self.tokenizer.convert_tokens_to_ids(
            [
                self.image_token,
                self.boi_token,
                self.eoi_token,
                self.img_line_break_token,
            ]
        )

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if text is None:
            raise ValueError("You have to specify text.")
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_patches = image_inputs["num_patches"][image_idx]
        img_patches_per_tile = int(self.patch_size**2)
        tile = self.image_token * img_patches_per_tile + self.img_line_break_token
        return self.boi_token + tile * num_patches + self.eoi_token

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_patches"]

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
            images_kwargs = Cohere2VisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            token_per_patch = int(self.patch_size**2)
            # +2 for BOI/EOI tokens, +1 per patch for img_line_break token
            num_image_tokens = [
                2 + sum(token_per_patch + 1 for _ in range(num_patches)) for num_patches in num_image_patches
            ]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


__all__ = ["Cohere2VisionProcessor"]
