# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, to_py_obj


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


@auto_docstring
class Gemma3Processor(ProcessorMixin):
    valid_processor_kwargs = Gemma3ProcessorKwargs

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        # The placeholder expanded by `get_text_with_replacements` is the begin-of-image token; the
        # repeated `<image_soft_token>` (id below) is only used to build `mm_token_type_ids`.
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.boi_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
        self.mm_image_token_id = tokenizer.image_token_id
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(images)) for images in batched_images]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            num_crops = to_py_obj(image_inputs.pop("num_crops"))
            batch_num_crops = [[num_crops.pop(0) for _ in range(len(images))] for images in batched_images]
            for batch_idx, (prompt, images, num_crops) in enumerate(zip(text, batched_images, batch_num_crops)):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token) :]
                        text[batch_idx] = prompt

            # Expand each begin-of-image placeholder to the full image token sequence
            num_images = sum(prompt.count(self.boi_token) for prompt in text)
            images_replacements = [self.replace_image_token(image_inputs, i) for i in range(num_images)]
            text, _ = self.get_text_with_replacements(list(text), images_replacements=images_replacements)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        # Every begin-of-image placeholder expands to the same fixed-length sequence, so the
        # replacement does not depend on `image_idx`.
        return self.full_image_sequence

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
            # NOTE: no image cropping supported yet
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["token_type_ids"]

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_crops"]

    @property
    def image_token_ids(self) -> list[int | None]:
        # `mm_token_type_ids` must mark the repeated `<image_soft_token>` positions, not the
        # begin-of-image placeholder that `image_token_id` now points to.
        return [self.mm_image_token_id]


__all__ = ["Gemma3Processor"]
