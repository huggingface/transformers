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

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


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
        self.image_token_id = tokenizer.boi_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.boi_token
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
        if text is not None and not isinstance(text, str):
            if not isinstance(text, list) or not isinstance(text[0], str):
                raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        model_inputs = super().__call__(images=images, text=text, **kwargs)
        if "mm_token_type_ids" in model_inputs:
            model_inputs["token_type_ids"] = model_inputs.pop("mm_token_type_ids")
        return model_inputs

    def prepare_inputs_layout(self, images=None, text=None, **kwargs):
        images, text, *_ = super().prepare_inputs_layout(images=images, text=text, **kwargs)
        if images is not None:
            images = make_nested_list_of_images(images)
            # Create empty text to be replaced with placeholders
            if not text:
                text = [" ".join([self.boi_token] * len(image_list)) for image_list in images]
        return images, text, None, None

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)

        if images is not None and text is not None:
            if len(images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(images)}) and text ({len(text)})."
                )

            for prompt, images in zip(text, images):
                if len(images) != prompt.count(self.boi_token):
                    raise ValueError(
                        f"Prompt contained {prompt.count(self.boi_token)} image tokens but received {len(images)} images."
                    )

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        """
        Checks that number of special tokens in text and processed text is same. The count can be different
        if tokenized text was truncated, leading to issues in model code.

        Gemma3 uses a different token as placeholder in input text than in the expanded text.
        """
        token_str = self.tokenizer.image_token
        token_id = self.tokenizer.image_token_id
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
            # NOTE: no image cropping supported yet
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.tokenizer.image_token_id]

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["token_type_ids"]

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_crops"]

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        num_crops = image_inputs["num_crops"][image_idx]
        if not num_crops:
            return self.full_image_sequence

        # Insert additional image tokens for Pan-and-Scan crops
        return (
            f"Here is the original image {self.full_image_sequence} and here are some crops to help you see better "
            + " ".join([self.full_image_sequence] * num_crops)
        )


__all__ = ["Gemma3Processor"]
