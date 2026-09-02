# Copyright 2021 The HuggingFace Inc. team.
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
Processor class for LayoutLMv2.
"""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, auto_docstring


@auto_docstring
class LayoutLMv2Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        text_pair: PreTokenizedInput | list[PreTokenizedInput] | None = None,
        boxes: list[list[int]] | list[list[list[int]]] | None = None,
        word_labels: list[int] | list[list[int]] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = False,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        # verify input
        if self.image_processor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True."
            )

        if self.image_processor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the image processor with apply_ocr set to True."
            )

        if return_overflowing_tokens is True and return_offsets_mapping is False:
            raise ValueError("You cannot return overflowing tokens without returning the offsets mapping.")

        # first, apply the image processor
        features = self.image_processor(images=images, return_tensors=return_tensors)

        # second, apply the tokenizer
        if text is not None and self.image_processor.apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the image processor always adds a batch dimension)
            text_pair = features["words"]

        encoded_inputs = self.tokenizer(
            text=text if text is not None else features["words"],
            text_pair=text_pair if text_pair is not None else None,
            boxes=boxes if boxes is not None else features["boxes"],
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )

        # add pixel values
        images = features.pop("pixel_values")
        if return_overflowing_tokens is True:
            images = self.get_overflowing_images(images, encoded_inputs["overflow_to_sample_mapping"])
        encoded_inputs["image"] = images

        return encoded_inputs

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # in case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]


__all__ = ["LayoutLMv2Processor"]
