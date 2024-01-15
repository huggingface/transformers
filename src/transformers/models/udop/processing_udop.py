# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for UDOP.
"""

from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class UdopProcessor(ProcessorMixin):
    r"""
    Constructs a UDOP processor which combines a UDOP image processor and a UDOP tokenizer into a single processor.

    [`UdopProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`UdopImageProcessor`] to resize, rescale and normalize document images, and optionally applies OCR
    to get words and normalized bounding boxes. These are then provided to [`UdopTokenizer`] or [`UdopTokenizerFast`],
    which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`UdopImageProcessor`, *optional*):
            An instance of [`UdopImageProcessor`]. The image processor is a required input.
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`, *optional*):
            An instance of [`UdopTokenizer`] or [`UdopTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "UdopImageProcessor"
    tokenizer_class = ("UdopTokenizer", "UdopTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method first forwards the `images` argument to [`~UdopImageProcessor.__call__`]. In case
        [`UdopImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~UdopTokenizer.__call__`] and returns the output,
        together with the prepared `pixel_values`. In case [`UdopImageProcessor`] was initialized with `apply_ocr` set
        to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the
        additional arguments to [`~UdopTokenizer.__call__`] and returns the output, together with the prepared
        `pixel_values`.

        Please refer to the docstring of the above two methods for more information.
        """
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

        if text_target is not None:
            # use the processor to prepare the targets of UDOP
            return self.tokenizer(
                text_target=text_target,
                text_pair_target=text_pair_target,
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

        else:
            # use the processor to prepare the inputs of UDOP
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
            pixel_values = features.pop("pixel_values")
            if return_overflowing_tokens is True:
                pixel_values = self.get_overflowing_images(pixel_values, encoded_inputs["overflow_to_sample_mapping"])
            encoded_inputs["pixel_values"] = pixel_values

            return encoded_inputs

    # Copied from transformers.models.layoutxlm.processing_layoutxlm.LayoutXLMProcessor.get_overflowing_images
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

    # Copied from transformers.models.layoutxlm.processing_layoutxlm.LayoutXLMProcessor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.layoutxlm.processing_layoutxlm.LayoutXLMProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]
