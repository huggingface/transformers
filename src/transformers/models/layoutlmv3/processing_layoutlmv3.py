# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processor class for LayoutLMv3.
"""
from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class LayoutLMv3Processor(ProcessorMixin):
    r"""
    Constructs a LayoutLMv3 processor which combines a LayoutLMv3 feature extractor and a LayoutLMv3 tokenizer into a
    single processor.

    [`LayoutLMv3Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3FeatureExtractor`] to resize and normalize document images, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv3Tokenizer`] or
    [`LayoutLMv3TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        feature_extractor (`LayoutLMv3FeatureExtractor`):
            An instance of [`LayoutLMv3FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`):
            An instance of [`LayoutLMv3Tokenizer`] or [`LayoutLMv3TokenizerFast`]. The tokenizer is a required input.
    """
    feature_extractor_class = "LayoutLMv3FeatureExtractor"
    tokenizer_class = ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast")

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
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
        **kwargs
    ) -> BatchEncoding:
        """
        This method first forwards the `images` argument to [`~LayoutLMv3FeatureExtractor.__call__`]. In case
        [`LayoutLMv3FeatureExtractor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output,
        together with resized and normalized `pixel_values`. In case [`LayoutLMv3FeatureExtractor`] was initialized
        with `apply_ocr` set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user
        along with the additional arguments to [`~LayoutLMv3Tokenizer.__call__`] and returns the output, together with
        resized and normalized `pixel_values`.

        Please refer to the docstring of the above two methods for more information.
        """
        # verify input
        if self.feature_extractor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes "
                "if you initialized the feature extractor with apply_ocr set to True."
            )

        if self.feature_extractor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the feature extractor with apply_ocr set to True."
            )

        # first, apply the feature extractor
        features = self.feature_extractor(images=images, return_tensors=return_tensors)

        # second, apply the tokenizer
        if text is not None and self.feature_extractor.apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the feature extractor always adds a batch dimension)
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
        encoded_inputs["pixel_values"] = images

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

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
