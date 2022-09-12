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
Processor class for MarkupLM.
"""
from typing import Optional, Union

from ...file_utils import TensorType
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy


class MarkupLMProcessor(ProcessorMixin):
    r"""
    Constructs a MarkupLM processor which combines a MarkupLM feature extractor and a MarkupLM tokenizer into a single
    processor.

    [`MarkupLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`MarkupLMFeatureExtractor`] to get nodes and corresponding xpaths from one or more HTML strings.
    Next, these are provided to [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`], which turns them into token-level
    `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and `xpath_subs_seq`.

    Args:
        feature_extractor (`MarkupLMFeatureExtractor`):
            An instance of [`MarkupLMFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`MarkupLMTokenizer` or `MarkupLMTokenizerFast`):
            An instance of [`MarkupLMTokenizer`] or [`MarkupLMTokenizerFast`]. The tokenizer is a required input.
    """
    feature_extractor_class = "MarkupLMFeatureExtractor"
    tokenizer_class = ("MarkupLMTokenizer", "MarkupLMTokenizerFast")

    def __call__(
        self,
        html_strings,
        text,
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
        **kwargs
    ) -> BatchEncoding:
        """
        This method first forwards the `html_strings` argument to [`~MarkupLMFeatureExtractor.__call__`]. Next, it
        passes the `nodes` and `xpaths` along with the additional arguments to [`~MarkupLMTokenizer.__call__`] and
        returns the output.

        Optionally, one can also provide a `text` argument which is passed along as first sequence.

        Please refer to the docstring of the above two methods for more information.
        """
        # first, apply the feature extractor
        features = self.feature_extractor(html_strings)

        # second, apply the tokenizer
        encoded_inputs = self.tokenizer(
            text=text if text is not None else features["nodes"],
            text_pair=features["nodes"] if text is not None else None,
            xpaths=features["xpaths"],
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

        return encoded_inputs
