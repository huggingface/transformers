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

from ...file_utils import TensorType
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import auto_docstring


@auto_docstring
class MarkupLMProcessor(ProcessorMixin):
    parse_html = True

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        html_strings=None,
        nodes=None,
        xpaths=None,
        node_labels=None,
        questions=None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = None,
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
        # first, create nodes and xpaths
        r"""
        html_strings (`str` or `list[str]`, *optional*):
            Raw HTML strings to parse and process. When `parse_html=True` (default), these strings are parsed
            to extract nodes and xpaths automatically. If provided, `nodes`, `xpaths`, and `node_labels` should
            not be provided. Required when `parse_html=True`.
        nodes (`list[list[str]]`, *optional*):
            Pre-extracted HTML nodes as a list of lists, where each inner list contains the text content of nodes
            for a single document. Required when `parse_html=False`. Should not be provided when `parse_html=True`.
        xpaths (`list[list[str]]`, *optional*):
            Pre-extracted XPath expressions corresponding to the nodes. Should be a list of lists with the same
            structure as `nodes`, where each XPath identifies the location of the corresponding node in the HTML
            tree. Required when `parse_html=False`. Should not be provided when `parse_html=True`.
        node_labels (`list[list[int]]`, *optional*):
            Labels for the nodes, typically used for training or fine-tuning tasks. Should be a list of lists
            with the same structure as `nodes`, where each label corresponds to a node. Optional and only used
            when `parse_html=False`.
        questions (`str` or `list[str]`, *optional*):
            Question strings for question-answering tasks. When provided, the tokenizer processes questions
            as the first sequence and nodes as the second sequence (text_pair). If a single string is provided,
            it is converted to a list to match the batch dimension of the parsed HTML.
        """
        if self.parse_html:
            if html_strings is None:
                raise ValueError("Make sure to pass HTML strings in case `parse_html` is set to `True`")

            if nodes is not None or xpaths is not None or node_labels is not None:
                raise ValueError(
                    "Please don't pass nodes, xpaths nor node labels in case `parse_html` is set to `True`"
                )

            features = self.feature_extractor(html_strings)
            nodes = features["nodes"]
            xpaths = features["xpaths"]
        else:
            if html_strings is not None:
                raise ValueError("You have passed HTML strings but `parse_html` is set to `False`.")
            if nodes is None or xpaths is None:
                raise ValueError("Make sure to pass nodes and xpaths in case `parse_html` is set to `False`")

        # # second, apply the tokenizer
        if questions is not None and self.parse_html:
            if isinstance(questions, str):
                questions = [questions]  # add batch dimension (as the feature extractor always adds a batch dimension)

        encoded_inputs = self.tokenizer(
            text=questions if questions is not None else nodes,
            text_pair=nodes if questions is not None else None,
            xpaths=xpaths,
            node_labels=node_labels,
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


__all__ = ["MarkupLMProcessor"]
