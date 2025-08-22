# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Bros.
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin


class BrosProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
    }


class BrosProcessor(ProcessorMixin):
    r"""
    Constructs a Bros processor which wraps a BERT tokenizer.

    [`BrosProcessor`] offers all the functionalities of [`BertTokenizerFast`]. See the docstring of
    [`~BrosProcessor.__call__`] and [`~BrosProcessor.decode`] for more information.

    Args:
        tokenizer (`BertTokenizerFast`, *optional*):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    valid_processor_kwargs = BrosProcessorKwargs

    def __init__(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(tokenizer)


__all__ = ["BrosProcessor"]
