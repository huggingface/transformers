# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
Processor class for BridgeTower.
"""

from typing import Optional

from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin


class BridgeTowerImagesKwargs(ImagesKwargs):
    size_divisor: Optional[int]


class BridgeTowerProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: BridgeTowerImagesKwargs
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
        "images_kwargs": {
            "do_normalize": True,
            "do_center_crop": True,
        },
    }


class BridgeTowerProcessor(ProcessorMixin):
    r"""
    Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single
    processor.

    [`BridgeTowerProcessor`] offers all the functionalities of [`BridgeTowerImageProcessor`] and
    [`RobertaTokenizerFast`]. See the docstring of [`~BridgeTowerProcessor.__call__`] and
    [`~BridgeTowerProcessor.decode`] for more information.

    Args:
        image_processor (`BridgeTowerImageProcessor`):
            An instance of [`BridgeTowerImageProcessor`]. The image processor is a required input.
        tokenizer (`RobertaTokenizerFast`):
            An instance of ['RobertaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BridgeTowerImageProcessor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")
    valid_processor_kwargs = BridgeTowerProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__all__ = ["BridgeTowerProcessor"]
