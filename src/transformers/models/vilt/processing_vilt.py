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
Processor class for ViLT.
"""

import warnings
from typing import Optional

from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin


class ViltImagesKwargs(ImagesKwargs):
    size_divisor: Optional[int]


class ViltProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ViltImagesKwargs
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


class ViltProcessor(ProcessorMixin):
    r"""
    Constructs a ViLT processor which wraps a BERT tokenizer and ViLT image processor into a single processor.

    [`ViltProcessor`] offers all the functionalities of [`ViltImageProcessor`] and [`BertTokenizerFast`]. See the
    docstring of [`~ViltProcessor.__call__`] and [`~ViltProcessor.decode`] for more information.

    Args:
        image_processor (`ViltImageProcessor`, *optional*):
            An instance of [`ViltImageProcessor`]. The image processor is a required input.
        tokenizer (`BertTokenizerFast`, *optional*):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViltImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    valid_processor_kwargs = ViltProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor


__all__ = ["ViltProcessor"]
