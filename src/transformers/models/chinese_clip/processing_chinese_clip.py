# coding=utf-8
# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
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
Image/Text processor class for Chinese-CLIP
"""

import warnings

from ...processing_utils import ProcessorMixin


class ChineseCLIPProcessor(ProcessorMixin):
    r"""
    Constructs a Chinese-CLIP processor which wraps a Chinese-CLIP image processor and a Chinese-CLIP tokenizer into a
    single processor.

    [`ChineseCLIPProcessor`] offers all the functionalities of [`ChineseCLIPImageProcessor`] and [`BertTokenizerFast`].
    See the [`~ChineseCLIPProcessor.__call__`] and [`~ChineseCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`ChineseCLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("ChineseCLIPImageProcessor", "ChineseCLIPImageProcessorFast")
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

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


__all__ = ["ChineseCLIPProcessor"]
