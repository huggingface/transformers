# coding=utf-8
# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang The HuggingFace Inc. team. All rights reserved.
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
Image/Text processor class for AltCLIP
"""

from ...processing_utils import ProcessorMixin
from ...utils.deprecation import deprecate_kwarg


class AltCLIPProcessor(ProcessorMixin):
    r"""
    Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
    processor.

    [`AltCLIPProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`XLMRobertaTokenizerFast`]. See
    the [`~AltCLIPProcessor.__call__`] and [`~AltCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("CLIPImageProcessor", "CLIPImageProcessorFast")
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

    @deprecate_kwarg(old_name="feature_extractor", version="5.0.0", new_name="image_processor")
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)


__all__ = ["AltCLIPProcessor"]
