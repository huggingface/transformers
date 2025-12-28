# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
Image/Text processor class for FLAVA
"""

from ...processing_utils import ProcessorMixin


class FlavaProcessor(ProcessorMixin):
    r"""
    Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

    [`FlavaProcessor`] offers all the functionalities of [`FlavaImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~FlavaProcessor.__call__`] and [`~FlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*): The tokenizer is a required input.
    """

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)


__all__ = ["FlavaProcessor"]
