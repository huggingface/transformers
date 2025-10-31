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
Audio/Text processor class for CLAP
"""

from ...processing_utils import ProcessorMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class ClapProcessor(ProcessorMixin):
    r"""
    Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

    [`ClapProcessor`] offers all the functionalities of [`ClapFeatureExtractor`] and [`RobertaTokenizerFast`]. See the
    [`~ClapProcessor.__call__`] and [`~ClapProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ClapFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """

    feature_extractor_class = "ClapFeatureExtractor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)


__all__ = ["ClapProcessor"]
