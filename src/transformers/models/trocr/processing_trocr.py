# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Processor class for TrOCR.
"""
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin


class TrOCRProcessor(ProcessorMixin):
    r"""
    Constructs a TrOCR processor which wraps a vision feature extractor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTFeatureExtractor`/`DeiTFeatureExtractor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`ViTFeatureExtractor`/`DeiTFeatureExtractor`]):
            An instance of [`ViTFeatureExtractor`/`DeiTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`]):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
        [`~AutoFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
