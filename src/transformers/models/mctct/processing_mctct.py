# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
Speech processor class for M-CTC-T
"""
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin


class MCTCTProcessor(ProcessorMixin):
    r"""
    Constructs a MCTCT processor which wraps a MCTCT feature extractor and a MCTCT tokenizer into a single processor.

    [`MCTCTProcessor`] offers all the functionalities of [`MCTCTFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~MCTCTProcessor.__call__`] and [`~MCTCTProcessor.decode`] for more information.

    Args:
        feature_extractor (`MCTCTFeatureExtractor`):
            An instance of [`MCTCTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "MCTCTFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's
        [`~MCTCTFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~MCTCTProcessor.as_target_processor`] this method forwards all its arguments to AutoTokenizer's
        [`~AutoTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's
        [`~MCTCTFeatureExtractor.pad`] and returns its output. If used in the context
        [`~MCTCTProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning MCTCT.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
