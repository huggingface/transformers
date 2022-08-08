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
Processor class for Donut.
"""
import warnings
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin


class DonutProcessor(ProcessorMixin):
    r"""
    Constructs a Donut processor which wraps a Donut feature extractor and an XLMRoBERTa tokenizer into a single
    processor.

    [`DonutProcessor`] offers all the functionalities of [`DonutFeatureExtractor`] and
    [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [`~DonutProcessor.__call__`] and
    [`~DonutProcessor.decode`] for more information.

    Args:
        feature_extractor ([`DonutFeatureExtractor`]):
            An instance of [`DonutFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]):
            An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.
    """
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
        [`~AutoFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~DonutProcessor.as_target_processor`] this method forwards all its arguments to DonutTokenizer's
        [`~DonutTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.feature_extractor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
