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
Processor class for CLVP
"""

from ...processing_utils import ProcessorMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class ClvpProcessor(ProcessorMixin):
    r"""
    Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

    [`ClvpProcessor`] offers all the functionalities of [`ClvpFeatureExtractor`] and [`ClvpTokenizer`]. See the
    [`~ClvpProcessor.__call__`], [`~ClvpProcessor.decode`] and [`~ClvpProcessor.batch_decode`] for more information.

    Args:
        feature_extractor (`ClvpFeatureExtractor`):
            An instance of [`ClvpFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`ClvpTokenizer`):
            An instance of [`ClvpTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "ClvpFeatureExtractor"
    tokenizer_class = "ClvpTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`
        argument to [`~ClvpTokenizer.__call__`]. Please refer to the docstring of the above two methods for more
        information.
        """
        raw_speech = kwargs.pop("raw_speech", None)
        if raw_speech is not None:
            logger.warning(
                "Using `raw_speech` keyword argument is deprecated when calling ClvpProcessor, instead use `audio`."
            )
        kwargs["audio"] = raw_speech
        return super().__call__(*args, **kwargs)


__all__ = ["ClvpProcessor"]
