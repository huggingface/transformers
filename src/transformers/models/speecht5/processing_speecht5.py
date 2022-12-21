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
"""Speech processor class for SpeechT5."""

import warnings

from ...processing_utils import ProcessorMixin


class SpeechT5Processor(ProcessorMixin):
    r"""
    Constructs a SpeechT5 processor which wraps a Wav2Vec2 feature extractor and a SpeechT5 tokenizer into a single
    processor.

    [`SpeechT5Processor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`SpeechT5Tokenizer`].
    See the docstring of [`~SpeechT5Processor.__call__`] and [`~SpeechT5Processor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "SpeechT5Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output.

        You can process your labels by using the argument `text` (either in the same call as your audio
        inputs, or in a separate call). This forwards its arguments to SpeechT5Tokenizer's
        [`~SpeechT5Tokenizer.__call__`].

        Please refer to the docstring of the above two methods for more information.
        """
        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)

        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output.

        You can process your labels by using the argument `text` (either in the same call as your audio
        inputs, or in a separate call). This forwards its arguments to SpeechT5Tokenizer's
        [`~SpeechT5Tokenizer.pad`].

        Please refer to the docstring of the above two methods for more information.
        """
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)

        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
