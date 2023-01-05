# coding=utf-8
# Copyright 2022-2023 The HuggingFace Inc. team. All rights reserved.
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


class SpeechT5ProcessorForSpeechToText(ProcessorMixin):
    r"""
    Constructs a SpeechT5 processor which wraps a waveform feature extractor and a tokenizer into a single
    processor.

    [`SpeechT5ProcessorForSpeechToText`] offers all the functionalities of [`SpeechT5WaveformFeatureExtractor`] and
    [`SpeechT5Tokenizer`]. See the docstring of [`~SpeechT5ProcessorForSpeechToText.__call__`] and
    [`~SpeechT5ProcessorForSpeechToText.decode`] for more information.

    Args:
        feature_extractor (`SpeechT5WaveformFeatureExtractor`):
            An instance of [`SpeechT5WaveformFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "SpeechT5WaveformFeatureExtractor"
    tokenizer_class = "SpeechT5Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5WaveformFeatureExtractor's
        [`~SpeechT5WaveformFeatureExtractor.__call__`] and returns its output.

        You can process your labels by using the argument `text` (either in the same call as your audio inputs, or in a
        separate call). This forwards its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.__call__`].

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
        This method forwards all its arguments to SpeechT5WaveformFeatureExtractor's
        [`~SpeechT5WaveformFeatureExtractor.pad`] and returns its output.

        You can process your labels by using the argument `text` (either in the same call as your audio inputs, or in a
        separate call). This forwards its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.pad`].

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
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


class SpeechT5ProcessorForCTC(SpeechT5ProcessorForSpeechToText):
    r"""
    Constructs a SpeechT5 processor which wraps a waveform feature extractor and a CTC tokenizer into a single
    processor.

    [`SpeechT5ProcessorForSpeechToText`] offers all the functionalities of [`SpeechT5WaveformFeatureExtractor`] and
    [`SpeechT5CTCTokenizer`]. See the docstring of [`~SpeechT5ProcessorForSpeechToText.__call__`] and
    [`~SpeechT5ProcessorForSpeechToText.decode`] for more information.

    Args:
        feature_extractor (`SpeechT5WaveformFeatureExtractor`):
            An instance of [`SpeechT5WaveformFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5CTCTokenizer`):
            An instance of [`SpeechT5CTCTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "SpeechT5WaveformFeatureExtractor"
    tokenizer_class = "SpeechT5CTCTokenizer"


class SpeechT5ProcessorForTextToSpeech(ProcessorMixin):
    r"""
    Constructs a SpeechT5 processor which wraps a tokenizer and a spectrogram feature extractor into a single
    processor.

    [`SpeechT5ProcessorForTextToSpeech`] offers all the functionalities of [`SpeechT5Tokenizer`] and
    [`SpeechT5SpectrogramFeatureExtractor`]. See the docstring of [`~SpeechT5ProcessorForTextToSpeech.__call__`] for
    more information.

    Args:
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
        feature_extractor (`SpeechT5SpectrogramFeatureExtractor`):
            An instance of [`SpeechT5SpectrogramFeatureExtractor`]. The feature extractor is a required input.
    """
    feature_extractor_class = "SpeechT5SpectrogramFeatureExtractor"
    tokenizer_class = "SpeechT5Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.__call__`] and returns its
        output.

        You can process your labels by using the argument `audio` (either in the same call as your text inputs, or in a
        separate call). This forwards its arguments to SpeechT5SpectrogramFeatureExtractor's
        [`~SpeechT5SpectrogramFeatureExtractor.__call__`].

        Please refer to the docstring of the above two methods for more information.
        """
        text = kwargs.pop("text", None)
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)

        if len(args) > 0:
            text = args[0]
            args = args[1:]

        if text is None and audio is None:
            raise ValueError("You need to specify either a `text` or `audio` input to process.")

        if text is not None:
            inputs = self.tokenizer(text, **kwargs)
        if audio is not None:
            encodings = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        if audio is None:
            return inputs
        elif text is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_values"]
            inputs["stop_labels"] = encodings["stop_labels"]
            return inputs
