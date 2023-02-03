# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from ...processing_utils import ProcessorMixin


class SpeechT5Processor(ProcessorMixin):
    r"""
    Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.

    [`SpeechT5Processor`] offers all the functionalities of [`SpeechT5FeatureExtractor`] and [`SpeechT5Tokenizer`]. See
    the docstring of [`~SpeechT5Processor.__call__`] and [`~SpeechT5Processor.decode`] for more information.

    Args:
        feature_extractor (`SpeechT5FeatureExtractor`):
            An instance of [`SpeechT5FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "SpeechT5FeatureExtractor"
    tokenizer_class = "SpeechT5Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        Processes audio and text input, as well as audio and text targets.

        You can process audio by using the argument `audio`, or process audio targets by using the argument
        `audio_target`. This forwards the arguments to SpeechT5FeatureExtractor's
        [`~SpeechT5FeatureExtractor.__call__`].

        You can process text by using the argument `text`, or process text labels by using the argument `text_target`.
        This forwards the arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.__call__`].

        Valid input combinations are:

        - `text` only
        - `audio` only
        - `text_target` only
        - `audio_target` only
        - `text` and `audio_target`
        - `audio` and `audio_target`
        - `text` and `text_target`
        - `audio` and `text_target`

        Please refer to the docstring of the above two methods for more information.
        """
        audio = kwargs.pop("audio", None)
        text = kwargs.pop("text", None)
        text_target = kwargs.pop("text_target", None)
        audio_target = kwargs.pop("audio_target", None)
        sampling_rate = kwargs.pop("sampling_rate", None)

        if audio is not None and text is not None:
            raise ValueError(
                "Cannot process both `audio` and `text` inputs. Did you mean `audio_target` or `text_target`?"
            )
        if audio_target is not None and text_target is not None:
            raise ValueError(
                "Cannot process both `audio_target` and `text_target` inputs. Did you mean `audio` or `text`?"
            )
        if audio is None and audio_target is None and text is None and text_target is None:
            raise ValueError(
                "You need to specify either an `audio`, `audio_target`, `text`, or `text_target` input to process."
            )

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        elif text is not None:
            inputs = self.tokenizer(text, **kwargs)
        else:
            inputs = None

        if audio_target is not None:
            audio_target_features = self.feature_extractor(
                audio_target=audio_target, *args, sampling_rate=sampling_rate, **kwargs
            )
            if inputs is None:
                return audio_target_features
            else:
                inputs["labels"] = audio_target_features["input_values"]
                inputs["stop_labels"] = audio_target_features["stop_labels"]
                decoder_attention_mask = audio_target_features.get("attention_mask")
                if decoder_attention_mask is not None:
                    inputs["decoder_attention_mask"] = decoder_attention_mask

        if text_target is not None:
            encodings_target = self.tokenizer(text_target, **kwargs)
            if inputs is None:
                return encodings_target
            else:
                inputs["labels"] = encodings_target["input_ids"]
                decoder_attention_mask = encodings_target.get("attention_mask")
                if decoder_attention_mask is not None:
                    inputs["decoder_attention_mask"] = decoder_attention_mask

        return inputs

    def pad(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5FeatureExtractor's [`~SpeechT5FeatureExtractor.pad`] and
        returns its output.

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
