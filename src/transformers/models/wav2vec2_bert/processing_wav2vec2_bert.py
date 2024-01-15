# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Speech processor class for Wav2Vec2-BERT
"""
import warnings

from ...processing_utils import ProcessorMixin
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class Wav2Vec2BertProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2-BERT processor which wraps a Wav2Vec2-BERT feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            An instance of [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
        and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audio` is not
        `None` to pre-process the audio. To prepare the target sequences(s), this method forwards the `text` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not `None`. Please refer to the doctsring of the above two methods for more information.
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
        This method forwards the `input_features` and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.pad`] if `input_features` is not
        `None` to pad the input features. To pad the label(s), this method forwards the `labels` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.pad`] if `labels` is not `None`. Please refer to the doctsring of the above two methods for more information.
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
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
