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

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ...utils import auto_docstring


class Wav2Vec2BertProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class Wav2Vec2BertProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[Wav2Vec2BertProcessorKwargs],
    ):
        r"""
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_features** -- Audio input features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which timestamps should be attended to by the model when `audio` is not `None`.
            When only `text` is specified, returns the token attention mask.
            - **labels** -- List of token ids to be fed to a model. Returned when both `text` and `audio` are not `None`.
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None` and `audio` is `None`.
        """

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")
        output_kwargs = self._merge_kwargs(
            Wav2Vec2BertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, input_features=None, labels=None, **kwargs):
        """
        If `input_features` is not `None`, this method forwards the `input_features` and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.pad`] to pad the input features.
        If `labels` is not `None`, this method forwards the `labels` and `kwargs` arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.pad`] to pad the label(s).
        Please refer to the docstring of the above two methods for more information.
        """
        if input_features is None and labels is None:
            raise ValueError("You need to specify either an `input_features` or `labels` input to pad.")

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    @property
    def model_input_names(self):
        # The processor doesn't return text ids and the model seems to not need them
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels"]


__all__ = ["Wav2Vec2BertProcessor"]
