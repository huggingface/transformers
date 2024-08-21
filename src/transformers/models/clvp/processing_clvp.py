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

import sys
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class ClvpAudioProcessorKwargs(AudioKwargs, total=False):
    raw_speech: Optional[AudioInput]


class ClvpProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: ClvpAudioProcessorKwargs
    _defaults = {}


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
    model_input_names = [
        "input_ids",
        "input_features",
        "attention_mask",
    ]

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio: Optional[AudioInput] = None,
        images=None,
        videos=None,
        **kwargs: Unpack[ClvpProcessorKwargs],
    ) -> BatchFeature:
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`
        argument to [`~ClvpTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`AudioInput`, *optional*):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_features** -- Audio features to be fed to a model. Returned when `audios` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            ClvpProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        raw_speech = output_kwargs["audio_kwargs"].pop("raw_speech", None)

        if audio is not None and raw_speech is not None:
            raise ValueError("Only one of `audio` and `raw_speech` must be specified.")
        if audio is None and raw_speech is not None:
            audio = raw_speech

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        data = {}
        if audio is not None:
            audio_features = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            data.update(audio_features)
        if text is not None:
            text_features = self.tokenizer(text, **output_kwargs["text_kwargs"])
            if audio is not None:
                data["input_ids"] = text_features["input_ids"]
                data["attention_mask"] = text_features["attention_mask"]
            else:
                data.update(text_features)
        return BatchFeature(data, tensor_type=output_kwargs["common_kwargs"].get("return_tensors"))

    # Copied from transformers.models.whisper.processing_whisper.WhisperProcessor.batch_decode with Whisper->Clvp
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.whisper.processing_whisper.WhisperProcessor.decode with Whisper->Clvp
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
