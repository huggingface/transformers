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
Audio/Text processor class for SeamlessM4T
"""

import sys
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class SeamlessM4TProcessorAudioKwargs(TextKwargs, total=False):
    do_normalize_per_mel_bins: Optional[bool]


class SeamlessM4TProcessorTextKwargs(TextKwargs, total=False):
    src_lang: Optional[str]
    tgt_lang: Optional[str]


class SeamlessM4TProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: SeamlessM4TProcessorTextKwargs
    audio_kwargs: SeamlessM4TProcessorAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "pad_to_multiple_of": 2,
        },
        "audio_kwargs": {
            "do_normalize_per_mel_bins": True,
            "padding": True,
            "pad_to_multiple_of": 2,
            "truncation": True,
        },
    }


class SeamlessM4TProcessor(ProcessorMixin):
    r"""
    Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
    single processor.

    [`SeamlessM4TProcessor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`SeamlessM4TTokenizerFast`]. See the [`~SeamlessM4TProcessor.__call__`] and [`~SeamlessM4TProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`SeamlessM4TFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`SeamlessM4TTokenizerFast`]):
            The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audios: Optional[AudioInput] = None,
        images=None,
        videos=None,
        **kwargs: Unpack[SeamlessM4TProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to SeamlessM4TTokenizerFast's [`~SeamlessM4TTokenizerFast.__call__`] if `text` is not
        `None` to encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audios` is not `None`. Please refer
        to the doctsring of the above two methods for more information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`AudioInput`, *optional*):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.
        """

        if text is None and audios is None:
            raise ValueError("You have to specify either text or audios. Both cannot be none.")
        elif text is not None and audios is not None:
            raise ValueError(
                "Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another."
            )

        output_kwargs = self._merge_kwargs(
            SeamlessM4TProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        data = {}
        if text is not None:
            text_features = self.tokenizer(text, **output_kwargs["text_kwargs"])
            data.update(text_features)
        if audios is not None:
            audio_features = self.feature_extractor(audios, **output_kwargs["audio_kwargs"])
            data.update(audio_features)
        return BatchFeature(data, tensor_type=output_kwargs["common_kwargs"].get("return_tensors"))

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))
