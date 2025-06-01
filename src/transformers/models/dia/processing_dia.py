# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Speech processor class for Dia
"""

from ...processing_utils import ProcessorMixin
from ..auto import AutoModel


class DiaProcessor(ProcessorMixin):
    r"""
    Constructs a Dia processor which wraps a Dia feature extractor and a Dia tokenizer into a single
    processor.

    [`DiaProcessor`] offers all the functionalities of [`DiaAudioProcessor`] and [`DiaTokenizer`]. See
    the [`~DiaProcessor.__call__`] and [`~DiaProcessor.decode`] for more information.

    Args:
        feature_extractor (`DiaAudioProcessor`):
            An instance of [`DiaAudioProcessor`]. The feature extractor is a required input.
        tokenizer (`DiaTokenizer`):
            An instance of [`DiaTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "DiaTokenizer"

    def __init__(self, feature_extractor, tokenizer, audio_model="descript/dac_44khz"):
        super().__init__(feature_extractor, tokenizer)
        self.audio_tokenizer = AutoModel.from_pretrained(audio_model)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to DiaAudioProcessor's [`~DiaAudioProcessor.__call__`] and the `text`
        argument to [`~DiaTokenizer.__call__`]. Please refer to the docstring of the above two methods for more
        information.
        """
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            input_audio = self.feature_extractor(
                audio, *args, sampling_rate=sampling_rate, return_tensors="pt", **kwargs
            )
            audio_tokens = self.audio_tokenizer.encode(input_audio["input_values"], **kwargs)
            inputs = {"input_values": input_audio["input_values"], "audio_tokens": audio_tokens.audio_codes}

        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs).audio_values

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DiaTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.audio_tokenizer.decode(*args, **kwargs).audio_values


__all__ = ["DiaProcessor"]
