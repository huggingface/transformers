# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class GraniteSpeech5ProcessorKwargs(ProcessingKwargs, total=False):
    # the defaults this model needs are shipped with the checkpoint rather than hardcoded here: the audio
    # ones are the feature extractor's own signature defaults, and the tokenizer's `padding` comes from
    # `tokenizer_config.json` (`ProcessorMixin._merge_kwargs` reads the tokenizer's init kwargs)
    _defaults = {}


@auto_docstring
class GraniteSpeech5Processor(ProcessorMixin):
    valid_processor_kwargs = GraniteSpeech5ProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        sampling_rate: int | None = None,
        **kwargs: Unpack[GraniteSpeech5ProcessorKwargs],
    ):
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate expected by the feature
            extractor (defaults to 16000 Hz). If provided, it will be validated against the processor's expected
            sampling rate, and an error will be raised if they don't match. If not provided, a warning will be
            issued and the default sampling rate will be assumed.
        """
        output_kwargs = self._merge_kwargs(
            GraniteSpeech5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        expected_sampling_rate = self.feature_extractor.sampling_rate
        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be {expected_sampling_rate}, which can result in silent errors."
            )
        elif sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({expected_sampling_rate}). Please provide resampled the audio to the expected sampling rate."
            )
        output_kwargs["audio_kwargs"]["sampling_rate"] = expected_sampling_rate

        model_inputs = super().__call__(audio=audio, text=text, **output_kwargs)
        if text is not None:
            model_inputs["labels"] = model_inputs.pop("input_ids")
        return model_inputs

    @property
    def model_input_names(self):
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return feature_extractor_input_names + ["labels"]


__all__ = ["GraniteSpeech5Processor"]
