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
"""Processor class for Granite Speech."""

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_python import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class GraniteSpeechProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "device": "cpu",
        },
    }


@auto_docstring
class GraniteSpeechProcessor(ProcessorMixin):
    valid_processor_kwargs = GraniteSpeechProcessorKwargs

    def __init__(
        self,
        audio_processor,
        tokenizer,
        audio_token="<|audio|>",
        chat_template=None,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The special token used to represent audio in the text sequence. This token serves as a placeholder
            that will be replaced with multiple audio tokens based on the actual audio length. The number of
            audio tokens inserted depends on the audio feature dimensions extracted by the audio processor.
        """
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        super().__init__(audio_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        audio: AudioInput | None = None,
        device: str = "cpu",
        **kwargs: Unpack[GraniteSpeechProcessorKwargs],
    ) -> BatchFeature:
        text = self._validate_inputs(text)
        kwargs.setdefault("audio_kwargs", {}).setdefault("device", device)
        return super().__call__(text=text, audio=audio, **kwargs)

    def _validate_inputs(self, text: str | list) -> list[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")

    def _process_audio(self, audio: AudioInput, **kwargs):
        # Audio samples are already collated
        if len(audio) == 1 and audio[0].ndim == 2:
            audio = audio[0]
        audio_inputs = self.audio_processor(audio, device=kwargs["device"])

        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        num_audio_tokens = audio_inputs["audio_embed_sizes"][audio_idx]
        return self.audio_token * num_audio_tokens

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["audio_embed_sizes"]


__all__ = ["GraniteSpeechProcessor"]
