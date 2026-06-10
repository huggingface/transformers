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

from typing import Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_python import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires_backends


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


@auto_docstring
class GraniteSpeechProcessor(ProcessorMixin):
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

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        return self.audio_token * audio_inputs["audio_embed_sizes"][audio_idx]

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        audio: Union["torch.Tensor", list["torch.Tensor"]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        requires_backends(self, ["torch"])

        text = self._get_validated_text(text)
        prompt_strings = text

        if audio is not None:
            # NOTE - we intentionally avoid throwing for potentially misaligned
            # text / audio inputs here because some inference engines will
            # trigger the conditions due to the way they call multimodal
            # processors, e.g., vLLM.
            audio_inputs = self.audio_processor(audio, device=device)

            # TODO (@alex-jw-brooks); we should add a util to get_num_audio_tokens
            # from feature lengths and call it here, rather than returning it
            # from the feature extractor.
            # Expand the audio placeholders to match the feature dims; this
            # is similar to how many VLMs handle image tokens, e.g., llava next
            audio_replacements = [
                self.replace_audio_token(audio_inputs, i) for i in range(len(audio_inputs["audio_embed_sizes"]))
            ]
            audio_inputs.pop("audio_embed_sizes")
            # `get_text_with_replacements` mutates the list in place, so copy to avoid editing the caller's input
            prompt_strings, _ = self.get_text_with_replacements(list(text), audio_replacements=audio_replacements)
        else:
            audio_inputs = {}

        if "padding" not in kwargs:
            kwargs["padding"] = True
        text_inputs = self.tokenizer(prompt_strings, **kwargs)
        return BatchFeature(data={**text_inputs, **audio_inputs})

    def _get_validated_text(self, text: str | list) -> list[str]:
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list) and isinstance(text[0], str):
            return text
        raise TypeError("Invalid text provided! Text should be a string or list of strings.")


__all__ = ["GraniteSpeechProcessor"]
