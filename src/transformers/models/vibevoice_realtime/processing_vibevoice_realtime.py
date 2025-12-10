# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Optional, Union

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_soundfile_available, is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class VibeVoiceRealTimeProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
            "return_attention_mask": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VibeVoiceRealTimeProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps [`VibeVoiceFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~VibeVoiceRealTimeProcessor.__call__`] for more information.

    Args:
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
    """

    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        if not hasattr(tokenizer, "speech_start_token"):
            self.speech_start_token = "<|vision_start|>"
            self.speech_start_id = tokenizer.convert_tokens_to_ids(self.speech_start_token)
        else:
            self.speech_start_token = tokenizer.speech_start_token
            self.speech_start_id = tokenizer.speech_start_id

        if not hasattr(tokenizer, "speech_end_token"):
            self.speech_end_token = "<|vision_end|>"
            self.speech_end_id = tokenizer.convert_tokens_to_ids(self.speech_end_token)
        else:
            self.speech_end_token = tokenizer.speech_end_token
            self.speech_end_id = tokenizer.speech_end_id

        if not hasattr(tokenizer, "speech_diffusion_token"):
            self.speech_diffusion_token = "<|vision_pad|>"
            self.speech_diffusion_id = tokenizer.convert_tokens_to_ids(self.speech_diffusion_token)
        else:
            self.speech_diffusion_token = tokenizer.speech_diffusion_token
            self.speech_diffusion_id = tokenizer.speech_diffusion_id

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        preset: Optional[Union[str, list[str]]] = None,
        **kwargs: Unpack[VibeVoiceRealTimeProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process text inputs with optional voice preset argument.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to tokenizer.
            preset (`str`, `List[str]`, *optional*):
                Preset(s) to set the voice for the generated audio.
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            `BatchFeature`: A BatchFeature with the following fields:
                - **input_ids** -- Token ID sequences ready for the model
                - **attention_mask** -- Attention masks for the sequences
                - **input_features** -- Processed audio tensors (if preset provided)
                - **input_features_mask** -- Masks for valid speech tokens (if preset provided)
        """
        output_kwargs = self._merge_kwargs(
            VibeVoiceRealTimeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise ValueError("text input must be a string or list of strings")
        n_audio_in_text = [sample.count(self.speech_diffusion_token) for sample in text]

        n_audio = 0
        if audio is not None:
            audio = make_list_of_audio(audio)
            n_audio = len(audio)

        if sum(n_audio_in_text) > 0 and n_audio != sum(n_audio_in_text):
            if audio is None:
                raise ValueError("No audio were provided, but there are audio tokens in the prompt")
            else:
                raise ValueError(
                    f"The number of audio tokens in each text ({n_audio_in_text}) should be the same as the "
                    f"number of provided audios ({n_audio})."
                )

        data = {}
        encoding = self.tokenizer(text, **text_kwargs)
        data.update(encoding)

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_audio(
        self,
        audio: AudioInput,
        output_path: Optional[str] = None,
    ) -> list[str]:
        """
        Save audio data to WAV file(s).
        TODO eventually move to AudioProcessor base class.

        Args:
            audio: Audio output from the model to be saved
            output_path: Output file path or directory for multiple files

        Returns:
            List[str]: Paths to the saved audio files.
        """
        sampling_rate = self.feature_extractor.sampling_rate

        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        audio = make_list_of_audio(audio)
        for idx, item in enumerate(audio):
            audio[idx] = item.detach().cpu().numpy().squeeze()

        if len(audio) == 1:
            if output_path is None:
                output_path = "vibevoice_output.wav"
            sf.write(output_path, audio[0], sampling_rate)
            return [output_path]
        else:
            if output_path is None:
                output_path = "vibevoice_outputs"
            os.makedirs(output_path, exist_ok=True)
            saved_paths = []
            for i, audio_array in enumerate(audio):
                file_path = os.path.join(output_path, f"audio_{i}.wav")
                sf.write(file_path, audio_array, sampling_rate)
                saved_paths.append(file_path)
        return saved_paths


__all__ = ["VibeVoiceRealTimeProcessor"]
