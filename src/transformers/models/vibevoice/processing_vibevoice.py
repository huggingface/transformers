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


class VibeVoiceProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
            "return_attention_mask": True,
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
            "padding": True,
            "return_attention_mask": True,
            "pad_to_multiple_of": 3200,  # acoustic_tokenizer.hop_length
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VibeVoiceProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps [`VibeVoiceFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~VibeVoiceProcessor.__call__`] for more information.

    Args:
        feature_extractor (`VibeVoiceFeatureExtractor`):
            The feature extractor for speech processing.
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
    """

    feature_extractor_class = "VibeVoiceFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(self, feature_extractor, tokenizer, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

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
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[VibeVoiceProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process text inputs with optional voice samples.

        This method processes text inputs (typically prepared by apply_chat_template) and optional voice samples for
        voice cloning. It expands speech diffusion tokens based on the actual audio length.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process, typically prepared by apply_chat_template with speech token placeholders.
            audio (`List[Union[str, np.ndarray]]`, *optional*):
                Audio samples for speaker voice cloning. Should match the number of speech token placeholders in text.
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            `BatchFeature`: A BatchFeature with the following fields:
                - **input_ids** -- Token ID sequences ready for the model
                - **attention_mask** -- Attention masks for the sequences
                - **input_features** -- Processed audio tensors (if audio provided)
                - **input_features_mask** -- Masks for valid speech tokens (if audio provided)
        """
        output_kwargs = self._merge_kwargs(
            VibeVoiceProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
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
        if audio is not None:
            audio = make_list_of_audio(audio)
            data = self.feature_extractor(audio, **audio_kwargs)

            # Create mask for audio tokenizer based on compression ratio
            padding_masks = data["input_features_mask"]
            speech_tok_compress_ratio = int(audio_kwargs["pad_to_multiple_of"])
            num_audio_tokens_list = torch.ceil(padding_masks.sum(dim=-1) / speech_tok_compress_ratio).int().tolist()
            input_features_mask = torch.zeros((len(padding_masks), max(num_audio_tokens_list)), dtype=torch.bool)
            for i, seq_len in enumerate(num_audio_tokens_list):
                input_features_mask[i, :seq_len] = True
            data["input_features_mask"] = input_features_mask

            # expand the text to repeat the audio token for the corresponding number of frames
            num_audio_tokens_list_copy = num_audio_tokens_list.copy()
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.speech_diffusion_token in sample:
                    num_speech_tokens = num_audio_tokens_list_copy.pop(0)
                    expanded_speech_token = self.speech_diffusion_token * num_speech_tokens

                    replace_str.append(expanded_speech_token)
                    sample = sample.replace(self.speech_diffusion_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)

            text = expanded_text

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
            audio[idx] = item.detach().cpu().float().numpy().squeeze()

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


__all__ = ["VibeVoiceProcessor"]
