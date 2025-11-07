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
import re
from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
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
            "speech_tok_compress_ratio": 3200,  # acoustic_tokenizer.hop_length
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

        # Handle speech tokens like CSM
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

        # Fixed text parts used in building text sequences
        self.system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        voice_samples: Optional[AudioInput] = None,
        **kwargs: Unpack[VibeVoiceProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process one or more podcast scripts with optional voice samples.

        This method processes text scripts with speaker annotations and optional voice samples for voice cloning.
        It builds complete text sequences with embedded speech tokens and creates speech input masks to identify
        where speech tokens are located in the sequence.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process. Each text should be a script with speaker annotations in the format:
                "Speaker 1: Hello world\nSpeaker 2: How are you?"
                Can be:
                - A single script string
                - A list of script strings for batch processing
            voice_samples (`List[Union[str, np.ndarray]]`, `List[List[Union[str, np.ndarray]]]`, *optional*):
                Voice samples for speaker voice cloning. Order should match speaker IDs appearance in script. Can be:
                - A list of audio samples for a single script
                - A list of lists of audio samples for batch processing
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            `BatchFeature`: A BatchFeature with the following fields:
                - **input_ids** -- Token ID sequences ready for the model
                - **attention_mask** -- Attention masks for the sequences
                - **input_features** -- Processed audio tensors (if voice_samples provided)
                - **input_features_mask** -- Masks for valid speech tokens (if voice_samples provided)
                - **audio_select_mask** -- Mask indicating which audio samples are used by each script (if voice_samples provided)
        """
        # Merge defaults with user kwargs
        output_kwargs = self._merge_kwargs(
            VibeVoiceProcessorKwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")
        speech_tok_compress_ratio = int(audio_kwargs.pop("speech_tok_compress_ratio"))

        # Handle text input validation
        if isinstance(text, str):
            texts = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        else:
            texts = text
        scripts = []
        for _text in texts:
            scripts.append(self.separate_script(_text))
        # -- Extract all speaker IDs from the script, and preserve order
        speakers_per_script = [list(dict.fromkeys(tup[0] for tup in _script)) for _script in scripts]
        # -- Get min value of all speaker IDs
        min_speaker_id = min([min(speakers) for speakers in speakers_per_script])
        if min_speaker_id < 0:
            raise ValueError(f"Speaker IDs must be non-negative integers, got min ID {min_speaker_id}.")
        if min_speaker_id > 0:
            # Normalize to start from 0
            speakers_per_script = [[spk - min_speaker_id for spk in speakers] for speakers in speakers_per_script]
            scripts = [[(spk - min_speaker_id, text) for spk, text in _script] for _script in scripts]

        # Handle voice samples
        processed_audio = {}
        if voice_samples is not None:
            voice_samples_list = [make_list_of_audio(_voices) for _voices in voice_samples]
            if len(texts) != len(voice_samples_list):
                raise ValueError(f"Got {len(texts)} texts but {len(voice_samples)} audio lists; they must match 1:1.")

            # check correct number of samples per script, and extract audio for unique speakers
            speaker_to_audio = {}
            for speakers, audios in zip(speakers_per_script, voice_samples_list):
                if len(speakers) != len(audios):
                    raise ValueError(f"Got {len(speakers)} speakers but {len(audios)} audio samples; they must match 1:1.")
                for _speaker, _audio in zip(speakers, audios):
                    if _speaker not in speaker_to_audio:
                        speaker_to_audio[_speaker] = _audio

            # Process audio samples
            voices = [voice for _voices in voice_samples_list for voice in _voices]
            processed_audio = self.feature_extractor(voices, **audio_kwargs)
            processed_audio["input_features"] = processed_audio.pop("audio")

            # Create speech masks for audio tokenizer based on its compression ratio
            padding_masks = processed_audio["padding_mask"]
            vae_tok_seqlens = torch.ceil(padding_masks.sum(dim=-1) / speech_tok_compress_ratio).int().tolist()
            input_features_mask = torch.zeros((len(padding_masks), max(vae_tok_seqlens)), dtype=torch.bool)
            for i, seq_len in enumerate(vae_tok_seqlens):
                input_features_mask[i, :seq_len] = True
            processed_audio["input_features_mask"] = input_features_mask
            del processed_audio["padding_mask"]

            # # TODO (ebezzam) not used by model, but could be used to process less audio by `get_audio_features`
            # # Create mask to know which audio is used by a particular script
            # audio_select_mask = np.zeros((len(scripts), len(speaker_to_audio)), dtype=np.bool_)
            # for i, speakers in enumerate(speakers_per_script):
            #     for spk in speakers:
            #         audio_select_mask[i, spk] = True
            # processed_audio["audio_select_mask"] = audio_select_mask
            processed_audio = BatchFeature(data=processed_audio, tensor_type=return_tensors)
            # Unsqueeze needed by tokenizer: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modeling_vibevoice_inference.py#L146
            processed_audio["input_features"] = processed_audio["input_features"].unsqueeze(1)

        # Build text sequences with placeholders for speech tokens
        text_sequences = []
        for i, _script in enumerate(scripts):
            # Start with system prompt
            text_parts = [self.system_prompt]

            # Add voice section if audio provided
            if processed_audio is not None:
                script_speakers = speakers_per_script[i]
                text_parts.append(' Voice input:\n')

                # Build speaker voice sections with placeholders
                vae_tok_lens = processed_audio["input_features_mask"][script_speakers].sum(dim=-1).int().tolist()
                for speaker_id, vae_tok_len in zip(script_speakers, vae_tok_lens):
                    # Use the actual speech tokens from the tokenizer
                    speech_placeholder = self.speech_diffusion_token * vae_tok_len
                    speaker_voice_text = f" Speaker {speaker_id}:{self.speech_start_token}{speech_placeholder}{self.speech_end_token}\n"
                    text_parts.append(speaker_voice_text)

            # Add text input section
            text_parts.append(' Text input:\n')

            # Add script text
            for speaker_id, speaker_text in _script:
                speaker_line = f" Speaker {speaker_id}:{speaker_text}\n"
                text_parts.append(speaker_line)

            # Add speech output section
            speech_output_text = f' Speech output:\n{self.speech_start_token}'
            text_parts.append(speech_output_text)

            # Join all text parts
            full_text = ''.join(text_parts)
            text_sequences.append(full_text)

        # Tokenize the complete text sequences
        batch_encoding = self.tokenizer(text_sequences, **text_kwargs)

        # Remove token_type_ids if present (VibeVoice doesn't use them)
        batch_encoding.pop("token_type_ids", None)

        # Add audio data if provided
        if processed_audio:
            batch_encoding.update(processed_audio)

        return BatchFeature(data=batch_encoding, tensor_type=return_tensors)

    # TODO (ebezzam) remove after properly using chat template
    def separate_script(self, script: str) -> list[tuple[int, str]]:
        """Separate script into list of (speaker_id, text) tuples."""
        lines = script.strip().split("\n")
        parsed_lines = []
        speaker_ids = []

        # Parse all lines and collect speaker IDs
        for line in lines:
            if not line.strip():
                continue

            # Use regex to handle edge cases like multiple colons
            match = re.match(r'^Speaker\s+(\d+)\s*:\s*(.*)$', line.strip(), re.IGNORECASE)

            if match:
                speaker_id = int(match.group(1))
                text = ' ' + match.group(2).strip()
                parsed_lines.append((speaker_id, text))
                speaker_ids.append(speaker_id)
            else:
                logger.warning(f"Could not parse line: '{line}'")

        if not parsed_lines:
            raise ValueError("No valid speaker lines found in script")

        return parsed_lines

    # TODO (ebezzam) remove?
    def _merge_inputs(self, text_inputs: BatchEncoding, audio_inputs: dict) -> BatchEncoding:
        """Merge text and audio inputs into a single BatchEncoding."""
        # Start with text inputs
        merged = BatchEncoding(text_inputs)

        # Add audio-specific fields
        if "audio" in audio_inputs:
            merged["speech_inputs"] = audio_inputs["audio"]
        if "streaming" in audio_inputs:
            merged["streaming"] = audio_inputs["streaming"]

        return merged

    @property
    def model_input_names(self):
        """
        Return the list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["input_features", "input_features_mask"]))

    def save_audio(
        self,
        audio: AudioInput,
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
    ) -> list[str]:
        """
        Save audio data to WAV file(s).
        
        Args:
            audio: Audio data to save (tensor, array, or list of them)
            output_path: Output file path or directory for multiple files
            sampling_rate: Sampling rate for the saved audio
                
        Returns:
            List[str]: Paths to the saved audio files.
        """
        if sampling_rate is None:
            sampling_rate = self.feature_extractor.sampling_rate

        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        # Convert various audio formats to list of 1D numpy arrays
        if isinstance(audio, list):
            # List of tensors/arrays
            audio_arrays = []
            for item in audio:
                if isinstance(item, torch.Tensor):
                    audio_arrays.append(item.detach().cpu().numpy().squeeze())
                else:
                    audio_arrays.append(np.array(item).squeeze())
        else:
            # Single tensor or array
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().to(torch.float32).cpu().numpy()
            else:
                audio_np = np.array(audio)

            # Handle different shapes
            if audio_np.ndim == 1:
                audio_arrays = [audio_np]
            elif audio_np.ndim == 2:
                # Could be (batch, time) or (channels, time)
                if audio_np.shape[0] <= 2:  # Assume channels if <= 2
                    audio_arrays = [audio_np.mean(axis=0)]  # Convert to mono
                else:  # Assume batch dimension
                    audio_arrays = [audio_np[i] for i in range(audio_np.shape[0])]
            elif audio_np.ndim == 3:
                # (batch, channels, time) - extract each item and convert to mono
                audio_arrays = [audio_np[i].mean(axis=0) for i in range(audio_np.shape[0])]
            else:
                raise ValueError(f"Unsupported audio shape: {audio_np.shape}")

        # Save audio(s)
        if len(audio_arrays) == 1:
            # Single audio file
            sf.write(output_path, audio_arrays[0], sampling_rate)
        else:
            # Multiple audio files - save to directory
            os.makedirs(output_path, exist_ok=True)
            saved_paths = []
            for i, audio_array in enumerate(audio_arrays):
                file_path = os.path.join(output_path, f"audio_{i}.wav")
                sf.write(file_path, audio_array, sampling_rate)
                saved_paths.append(file_path)


__all__ = ["VibeVoiceProcessor"]
