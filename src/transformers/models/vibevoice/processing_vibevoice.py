# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc. team. All rights reserved.
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
"""Processor class for VibeVoice."""

import json
import os
import re
from typing import Any, Optional, Union

import numpy as np
import torch

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging

logger = logging.get_logger(__name__)


class AudioNormalizer:
    def __init__(self, target_dB_FS=-25, eps=1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def __call__(self, wav):
        rms = np.sqrt(np.mean(wav**2))
        if rms < self.eps:
            return wav
        target_linear = 10 ** (self.target_dB_FS / 20)
        gain = target_linear / rms
        return wav * gain


class VibeVoiceTokenizerProcessor:
    """
    Simple audio processor for VibeVoice that handles normalization.
    Acts as a feature extractor equivalent.
    """

    def __init__(self, sampling_rate=24000, normalize_audio=True, target_dB_FS=-25, eps=1e-6):
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps
        self.audio_normalizer = AudioNormalizer(target_dB_FS, eps) if normalize_audio else None

    def _load_audio_from_path(self, path):
        # Placeholder for audio loading - VibeVoice implementation relies on external lib or user providing array
        # This function is used by processor if string path is passed.
        # We will require numpy/scipy/librosa here only if called.
        try:
            import librosa

            wav, _ = librosa.load(path, sr=self.sampling_rate)
            return wav
        except ImportError:
            raise ImportError("librosa is required to load audio from files. Please install it.")

    def __call__(self, audio, **kwargs):
        # Basic processing if needed directly
        return audio


class VibeVoiceProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice processor which wraps a VibeVoice tokenizer and audio processor into a single processor.
    """

    attributes = ["tokenizer", "audio_processor"]
    tokenizer_class = ("VibeVoiceTokenizer", "VibeVoiceTokenizerFast")

    def __init__(
        self, tokenizer=None, audio_processor=None, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs
    ):
        super().__init__(tokenizer, audio_processor)
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None
        self.system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

    def __call__(
        self,
        text: Optional[
            Union[str, list[str], TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]
        ] = None,
        voice_samples: Optional[Union[list[Union[str, np.ndarray]], list[list[Union[str, np.ndarray]]]]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Handle single vs batch input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and not isinstance(text[0], str)):
            texts = [text]
            is_batched = False
        else:
            texts = text
            is_batched = True

        # Handle voice samples
        if voice_samples is not None:
            if not is_batched or (isinstance(voice_samples[0], (str, np.ndarray))):
                voice_samples_list = [voice_samples]
            else:
                voice_samples_list = voice_samples
        else:
            voice_samples_list = [None] * len(texts)

        all_encodings = []
        for text_input, voice_input in zip(texts, voice_samples_list):
            encoding = self._process_single(text_input, voice_input)
            all_encodings.append(encoding)

        batch_encoding = self._batch_encode(
            all_encodings,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
        )

        return batch_encoding

    def _process_single(
        self,
        text: Union[str, TextInput],
        voice_samples: Optional[list[Union[str, np.ndarray]]] = None,
    ) -> dict[str, Any]:
        script = None
        if isinstance(text, str):
            if text.endswith(".json") and os.path.exists(text):
                script = self._convert_json_to_script(text)
            elif text.endswith(".txt") and os.path.exists(text):
                script = self._convert_text_to_script(text)
            else:
                script = text

        if script is None:
            raise ValueError(f"Could not process input text: {text}")

        parsed_lines = self._parse_script(script)
        all_speakers = list({speaker_id for speaker_id, _ in parsed_lines})

        system_tokens = self.tokenizer.encode(self.system_prompt)

        if voice_samples:
            voice_tokens, voice_speech_inputs, voice_speech_masks = self._create_voice_prompt(
                voice_samples[: len(all_speakers)]
            )
        else:
            voice_tokens, voice_speech_inputs, voice_speech_masks = [], [], []

        full_tokens = system_tokens + voice_tokens
        speech_input_mask = [False] * len(system_tokens) + voice_speech_masks

        full_tokens += self.tokenizer.encode(" Text input:\n", add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(" Text input:\n", add_special_tokens=False))

        for speaker_id, speaker_text in parsed_lines:
            speaker_text_tokens = self.tokenizer.encode(
                f" Speaker {speaker_id}:{speaker_text}\n", add_special_tokens=False
            )
            full_tokens += speaker_text_tokens
            speech_input_mask += [False] * len(speaker_text_tokens)

        full_tokens += self.tokenizer.encode(" Speech output:\n", add_special_tokens=False) + [
            self.tokenizer.speech_start_id
        ]
        speech_input_mask += [False] * (len(self.tokenizer.encode(" Speech output:\n", add_special_tokens=False)) + 1)

        return {
            "input_ids": full_tokens,
            "speech_inputs": voice_speech_inputs if voice_speech_inputs else None,
            "speech_input_mask": speech_input_mask,
            "parsed_script": parsed_lines,
            "all_speakers": all_speakers,
        }

    def _parse_script(self, script: str) -> list[tuple[int, str]]:
        lines = script.strip().split("\n")
        parsed_lines = []
        speaker_ids = []

        for line in lines:
            if not line.strip():
                continue
            match = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line.strip(), re.IGNORECASE)

            if match:
                speaker_id = int(match.group(1))
                text = " " + match.group(2).strip()
                parsed_lines.append((speaker_id, text))
                speaker_ids.append(speaker_id)
            else:
                logger.warning(f"Could not parse line: '{line}'")

        if not parsed_lines:
            raise ValueError("No valid speaker lines found in script")

        min_speaker_id = min(speaker_ids)
        if min_speaker_id > 0:
            normalized_lines = []
            for speaker_id, text in parsed_lines:
                normalized_lines.append((speaker_id - 1, text))
            return normalized_lines
        else:
            return parsed_lines

    def _create_voice_prompt(
        self, speaker_samples: list[Union[str, np.ndarray]]
    ) -> tuple[list[int], list[np.ndarray], list[bool]]:
        vae_token_id = self.tokenizer.speech_diffusion_id
        voice_full_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
        voice_speech_inputs = []
        voice_speech_masks = [False] * len(voice_full_tokens)

        for speaker_id, speaker_audio in enumerate(speaker_samples):
            prefix_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:", add_special_tokens=False)

            if isinstance(speaker_audio, str):
                wav = self.audio_processor._load_audio_from_path(speaker_audio)
            else:
                wav = np.array(speaker_audio, dtype=np.float32)

            if self.db_normalize and self.audio_normalizer:
                wav = self.audio_normalizer(wav)

            vae_tok_len = math.ceil(wav.shape[0] / self.speech_tok_compress_ratio)

            speaker_tokens = (
                prefix_tokens
                + [self.tokenizer.speech_start_id]
                + [vae_token_id] * vae_tok_len
                + [self.tokenizer.speech_end_id]
                + self.tokenizer.encode("\n", add_special_tokens=False)
            )

            vae_input_mask = [False] * len(prefix_tokens) + [False] + [True] * vae_tok_len + [False] + [False]

            voice_full_tokens.extend(speaker_tokens)
            voice_speech_masks.extend(vae_input_mask)
            voice_speech_inputs.append(wav)

        return voice_full_tokens, voice_speech_inputs, voice_speech_masks

    def prepare_speech_inputs(
        self,
        speech_inputs: list[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, Any]:
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}

        vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs]
        max_speech_length = max(s.shape[0] for s in speech_inputs)

        if speech_inputs[0].ndim == 1:
            padded_speeches = np.full((len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32)
        else:
            padded_speeches = np.full(
                (len(speech_inputs), max_speech_length, speech_inputs[0].shape[-1]), fill_value=0, dtype=np.float32
            )
        speech_masks = np.zeros((len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_)

        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, : len(speech)] = speech
            speech_masks[i, :vae_tok_length] = True

        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }

        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(padded_speeches, device=device, dtype=dtype or torch.float32)
            result["speech_masks"] = torch.tensor(speech_masks, device=device, dtype=torch.bool)

        return result

    def _batch_encode(
        self,
        encodings: list[dict[str, Any]],
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: bool = True,
    ) -> BatchEncoding:
        input_ids_list = [enc["input_ids"] for enc in encodings]
        speech_input_masks_list = [enc["speech_input_mask"] for enc in encodings]

        if isinstance(padding, bool):
            padding_strategy = PaddingStrategy.LONGEST if padding else PaddingStrategy.DO_NOT_PAD
        elif isinstance(padding, str):
            padding_strategy = PaddingStrategy(padding)
        else:
            padding_strategy = padding

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            if padding_strategy == PaddingStrategy.LONGEST:
                max_len = max(len(ids) for ids in input_ids_list)
            elif padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
                max_len = max_length
            else:
                max_len = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            attention_masks = []
            padded_speech_input_masks = []

            for input_ids, speech_mask in zip(input_ids_list, speech_input_masks_list):
                if truncation and len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    speech_mask = speech_mask[:max_len]

                padding_length = max_len - len(input_ids)
                padded_ids = [self.tokenizer.pad_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + [1] * len(input_ids)
                padded_speech_mask = [False] * padding_length + speech_mask

                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
                padded_speech_input_masks.append(padded_speech_mask)

            input_ids_list = padded_input_ids
            speech_input_masks_list = padded_speech_input_masks
        else:
            attention_masks = [[1] * len(ids) for ids in input_ids_list] if return_attention_mask else None

        all_speech_inputs = []
        has_speech = False
        for enc in encodings:
            if enc["speech_inputs"] is not None:
                all_speech_inputs.extend(enc["speech_inputs"])
                has_speech = True

        batch_encoding = BatchEncoding()

        if return_tensors is not None:
            batch_encoding["input_ids"] = torch.tensor(input_ids_list, dtype=torch.long)
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            batch_encoding["speech_input_mask"] = torch.tensor(speech_input_masks_list, dtype=torch.bool)
        else:
            batch_encoding["input_ids"] = input_ids_list
            if return_attention_mask and attention_masks is not None:
                batch_encoding["attention_mask"] = attention_masks
            batch_encoding["speech_input_mask"] = speech_input_masks_list

        if has_speech:
            speech_dict = self.prepare_speech_inputs(
                all_speech_inputs,
                return_tensors=return_tensors,
            )
            batch_encoding["speech_tensors"] = speech_dict["padded_speeches"]
            batch_encoding["speech_masks"] = speech_dict["speech_masks"]
        else:
            batch_encoding["speech_tensors"] = None
            batch_encoding["speech_masks"] = None

        batch_encoding["parsed_scripts"] = [enc["parsed_script"] for enc in encodings]
        batch_encoding["all_speakers_list"] = [enc["all_speakers"] for enc in encodings]

        return batch_encoding

    def _convert_json_to_script(self, json_file: str) -> str:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON must be list")
        lines = []
        for item in data:
            if isinstance(item, dict) and "speaker" in item and "text" in item:
                lines.append(f"Speaker {item['speaker']}: {item['text']}")
        return "\n".join(lines)

    def _convert_text_to_script(self, text_file: str) -> str:
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Simplified
        return "".join(lines)
