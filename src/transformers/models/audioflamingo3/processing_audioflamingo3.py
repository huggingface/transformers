# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and The HuggingFace Inc. team.
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
Processor class for AudioFlamingo3.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput


class AudioFlamingo3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
    }


class AudioFlamingo3Processor(ProcessorMixin):
    r"""
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3 tokenizer into a single processor.

    [`AudioFlamingo3Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~AudioFlamingo3Processor.__call__`] and [`~AudioFlamingo3Processor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`AutoTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def _tokenize_conversation(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = False,
        overrides: Optional[Dict[str, str]] = None,
        no_system_prompt: bool = False,
    ) -> torch.Tensor:
        # Normalize conversation before tokenization
        for message in messages:
            message["value"] = message["value"].strip()

        conversation = []
        for m in messages:
            message = {}
            if m["from"] == "human":
                message["role"] = "user"
            elif m["from"] == "gpt":
                message["role"] = "assistant"
            else:
                raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

            message["content"] = m["value"]
            if overrides is not None and m["from"] in overrides:
                message["content"] = overrides[m["from"]]
            conversation.append(message)

        if no_system_prompt:
            conversation = [{"role": "system", "content": ""}] + conversation

        text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        return self.tokenizer(text, return_tensors="pt").input_ids[0]

    def _get_num_windows(self, T: int, sr: int) -> Tuple[int, int]:
        window_length = int(30.0 * sr)
        window_overlap = int(0.0 * sr)
        max_num_window = 20
        num_windows = 1
        if T <= window_length:
            num_windows = 1
            full_length = window_length
        elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
            num_windows = max_num_window
            full_length = max_num_window * window_length - (max_num_window - 1) * window_overlap
        else:
            num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
            full_length = num_windows * window_length - (num_windows - 1) * window_overlap

        return num_windows, full_length

    def _load_sound_mask(
        self,
        audio_data: Optional[np.ndarray],
        sample_rate: int = 16000,
        window_length: float = 30.0,
        window_overlap: float = 0.0,
        max_num_window: int = 20,
    ) -> Optional[Tuple[List[List[List[float]]], torch.Tensor, torch.Tensor]]:
        if audio_data is None:
            return None
        window_length = int(window_length * sample_rate)
        window_overlap = int(window_overlap * sample_rate)
        max_num_window = int(max_num_window)

        sound_outputs = []
        audio_feature_masks = []
        audio_embed_masks = []

        T = len(audio_data)
        audio_data = audio_data.reshape(1, -1)
        num_windows = self._get_num_windows(T, sample_rate)[0]

        int16_to_float32 = lambda x: (x / 32767.0).astype(np.float32)
        float32_to_int16 = lambda x: (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        for i in range(num_windows):
            audio_embed_mask = torch.zeros(750)
            start = i * (window_length - window_overlap)
            audio_data_tensor_this = audio_data_tensor[:, start : start + window_length]
            orig_length = audio_data_tensor_this.shape[1]
            audio_data_tensor_this = self.feature_extractor(audio_data_tensor_this.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")
            sound_outputs.append(audio_data_tensor_this["input_features"])
            # Mask for the input mel-spectrogram to Whisper
            melspec_frames_this_window = int(math.ceil(orig_length / 160))
            feature_attention_mask = torch.zeros(3000, dtype=torch.int32)
            feature_attention_mask[:melspec_frames_this_window] = 1
            audio_feature_masks.append(feature_attention_mask.unsqueeze(0))
            # Mask for the output embedding used in AF3
            conv_lengths = (melspec_frames_this_window - 1) // 2 + 1
            output_embedding_lengths = (conv_lengths - 2) // 2 + 1
            audio_embed_mask[:output_embedding_lengths] = 1
            audio_embed_masks.append(audio_embed_mask)

        sound_outputs = torch.stack(sound_outputs, dim=0)
        audio_feature_masks = torch.stack(audio_feature_masks, dim=0)
        audio_embed_masks = torch.stack(audio_embed_masks, dim=0)
        return sound_outputs.numpy().tolist(), audio_feature_masks, audio_embed_masks

    def __call__(
        self,
        text: TextInput,
        audio: np.ndarray,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to AutoTokenizer's [`~AutoTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`):
                The sequence to be encoded. Can be a string.
            audio (`np.ndarray`):
                The audio to be prepared. Should be a NumPy array.

        Returns:
            [`BatchFeature`]: A BatchFeature containing:
                - input_ids: Tokenized input IDs
                - media: List of processed audio tensors
                - audio_feature_masks: List of attention masks for audio features
                - audio_embed_masks: List of embedding masks for audio features
        """
        audio_features = []

        final_text = ""
        sound, audio_feature_masks, audio_embed_masks = self._load_sound_mask(audio)
        audio_features.append(sound)
        audio_feature_masks_dict = [audio_feature_masks]
        audio_embed_masks_dict = [audio_embed_masks]
        final_text += "<sound>" * len(sound)
        final_text += text.replace("<sound>", "").strip()

        conversation = [{"from": "human", "value": final_text}]
        input_ids = self._tokenize_conversation(conversation, add_generation_prompt=True).unsqueeze(0)

        sounds = torch.tensor(audio_features).half()
        audio_features = [sound for sound in sounds]
        audio_feature_masks = audio_feature_masks_dict[0].detach().clone().half()
        audio_feature_masks_dict = [sound_mask for sound_mask in audio_feature_masks]
        audio_embed_masks = audio_embed_masks_dict[0].detach().clone().half()
        audio_embed_masks_dict = [sound_mask for sound_mask in audio_embed_masks]

        return BatchFeature(data={"input_ids": input_ids, "audio_features": audio_features, "audio_feature_masks": audio_feature_masks_dict, "audio_embed_masks": audio_embed_masks_dict})

    def decode(self, token_ids: torch.Tensor) -> str:
        result = [self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in token_ids]
        return result[0] if len(result) == 1 else result


__all__ = ["AudioFlamingo3Processor"]
