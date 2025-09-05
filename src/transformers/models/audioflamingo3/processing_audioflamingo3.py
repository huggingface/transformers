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
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput


class AudioFlamingo3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,  # pad to longest in the batch
            "truncation": True,  # safety: clip overlong prompts
        },
        "audio_kwargs": {},
    }


class AudioFlamingo3Processor(ProcessorMixin):
    r"""
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3 tokenizer into a single processor.

    [`AudioFlamingo3Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~AudioFlamingo3Processor.__call__`] and [`~AudioFlamingo3Processor.decode`] for more information.

    Args:
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def _tokenize_conversation(
        self,
        messages: Sequence[dict[str, str]],
        add_generation_prompt: bool = False,
        overrides: Optional[dict[str, str]] = None,
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

    def _get_num_windows(self, T: int, sr: int) -> tuple[int, int]:
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
    ) -> Optional[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
        """
        Returns 3 flat lists (one entry per window):
          - input_features      : [ (num_mel_bins, T_mel), ... ]
          - feature_attn_masks  : [ (T_mel,), ... ]               # mel-frame masks for tower attention
          - embed_len_masks     : [ (750,), ... ]                 # token-length masks for LLM fusion
        """
        if audio_data is None:
            return None

        wl = int(window_length * sample_rate)
        wo = int(window_overlap * sample_rate)

        feats_per_win: list[torch.Tensor] = []
        feat_masks: list[torch.Tensor] = []
        embed_masks: list[torch.Tensor] = []

        T = len(audio_data)
        audio = audio_data.reshape(1, -1)

        # convert to float32 in a stable way (keeps your original normalization)
        def int16_to_float32(x):
            return (x / 32767.0).astype(np.float32)

        def float32_to_int16(x):
            return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

        audio_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()

        num_windows, _ = self._get_num_windows(T, sample_rate)

        for i in range(num_windows):
            start = i * (wl - wo)
            chunk = audio_tensor[:, start : start + wl]  # (1, N_samples_window)
            orig_len = chunk.shape[1]

            # Whisper FE -> (1, num_mel_bins, T_mel), take [0] to drop batch dim
            fe = self.feature_extractor(chunk.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt")
            mel = fe["input_features"][0]  # (M, T_mel)
            feats_per_win.append(mel)

            # mel-frame mask length (before the conv downsamples in the tower)
            melspec_frames = int(math.ceil(orig_len / 160))  # 160 samples per mel frame @16kHz
            fm = torch.zeros(3000, dtype=torch.int32)
            fm[:melspec_frames] = 1
            feat_masks.append(fm)

            # embedding-length mask after the tower's conv downsampling
            conv_lengths = (melspec_frames - 1) // 2 + 1
            out_len = (conv_lengths - 2) // 2 + 1
            em = torch.zeros(750, dtype=torch.int32)
            em[:out_len] = 1
            embed_masks.append(em)

        return feats_per_win, feat_masks, embed_masks

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Union[np.ndarray, list[np.ndarray]],
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Batched processing:
          - `text`: str or list[str]
          - `audio`: np.ndarray or list[np.ndarray] (one audio per text sample)
        Returns a BatchFeature with:
          - input_ids          : (B, L)
          - attention_mask     : (B, L)
          - audio_features     : List[Tensor (M, T_mel)], flattened across the whole batch (row-major)
          - audio_feature_masks: List[Tensor (T_mel,)]
          - audio_embed_masks  : List[Tensor (750,)]
        """
        # normalize inputs to lists
        if isinstance(text, str):
            text = [text]
        if isinstance(audio, np.ndarray):
            audio = [audio]
        if not isinstance(text, list) or not isinstance(audio, list):
            raise ValueError("`text` and `audio` must be str/np.ndarray or lists of them.")
        if len(text) != len(audio):
            raise ValueError(f"Got {len(text)} texts but {len(audio)} audios.")

        output_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Build per-sample prompts and flatten windowed audio across the batch
        final_texts: list[str] = []
        audio_features_all: list[torch.Tensor] = []
        audio_feat_masks_all: list[torch.Tensor] = []
        audio_embed_masks_all: list[torch.Tensor] = []

        for t, a in zip(text, audio):
            loaded = self._load_sound_mask(a)
            if loaded is None:
                # no audio for this sample: keep text as-is (no <sound> injection)
                final_texts.append(t.strip())
                continue

            feats_per_win, feat_masks, embed_masks = loaded
            # IMPORTANT: one <sound> **per window**. They will be merged in the model using embed_masks.
            num_windows = len(feats_per_win)

            # If user inserted <sound> manually, drop it and prepend the correct count
            clean_t = t.replace("<sound>", "").strip()
            final_texts.append(("<sound>" * num_windows) + clean_t)

            # Flatten (row-major order): all windows of sample_0, then sample_1, ...
            audio_features_all.extend(list(feats_per_win))  # each f: (M, T_mel)
            audio_feat_masks_all.extend(list(feat_masks))  # each m: (T_mel,)
            audio_embed_masks_all.extend(list(embed_masks))  # each m: (750,)

        # Tokenize all prompts as a batch using the chat template
        convs = [[{"role": "user", "content": txt}] for txt in final_texts]
        prompts = [self.tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
        enc = self.tokenizer(
            prompts,
            padding=output_kwargs["text_kwargs"].get("padding", True),
            truncation=output_kwargs["text_kwargs"].get("truncation", True),
            return_tensors="pt",
        )

        return BatchFeature(
            data={
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "audio_features": audio_features_all,
                "audio_feature_masks": audio_feat_masks_all,
                "audio_embed_masks": audio_embed_masks_all,
            }
        )

    def decode(self, token_ids: torch.Tensor) -> Union[str, list[str]]:
        out = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in token_ids]
        return out[0] if len(out) == 1 else out


__all__ = ["AudioFlamingo3Processor"]
