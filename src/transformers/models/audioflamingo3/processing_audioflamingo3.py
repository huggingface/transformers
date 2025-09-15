# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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
Processor for AudioFlamingo3.

Behaviors:
- Splits each raw waveform into fixed-length windows using the feature extractor's
  `chunk_length` (seconds) and `sampling_rate`.
- For each window, computes the post-pool frame count K (matching the encoder's
  conv/pool schedule) and expands the audio placeholder token K times.
- If a sample contains no placeholders, all expanded tokens for that sample are
  prepended to the text before tokenization.
- Returns a `BatchFeature` containing text tokenization and audio features.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class AudioFlamingo3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            # Tokenizer pads to the longest in the batch by default.
            "padding": True,
        },
        "audio_kwargs": {
            # Placeholder token used in text for audio expansion.
            "sound_token": "<sound>",
        },
    }


class AudioFlamingo3Processor(ProcessorMixin):
    """
    AudioFlamingo3 processor that wraps a Whisper-style feature extractor and a tokenizer.

    Expected placeholder flow per sample:
      text: "... <sound> ...", audio: raw waveform (1-D np.ndarray)
      - audio is split into N windows (based on chunk_length & sampling_rate)
      - for each window i, compute post-pool frames K_i
      - if the text contains exactly N `<sound>` placeholders, replace each with K_i copies
        of `<sound>` in order; if the text has no `<sound>`, prepend all expanded tokens.

    This ensures a 1:1 alignment between `<sound>` token positions in the text and
    post-pool audio frames produced by the encoder.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Union[np.ndarray, list[np.ndarray]],
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        # Capture desired tensor type for BatchFeature (so `.to(device)` works later)
        tensor_type = kwargs.pop("tensor_type", None)

        # -----------------------
        # Normalize & validate IO
        # -----------------------
        if isinstance(text, str):
            texts: list[str] = [text]
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            texts = text
        else:
            raise ValueError("`text` must be a str or list[str].")

        if isinstance(audio, np.ndarray):
            audios: list[np.ndarray] = [audio]
        elif isinstance(audio, list) and all(isinstance(a, np.ndarray) for a in audio):
            audios = audio
        else:
            raise ValueError("`audio` must be a np.ndarray or list[np.ndarray].")

        if len(texts) != len(audios):
            raise ValueError(f"Got {len(texts)} texts but {len(audios)} audios; they must match 1:1.")

        # Merge defaults with user kwargs (and tokenizer init defaults)
        call_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        sound_token: str = call_kwargs["audio_kwargs"].pop("sound_token")

        # -----------------------
        # Window planning per sample
        # -----------------------
        if not hasattr(self.feature_extractor, "chunk_length") or not hasattr(self.feature_extractor, "sampling_rate"):
            raise AttributeError("Feature extractor must expose `chunk_length` (sec) and `sampling_rate` (Hz).")

        max_seconds = float(getattr(self.feature_extractor, "chunk_length"))
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate"))
        window_size = int(max_seconds * sampling_rate)

        CAP_WINDOWS = 20  # 10 minutes, since chunk_length=30s

        per_sample_windows: list[int] = []
        flat_chunks: list[np.ndarray] = []

        for wav in audios:
            total = int(wav.shape[0])
            n_win = max(1, (total + window_size - 1) // window_size)
            if n_win > CAP_WINDOWS:
                logger.warning(
                    f"Audio duration ({total / sampling_rate:.1f}s) exceeds 600s; truncating to first 10 minutes."
                )
                n_win = CAP_WINDOWS
            per_sample_windows.append(n_win)

            T_cap = min(total, n_win * window_size)
            for i in range(n_win):
                s = i * window_size
                e = min((i + 1) * window_size, T_cap)
                flat_chunks.append(wav[s:e])

        # -----------------------
        # Feature extraction (audio)
        # -----------------------
        # Ensure fixed shape with attention mask; avoid key collision with text mask.
        call_kwargs["audio_kwargs"]["return_attention_mask"] = True
        call_kwargs["audio_kwargs"]["padding"] = "max_length"

        audio_inputs = self.feature_extractor(
            flat_chunks,
            sampling_rate=sampling_rate,
            **call_kwargs["audio_kwargs"],
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        # -----------------------
        # Post-pool frame counts per window (match encoder schedule)
        # -----------------------
        # feature_attention_mask: (num_windows, T_mel_pad)
        # Conv stack:    L1 = (L_mel - 1)//2 + 1
        # AvgPool(2,2):  K  = (L1 - 2)//2 + 1
        feat_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()
        frames_per_window: list[int] = []
        for L_mel in feat_lengths:
            L1 = (int(L_mel) - 1) // 2 + 1
            K = (L1 - 2) // 2 + 1
            frames_per_window.append(max(1, K))

        # -----------------------
        # Expand text per sample
        # -----------------------
        expanded_texts: list[str] = []
        w_ptr = 0
        for idx, t in enumerate(texts):
            n_win = per_sample_windows[idx]
            Ks = frames_per_window[w_ptr : w_ptr + n_win]
            w_ptr += n_win

            sample = t
            n_placeholders = sample.count(sound_token)

            if n_placeholders and n_placeholders != n_win:
                raise ValueError(
                    f"Sample {idx}: found {n_placeholders} '{sound_token}' placeholders, "
                    f"but audio was split into {n_win} window(s)."
                )

            if n_placeholders == 0:
                # No placeholders: prepend all expanded tokens
                prefix = "".join(sound_token * k for k in Ks)
                sample = prefix + sample
            else:
                # Replace each placeholder in order with k repeated tokens
                for k in Ks:
                    sample = sample.replace(sound_token, sound_token * k, 1)

            expanded_texts.append(sample)

        # -----------------------
        # Tokenize with chat template (single user turn + generation prompt)
        # -----------------------
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": txt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for txt in expanded_texts
        ]
        text_inputs = self.tokenizer(prompts, **call_kwargs["text_kwargs"])

        # -----------------------
        # Pack and return
        # -----------------------
        text_inputs.update(audio_inputs)
        return BatchFeature(data=text_inputs, tensor_type=tensor_type)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        fea_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tok_names + fea_names + ["feature_attention_mask"]))


__all__ = ["AudioFlamingo3Processor"]
