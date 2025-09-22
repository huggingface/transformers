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

from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import logging


logger = logging.get_logger(__name__)

MAX_AUDIO_LEN = 10 * 60  # 10 minutes


class AudioFlamingo3ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            # Tokenizer pads to the longest in the batch by default.
            "padding": True,
        },
        "audio_kwargs": {
            "sound_token": "<sound>",  # Placeholder token used in text for audio expansion.
            "return_attention_mask": True,
            "padding": "max_length",
        },
        "common_kwargs": {"return_tensors": "pt"},
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

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        # Merge defaults with user kwargs (and tokenizer init defaults)
        call_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,  # TODO keep?
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        common_kwargs = call_kwargs["common_kwargs"]
        return_tensors = common_kwargs.pop("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        # Handle text
        if isinstance(text, str):
            texts: list[str] = [text]
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            texts = text
        else:
            raise ValueError("`text` must be a str or list[str].")

        # Handle audio
        audio_inputs = {}
        if audio is not None:
            audios = make_list_of_audio(audio)
            if len(texts) != len(audios):
                raise ValueError(f"Got {len(texts)} texts but {len(audios)} audios; they must match 1:1.")
            sound_token: str = audio_kwargs.pop("sound_token")

            # Determine number of chunks per sample, and flatten
            if not hasattr(self.feature_extractor, "chunk_length") or not hasattr(
                self.feature_extractor, "sampling_rate"
            ):
                raise AttributeError("Feature extractor must expose `chunk_length` (sec) and `sampling_rate` (Hz).")

            max_seconds = float(getattr(self.feature_extractor, "chunk_length"))
            sampling_rate = int(getattr(self.feature_extractor, "sampling_rate"))
            window_size = int(max_seconds * sampling_rate)
            max_windows = int(MAX_AUDIO_LEN // max_seconds)

            per_sample_windows: list[int] = []
            flat_chunks: list[np.ndarray] = []

            for wav in audios:
                total = int(wav.shape[0])
                n_win = max(1, (total + window_size - 1) // window_size)
                if n_win > max_windows:
                    logger.warning(
                        f"Audio duration ({total / sampling_rate:.1f}s) exceeds {MAX_AUDIO_LEN}s; truncating to first {MAX_AUDIO_LEN}s."
                    )
                    n_win = max_windows
                per_sample_windows.append(n_win)

                T_cap = min(total, n_win * window_size)
                for i in range(n_win):
                    start = i * window_size
                    end = min((i + 1) * window_size, T_cap)
                    flat_chunks.append(wav[start:end])
            # feature extraction
            audio_inputs = self.feature_extractor(
                flat_chunks,
                sampling_rate=sampling_rate,
                **audio_kwargs,
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

            # Expand text per sample
            expanded_texts: list[str] = []
            w_ptr = 0
            for idx, t in enumerate(texts):
                n_win = per_sample_windows[idx]
                Ks = frames_per_window[w_ptr : w_ptr + n_win]
                w_ptr += n_win

                sample = t
                n_placeholders = sample.count(sound_token)

                if n_placeholders != 1:
                    raise ValueError(
                        f"Sample {idx}: expected exactly 1 '{sound_token}' in the (already templated) text. "
                        "Place it where audio should appear; the processor will expand it."
                    )
                sample = sample.replace(sound_token, sound_token * sum(Ks), 1)

                expanded_texts.append(sample)

        # tokenize
        text_inputs = self.tokenizer(expanded_texts, **text_kwargs)

        return BatchFeature(data={**text_inputs, **audio_inputs}, tensor_type=return_tensors)


__all__ = ["AudioFlamingo3Processor"]
