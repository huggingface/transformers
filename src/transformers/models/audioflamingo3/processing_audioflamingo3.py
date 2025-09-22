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
Processor class for AudioFlamingo3.
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
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sound_token": "<sound>",  # Placeholder token used in text for audio expansion.
            "return_attention_mask": True,
            "padding": "max_length",
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class AudioFlamingo3Processor(ProcessorMixin):
    r"""
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3
    tokenizer into a single processor.

    [`AudioFlamingo3Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~AudioFlamingo3Processor.__call__`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Main method to prepare one or several text sequence(s) and audio waveform(s) for the model. This
        method expands `<sound>` placeholders in the text based on the post-pool frame counts of the
        audio windows, then tokenizes the provided strings as-is, and extracts log-mel features
        with [`WhisperFeatureExtractor`].

        Args:
            text (`str` or `list[str]`):
                Input sequence or batch of sequences.
            audio (`np.ndarray` or `list[np.ndarray]`):
                Input audio or batch of audios as NumPy arrays. If provided, there must be as many `text` inputs as
                `audio` inputs.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `feature_attention_mask`).
        """

        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
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

            # Post-pool frame counts per window (match encoder schedule)
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
                total_tokens = sum(frames_per_window[w_ptr : w_ptr + n_win])
                w_ptr += n_win

                sample = t
                n_placeholders = sample.count(sound_token)

                if n_placeholders == 1:
                    # Single placeholder: expand it with the total number of tokens needed
                    sample = sample.replace(sound_token, sound_token * total_tokens, 1)
                elif n_placeholders == 0:
                    # No placeholders: insert tokens based on text format
                    prefix = sound_token * total_tokens

                    # Check if it's a chat template format
                    user_start = "<|im_start|>user\n"
                    if user_start in sample:
                        # Chat template: insert after user start
                        sample = sample.replace(user_start, user_start + prefix, 1)
                    else:
                        # Plain text: prepend to the beginning
                        sample = prefix + sample
                else:
                    # Multiple placeholders not supported for simplicity
                    raise ValueError(
                        f"Sample {idx}: found {n_placeholders} '{sound_token}' placeholders. "
                        f"Expected exactly 1 or 0 placeholders."
                    )

                expanded_texts.append(sample)

        # Tokenize
        text_inputs = self.tokenizer(expanded_texts, **text_kwargs)

        return BatchFeature(data={**text_inputs, **audio_inputs}, tensor_type=return_tensors)


__all__ = ["AudioFlamingo3Processor"]
