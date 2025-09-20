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
            "padding": True,
            "add_generation_prompt": True,
        },
        "audio_kwargs": {
            "sound_token": "<sound>",
        },
    }


class AudioFlamingo3Processor(ProcessorMixin):
    r"""
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3 tokenizer into a single processor.

    [`AudioFlamingo3Processor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~AudioFlamingo3Processor.__call__`] and [`~AudioFlamingo3Processor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Union[np.ndarray, list[np.ndarray]],
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Main method to prepare one or several text sequence(s) and audio waveform(s) for the model. This
        method expands `<sound>` placeholders in the text based on the post-pool frame counts of the
        audio windows, applies the tokenizer's chat template to the text, and extracts log-mel features
        with [`WhisperFeatureExtractor`].

        Args:
            text (`str` or `list[str]`):
                Input sequence or batch of sequences. Must match 1:1 with `audio`.
            audio (`np.ndarray` or `list[np.ndarray]`):
                Input audio or batch of audios as NumPy arrays.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `feature_attention_mask`).
        """

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
        if not sound_token or not isinstance(sound_token, str):
            raise ValueError("`sound_token` must be a non-empty string.")

        # -----------------------
        # Window planning per sample
        # -----------------------
        if (
            not hasattr(self.feature_extractor, "chunk_length")
            or not hasattr(self.feature_extractor, "sampling_rate")
            or not hasattr(self.feature_extractor, "return_attention_mask")
        ):
            raise AttributeError(
                "Feature extractor must expose `chunk_length` (sec), `sampling_rate` (Hz), and `return_attention_mask` (bool) attributes."
            )

        max_seconds = float(getattr(self.feature_extractor, "chunk_length"))
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate"))
        return_attention_mask = bool(getattr(self.feature_extractor, "return_attention_mask"))
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
        audio_inputs = self.feature_extractor(
            flat_chunks,
            sampling_rate=sampling_rate,
            return_attention_mask=return_attention_mask,
            **call_kwargs["audio_kwargs"],
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        # -----------------------
        # Post-pool frame counts per window (match encoder schedule)
        # -----------------------
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
        # Tokenize with chat template
        # -----------------------
        add_generation_prompt = bool(call_kwargs["text_kwargs"].pop("add_generation_prompt", True))
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": txt}],
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            for txt in expanded_texts
        ]
        padding_side = kwargs.pop("padding_side", None)
        text_inputs = self.tokenizer(
            prompts,
            padding_side=padding_side,
            **call_kwargs["text_kwargs"],
        )

        # -----------------------
        # Pack and return
        # -----------------------
        text_inputs.update(audio_inputs)
        tensor_type = kwargs.pop("tensor_type", None)
        return BatchFeature(data=text_inputs, tensor_type=tensor_type)

    @property
    def model_input_names(self) -> list[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))


__all__ = ["AudioFlamingo3Processor"]
