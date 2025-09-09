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
            "padding": True,  # Pad to longest sequence in the batch
            "truncation": True,  # Truncate overlong prompts for safety
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
        - input_ids, attention_mask : (B, L)
        - audio_features            : (N, M, T_mel)                 or None
        - audio_feature_masks       : (N, nb_max_frames)            or None   # Feature extractor attention masks (mel-frame)
        - audio_features_mask       : (N, S_max+1)                  or None   # Encoder mask with reserved end token
        """
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

        fe = self.feature_extractor
        sr = int(getattr(fe, "sampling_rate", 16_000))
        hop = int(getattr(fe, "hop_length", 160))
        n_samples = int(getattr(fe, "n_samples", int(30.0 * sr)))
        nb_max_frames = int(getattr(fe, "nb_max_frames", math.ceil(n_samples / hop)))

        # Frontend downsampling (conv k=3,p=1,s=2 → pool k=2,s=2)
        length_after_conv2_max = (nb_max_frames - 1) // 2 + 1
        tokens_per_window_max = (length_after_conv2_max - 2) // 2 + 1
        encoder_mask_len = tokens_per_window_max + 1  # Reserve one additional slot for end token

        # Audio windowing configuration: 30 second windows with no overlap by default
        wl = n_samples
        wo = 0

        final_texts: list[str] = []
        feats_all: list[torch.Tensor] = []
        feat_masks_all: list[torch.Tensor] = []
        token_masks_all: list[torch.Tensor] = []

        # Feature extraction call arguments with mask and padding enforced
        fe_kwargs = dict(output_kwargs.get("audio_kwargs", {}))
        fe_kwargs.update(
            {
                "return_attention_mask": True,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "sampling_rate": sr,
            }
        )

        for t, a in zip(text, audio):
            if a.ndim != 1:
                a = np.asarray(a).reshape(-1)
            T = a.shape[0]

            if T <= wl:
                num_windows = 1
            elif T >= (20 * wl - 19 * wo):  # Limit to maximum of 20 windows
                num_windows = 20
            else:
                num_windows = 1 + int(math.ceil((T - wl) / float(wl - wo)))

            # Construct prompt with appropriate number of <sound> tokens
            clean_t = t.replace("<sound>", "").strip()
            final_texts.append(("<sound>" * num_windows) + clean_t)

            for i in range(num_windows):
                start = i * (wl - wo)
                chunk = a[start : start + wl]
                # Extract features and mel-frame mask
                out = fe(chunk.reshape(1, -1), **fe_kwargs)
                mel = out["input_features"][0]  # (M, T_mel)
                fm = out["attention_mask"][0].to(torch.int32)  # (nb_max_frames,)

                feats_all.append(mel)
                feat_masks_all.append(fm)

                # Calculate encoder output length for this window from mel length
                melspec_frames = int(fm.sum().item())
                l1 = (melspec_frames - 1) // 2 + 1
                out_len = max(0, (l1 - 2) // 2 + 1)

                tm = torch.zeros(encoder_mask_len, dtype=torch.bool)
                tm[: min(out_len, tokens_per_window_max)] = True  # Reserve last slot for end token
                token_masks_all.append(tm)

        if len(feats_all) > 0:
            audio_features = torch.stack(feats_all, dim=0)  # (N, M, T_mel)
            audio_feature_masks = torch.stack(feat_masks_all, 0)  # (N, nb_max_frames)
            audio_features_mask = torch.stack(token_masks_all, 0)  # (N, S_max+1)
        else:
            audio_features = None
            audio_feature_masks = None
            audio_features_mask = None

        # Tokenize text prompts (single-turn user message)
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
                "audio_features": audio_features,
                "audio_feature_masks": audio_feature_masks,  # Feature extractor attention masks (mel-frame)
                "audio_features_mask": audio_features_mask,  # Encoder mask with end token
            }
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> list[str]:
        """
        Batch text decoding functionality.
        Returns a list[str] of decoded strings, one per sequence.
        """
        output_kwargs = self._merge_kwargs(
            AudioFlamingo3ProcessorKwargs,
            **kwargs,
        )
        text_kwargs = dict(output_kwargs.get("text_kwargs", {}))
        text_kwargs.setdefault("skip_special_tokens", True)

        decoded = self.tokenizer.batch_decode(token_ids, **text_kwargs)
        return [s.strip() for s in decoded]

    def decode(
        self,
        token_ids: torch.Tensor,
        **kwargs: Unpack[AudioFlamingo3ProcessorKwargs],
    ) -> str:
        """
        Single-sample text decoding.
        Enforces a batch size of 1 and delegates to `batch_decode`.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.shape[0] != 1:
            raise ValueError(f"Expected a single sequence for `decode`, but got batch size {token_ids.shape[0]}. " "Use `batch_decode` for multi-sample outputs.")

        return self.batch_decode(token_ids, **kwargs)[0]


__all__ = ["AudioFlamingo3Processor"]
