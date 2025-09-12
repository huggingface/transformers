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
from typing import Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class AudioFlamingo3Processor(ProcessorMixin):
    """
    Constructs an AudioFlamingo3 processor which wraps an AudioFlamingo3 feature extractor and an AudioFlamingo3 tokenizer into a single processor.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        audio: Union[np.ndarray, list[np.ndarray]],
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        # Normalize inputs
        if isinstance(text, str):
            text = [text]
        if isinstance(audio, np.ndarray):
            audio = [audio]
        if not (isinstance(text, list) and isinstance(audio, list)):
            raise ValueError("`text` and `audio` must be str/np.ndarray or lists of them.")
        if len(text) != len(audio):
            raise ValueError(f"Got {len(text)} texts but {len(audio)} audios.")

        # Window planning (single pass)
        max_seconds = float(getattr(self.feature_extractor, "chunk_length"))
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate"))
        n_samples = int(max_seconds * sampling_rate)
        CAP_WINDOWS = 20  # 10 minutes at 30s windows

        final_texts: list[str] = []
        audio_chunks: list[np.ndarray] = []

        for t, a in zip(text, audio):
            T = a.shape[0]
            n_win = max(1, int(math.ceil(T / n_samples)))
            if n_win > CAP_WINDOWS:
                logger.warning(f"Audio duration ({T/sampling_rate:.1f}s) exceeds maximum supported length (600s). " "Audio will be truncated to first 10 minutes.")
                n_win = CAP_WINDOWS
            final_texts.append("<sound>" * n_win + t)

            T_cap = min(T, n_win * n_samples)
            for i in range(n_win):
                s, e = i * n_samples, min((i + 1) * n_samples, T_cap)
                audio_chunks.append(a[s:e])

        # Single batched FE call
        audio_inputs = self.feature_extractor(
            audio_chunks,
            return_attention_mask=getattr(self.feature_extractor, "return_attention_mask"),
            sampling_rate=sampling_rate,
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        # Tokenize expanded prompts (single call)
        convs = [[{"role": "user", "content": txt}] for txt in final_texts]
        prompts = [self.tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
        inputs = self.tokenizer(prompts, padding=True)

        inputs.update(audio_inputs)
        return BatchFeature(data=inputs, tensor_type=kwargs.pop("tensor_type", None))


__all__ = ["AudioFlamingo3Processor"]
