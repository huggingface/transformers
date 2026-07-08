# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import torch

from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import LogitsProcessor


class UnlimitedOcrGenerationConfig(GenerationConfig):
    r"""
    A GenerationConfig class with parameterization customized for UnlimitedOcr.

    Args:
        no_repeat_ngram_window_size (`int`, *optional*):
            If set together with `no_repeat_ngram_size`, n-gram repetitions are blocked only within this many
            trailing tokens instead of over the whole sequence.
    """

    def __init__(self, no_repeat_ngram_window_size: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.no_repeat_ngram_window_size = no_repeat_ngram_window_size


class UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that blocks n-gram repetitions within a sliding window over the most recently generated
    tokens, rather than the full sequence. Aligned with SGLang's `DeepseekOCRNoRepeatNGramLogitProcessor`.

    Args:
        no_repeat_ngram_size (`int`):
            Size of the n-grams that are not allowed to repeat.
        no_repeat_ngram_window_size (`int`):
            Number of trailing tokens to search for repeated n-grams.
    """

    def __init__(self, no_repeat_ngram_size: int, no_repeat_ngram_window_size: int):
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.no_repeat_ngram_window_size = no_repeat_ngram_window_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.no_repeat_ngram_size:
                continue
            search_start = max(0, len(sequence) - self.no_repeat_ngram_window_size)
            search_end = len(sequence) - self.no_repeat_ngram_size + 1
            if search_end <= search_start:
                continue
            if self.no_repeat_ngram_size > 1:
                current_prefix = tuple(sequence[-(self.no_repeat_ngram_size - 1) :])
            else:
                current_prefix = ()
            banned = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + self.no_repeat_ngram_size]
                if self.no_repeat_ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned.add(ngram[-1])
            for token_id in banned:
                scores[batch_idx, token_id] = float("-inf")
        return scores


__all__ = ["UnlimitedOcrGenerationConfig", "UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor"]
