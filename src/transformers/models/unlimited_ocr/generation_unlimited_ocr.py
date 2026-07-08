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
from ...generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING, NoRepeatNGramLogitsProcessor
from ...utils import add_start_docstrings


class UnlimitedOcrGenerationConfig(GenerationConfig):
    r"""A GenerationConfig class with parameterization customized for UnlimitedOcr.

    Args:
        no_repeat_ngram_window_size (`int`, *optional*):
            If set together with `no_repeat_ngram_size`, n-gram repetitions are blocked only within this many
            trailing tokens instead of over the whole sequence.
    """

    def __init__(self, no_repeat_ngram_window_size: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.no_repeat_ngram_window_size = no_repeat_ngram_window_size


class UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(NoRepeatNGramLogitsProcessor):
    r"""Identical to [`NoRepeatNGramLogitsProcessor`] but blocks n-gram repetitions only within the last
    `window_size` generated tokens, rather than the full sequence.

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once in `window_size`.
        window_size (`int`):
            Number of trailing tokens to search for repeated n-grams.
    """

    def __init__(self, ngram_size: int, window_size: int):
        super().__init__(ngram_size=ngram_size)
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.window_size = window_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return super().__call__(input_ids[:, -self.window_size :], scores)


__all__ = ["UnlimitedOcrGenerationConfig", "UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor"]
