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
from collections.abc import Callable
from typing import Any

import torch

from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
)
from ...generation.utils import GenerationMixin
from ...utils import add_start_docstrings


class UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(NoRepeatNGramLogitsProcessor):
    r"""
    Identical to [`NoRepeatNGramLogitsProcessor`] but blocks n-gram repetitions only within the last
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


class UnlimitedOcrGenerationMixin(GenerationMixin):
    r"""
    Adds support for the `no_repeat_ngram_window_size` generation option. If set together with `no_repeat_ngram_size`,
    n-gram repetitions are blocked only within this many trailing tokens instead of over the whole sequence.

    `no_repeat_ngram_window_size` is a model specific option and must already be present on the generation config
    before it can be passed to `generate`, either through the checkpoint's `generation_config.json` or by setting
    `model.generation_config.no_repeat_ngram_window_size` after loading the model.
    """

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int | None = None,
        encoder_input_ids: torch.LongTensor | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        logits_processor: LogitsProcessorList | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
    ) -> LogitsProcessorList:
        no_repeat_ngram_size = generation_config.no_repeat_ngram_size
        no_repeat_ngram_window_size = getattr(generation_config, "no_repeat_ngram_window_size", None)
        use_sliding_window_processor = False

        if no_repeat_ngram_window_size is not None and no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            use_sliding_window_processor = True
            logits_processor = LogitsProcessorList(logits_processor or [])
            logits_processor.append(
                UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(
                    ngram_size=no_repeat_ngram_size,
                    window_size=no_repeat_ngram_window_size,
                )
            )
            # Set to None to avoid also adding the default `NoRepeatNGramLogitsProcessor`
            generation_config.no_repeat_ngram_size = None

        try:
            processors = super()._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=encoder_input_ids,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                device=device,
                model_kwargs=model_kwargs,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
            )
        finally:
            if use_sliding_window_processor:
                generation_config.no_repeat_ngram_size = no_repeat_ngram_size
        return processors
