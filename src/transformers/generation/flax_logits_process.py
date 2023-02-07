# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team
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

import inspect

import jax
import jax.lax as lax
import jax.numpy as jnp

from ..utils import add_start_docstrings
from ..utils.logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs:
            Additional logits processor specific kwargs.

    Return:
        `jnp.ndarray` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class FlaxLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to subsequently process
    a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = scores / self.temperature
        return scores


class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])

        mask_scores = jnp.full_like(scores, self.filter_value)
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        score_mask = cumulative_probs < self.top_p

        # include the token that is higher than top_p as well
        score_mask = jnp.roll(score_mask, 1)
        score_mask |= score_mask.at[:, 0].set(True)

        # min tokens to keep
        score_mask = score_mask.at[:, : self.min_tokens_to_keep].set(True)

        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
        next_scores = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]

        return next_scores


class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        batch_size, vocab_size = scores.shape
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)

        topk = min(self.top_k, scores.shape[-1])  # Safety check
        topk_scores, topk_indices = lax.top_k(scores, topk)
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift

        next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
        next_scores = next_scores_flat.reshape(batch_size, vocab_size)
        return next_scores


class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - 1)

        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)

        return scores


class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)

        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)

        return scores


class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # create boolean flag to decide if min length penalty should be applied
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)

        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores)

        return scores
