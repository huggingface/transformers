# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
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
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..utils import add_start_docstrings
from ..utils.logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")
        return scores


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Note that for decoder-only models, such as Llama2, `min_length` will compute the length of `prompt + newly
    generated tokens` whereas for other models it will behave as `min_new_tokens`, that is, taking only into account
    the newly generated ones.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # If the maximum length (default = 20) is smaller than the minimum length constraint, the latter is ignored!
    >>> outputs = model.generate(**inputs, min_new_tokens=30)
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    Hugging Face Company is a company that has been working on a new product for the past year.

    >>> # For testing purposes, let's set `eos_token` to `"company"`, the first generated token. This will make
    >>> # generation end there.
    >>> outputs = model.generate(**inputs, eos_token_id=1664)
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    Hugging Face Company is a company

    >>> # Increasing `min_new_tokens` will make generation ignore occurences `"company"` (eos token) before the
    >>> # minimum length condition is honored.
    >>> outputs = model.generate(**inputs, min_new_tokens=2, eos_token_id=1664)
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    Hugging Face Company is a new company
    ```
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int]]):
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all(isinstance(i, int) for i in eos_token_id) or any(i < 0 for i in eos_token_id):
            logger.warning(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        return scores


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens.

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a joint venture between GEO Group, one of',
    'Hugging Face Company is not an exact science – but what we believe does']

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the repetition of previous tokens through an exponential penalty. This technique
    shares some similarities with coverage mechanisms and other aimed at reducing repetition. During the text
    generation process, the probability distribution for the next token is determined using a formula that incorporates
    token scores based on their occurrence in the generated sequence. Tokens with higher scores are more likely to be
    selected. The formula can be seen in the original [paper](https://arxiv.org/pdf/1909.05858.pdf). According to the
    paper a penalty of around 1.2 yields a good balance between truthful generation and lack of repetition.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> # Initializing the model and tokenizer for it
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

    >>> # This shows a normal generate without any specific parameters
    >>> summary_ids = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'm going to be able to do that

    >>> # This generates a penalty for repeated tokens
    >>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
    >>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'll just have to go out and play
    ```
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on tokens that are not in the original input.

    Args:
        hallucination_penalty (`float`):
            The parameter for hallucination penalty. 1.0 means no penalty.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.
    """

    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, self.encoder_input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, self.encoder_input_ids, score)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

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

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class EpsilonLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
    >>> # Top P sampling, which restricts tokens based on their cumulative probability.
    >>> # Pro tip: The paper recomends using `epsilon_cutoff` values between 3e-4 and 9e-4
    >>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Determine which indices to remove
        probabilities = scores.softmax(dim=-1)
        indices_to_remove = probabilities < self.epsilon

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class EtaLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
    cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
    the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
    min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
    samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
    Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample`
    must be set to `True` for this `LogitsWarper` to work.


    Args:
        epsilon (`float`):
            A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
            suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
            parameter is useful when logits need to be modified for very low probability tokens that should be excluded
            from generation entirely.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
            For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
            even if all tokens have probabilities below the cutoff `eta`.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)
    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

    >>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
    >>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
    >>> # Pro tip: The paper recomends using `eta_cutoff` values between 3e-4 to 4e-3
    >>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = torch.tensor(epsilon)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Calculate the adaptive cutoff
        probabilities = scores.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits=scores).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    <Tip>

    Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
    might lead to undesirable outcomes where the city's name appears only once in the entire text.
    [Reference](https://huggingface.co/blog/how-to-generate)

    </Tip>

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    >>> inputs = tokenizer(["Today I"], return_tensors="pt")

    >>> output = model.generate(**inputs)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I’m going to be able to do it.

    >>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I’m") in the output.
    >>> output = model.generate(**inputs, no_repeat_ngram_size=2)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I can get a better understanding of the nature of this issue
    ```
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of encoder input ids n-grams for the decoder ids. See
    [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350).

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores


class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`Dict[Tuple[int], float]`):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


    >>> def get_tokens_as_tuple(word):
    ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    """

    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        self.sequence_bias = sequence_bias
        self._validate_arguments()

        # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
        # is infered in the first usage, which inhibits initializing here)
        self.length_1_bias = None
        self.prepared_bias_variables = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - prepares an empty bias to add
        bias = torch.zeros_like(scores)

        # 3 - include the bias from length = 1
        bias += self.length_1_bias

        # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = torch.eq(
                input_ids[:, -prefix_length:],
                torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
            ).prod(dim=1)
            bias[:, last_token] += torch.where(
                matching_rows.bool(),
                torch.tensor(sequence_bias, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )

        # 5 - apply the bias to the scores
        scores = scores + bias
        return scores

    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        vocabulary_size = scores.shape[-1]

        # Check biased tokens out of bounds
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # Precompute the bias tensors to be applied. Sequences of length 1 are kept separately, as they can be applied
        # with simpler logic.
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True

    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
        if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )
        if any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")


class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


    >>> def get_tokens_as_list(word_list):
    ...     "Converts a sequence of words into a list of tokens"
    ...     tokens_list = []
    ...     for word in word_list:
    ...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
    ...         tokens_list.append(tokenized_word)
    ...     return tokens_list


    >>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        # Filter EOS token from bad_words_ids
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # Forbidding a sequence is equivalent to setting its bias to -inf
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0

        return scores + mask


class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more
    details.

    <Tip>

    Diverse beam search can be particularly useful in scenarios where a variety of different outputs is desired, rather
    than multiple similar sequences. It allows the model to explore different generation paths and provides a broader
    coverage of possible outputs.

    </Tip>

    <Tip warning={true}>

    This logits processor can be resource-intensive, especially when using large models or long sequences.

    </Tip>

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    How It Works:
    - **Grouping Beams**: Beams are divided into groups. Each group selects tokens independently of the others.
    - **Penalizing Repeated Tokens**: If a beam in a group selects a token already chosen by another group in the
        same step, a penalty is applied to that token's score.
    - **Promoting Diversity**: This penalty discourages beams within a group from selecting the same tokens as
        beams in other groups.

    Benefits:
    - **Diverse Outputs**: Produces a variety of different sequences.
    - **Exploration**: Allows the model to explore different paths.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if group beam search is enabled. The
            penalty applied to a beam's score when it generates a token that has already been chosen by another beam
            within the same group during the same time step. A higher `diversity_penalty` will enforce greater
            diversity among the beams, making it less likely for multiple beams to choose the same token. Conversely, a
            lower penalty will allow beams to more freely choose similar tokens. Adjusting this value can help strike a
            balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams used for group beam search. Beam search is a method used that maintains beams (or "multiple
            hypotheses") at each step, expanding each one and keeping the top-scoring sequences. A higher `num_beams`
            will explore more potential sequences. This can increase chances of finding a high-quality output but also
            increases computational cost.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            Each group of beams will operate independently, selecting tokens without considering the choices of other
            groups. This division promotes diversity by ensuring that beams within different groups explore different
            paths. For instance, if `num_beams` is 6 and `num_beam_groups` is 2, there will be 2 groups each containing
            3 beams. The choice of `num_beam_groups` should be made considering the desired level of output diversity
            and the total number of beams. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> # A long text about the solar system
    >>> text = "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud."
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

    >>> # With `diversity_penalty`, the resulting beams are much more diverse
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step.
            beam_group_idx (`int`):
                The index of the beam group currently being processed.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.
        """
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_id]] = -float("inf")
            for i in self.eos_token_id:
                scores[:, i] = 0
        return scores


class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # set all nan values to 0.0
        scores[scores != scores] = 0.0

        # set all inf values to max possible value
        scores[scores == float("inf")] = torch.finfo(scores.dtype).max

        return scores


class ExponentialDecayLengthPenalty(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that exponentially increases the score of the `eos_token_id` after `start_index` has been
    reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
    predicted in a meaningful position.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        input_ids_seq_length (`int`):
            The length of the input sequence.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

    >>> text = "Just wanted to let you know, I"
    >>> inputs = tokenizer(text, return_tensors="pt")

    >>> # Generate sequences without exponential penalty. We want short sentences, so we limit max_length=30
    >>> # see that the answer tends to end abruptly
    >>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I'm not even a lawyer. I'm a man. I have no real knowledge of politics. I'm a

    >>> # Generate sequences with exponential penalty, we add the exponential_decay_length_penalty=(start_index, decay_factor)
    >>> # We see that instead of cutting at max_tokens, the output comes to an end before (at 25 tokens) and with more meaning
    >>> # What happens is that starting from `start_index` the EOS token score will be increased by decay_factor exponentially
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.6),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I've got a very cool t-shirt educating people on how to use the Internet<|endoftext|>

    >>> # Generate sequences with smaller decay_factor, still improving the hard cutoff mid-sentence
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.05),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I've been working on it for about 6 months and now it's in Alpha.<|endoftext|>
    ```
    """

    def __init__(
        self,
        exponential_decay_length_penalty: Tuple[int, float],
        eos_token_id: Union[int, List[int]],
        input_ids_seq_length: int,
    ):
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            for i in self.eos_token_id:
                penalty_idx = cur_len - self.regulation_start
                # To support negative logits we compute the penalty of the absolute value and add to the original logit
                scores[:, i] = scores[:, i] + torch.abs(scores[:, i]) * (pow(self.regulation_factor, penalty_idx) - 1)
        return scores


class LogitNormalization(LogitsProcessor, LogitsWarper):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.log_softmax(dim=-1)
        return scores


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == self.begin_index:
            scores[:, self.begin_suppress_tokens] = -float("inf")

        return scores


class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they
    are not sampled."""

    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.suppress_tokens] = -float("inf")
        return scores


class ForceTokensLogitsProcessor(LogitsProcessor):
    r"""This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index."""

    def __init__(self, force_token_map: List[List[int]]):
        self.force_token_map = dict(force_token_map)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generation_idx = input_ids.shape[-1]
        current_token = self.force_token_map.get(generation_idx, None)
        if current_token is not None:
            scores[:, :] = -float("inf")
            scores[:, current_token] = 0
        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """

    def __init__(self, generate_config):  # support for the kwargs
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        self.begin_index = len(generate_config.forced_decoder_ids) + 2
        if generate_config.forced_decoder_ids[-1][1] == self.no_timestamps_token_id:
            self.begin_index -= 1
        self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores[:, self.no_timestamps_token_id] = -float("inf")

        if input_ids.shape[1] == self.begin_index - 1:
            scores[:, :] = -float("inf")
            scores[:, self.timestamp_begin] = 0
            return scores

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            seq = list(input_ids[k, self.begin_index :].tolist())
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores[k, self.timestamp_begin :] = -float("inf")
                else:  # cannot be normal text tokens
                    scores[k, : self.eos_token_id] = -float("inf")

            # apply the `max_initial_timestamp` option
            if input_ids.shape[1] == self.begin_index and self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores[:, last_allowed + 1 :] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                scores[k, : self.timestamp_begin] = -float("inf")

        return scores


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""Logits processor for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
    """

    def __init__(self, guidance_scale):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # simple check to make sure we have compatible batch sizes between our
        # logits scores (cond + uncond) and input ids (cond only)
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores


class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of [`Bark`]'s fine submodel.

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    """

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[-1]

        # even -> first codebook, odd -> second codebook
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        if is_first_codebook:
            scores[:, : self.semantic_vocab_size] = -float("inf")
            scores[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        return scores


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""Logits processor for Classifier-Free Guidance (CFG). The processors
    computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits,
    parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with
    the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, **optional**):
            Attention mask for unconditional_ids.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        smooth_factor (`float`, **optional**):
            The interpolation weight for CFG Rescale. 1 means no rescaling, 0 reduces to the conditional scores without
            CFG. Turn it lower if the output degenerates.
        use_cache (`bool`, **optional**):
            Whether to cache key/values during the negative prompt forward pass.


    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        out = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return out
