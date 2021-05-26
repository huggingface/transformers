# coding=utf-8
# Copyright 2021 The Google AI Flax Team Authors, and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Dict, Optional

import flax
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
from jax import lax

from .file_utils import ModelOutput
from .generation_flax_logits_process import (
    FlaxLogitsProcessorList,
    FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper,
)
from .utils import logging


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. If all batches finished early due to the :obj:`eos_token_id`, :obj:`sequences` is
            padded to :obj:`max_length`.
    """

    sequences: jax_xla.DeviceArray = None


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_length)`):
            The generated sequences. If all batches finished early due to the :obj:`eos_token_id`, :obj:`sequences` is
            padded to :obj:`max_length`.
    """

    sequences: jax_xla.DeviceArray = None


@flax.struct.dataclass
class GreedyState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    current_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]


@flax.struct.dataclass
class SampleState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    current_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    prng_key: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]


class FlaxGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.FlaxPreTrainedModel`.
    """

    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        """
        Run generation in untraced mode. This should only be used for debugging purposes.
        """
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    def generate(
        self,
        input_ids: jax_xla.DeviceArray,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jax_xla.DeviceArray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        trace: bool = True,
        **model_kwargs,
    ):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        and, multinomial sampling.

        Apart from :obj:`input_ids`, all the arguments below will default to the value of the attribute of the same
        name inside the :class:`~transformers.PretrainedConfig` of the model. The default values indicated are the
        default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            trace (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to trace generation. Setting ``trace=False`` should only be used for debugging and will lead to
                a considerably slower runtime.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :class:`~transformers.file_utils.ModelOutput`.

        Examples::
            >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="jax").input_ids
            >>> # generate candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        do_sample = do_sample if do_sample is not None else self.config.do_sample

        if do_sample:
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)
            return self._sample(
                input_ids,
                max_length,
                pad_token_id,
                eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                model_kwargs=model_kwargs,
                trace=trace,
            )
        else:
            return self._greedy_search(
                input_ids, max_length, pad_token_id, eos_token_id, trace=trace, model_kwargs=model_kwargs
            )

    def _get_logits_warper(
        self, top_k: int = None, top_p: float = None, temperature: float = None
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a :obj:`~transformers.FlaxLogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.FlaxLogitsWarper` instances used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        temperature = temperature if temperature is not None else self.config.temperature
        # instantiate warpers list
        warpers = FlaxLogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(FlaxTemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(FlaxTopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p is not None and top_p < 1.0:
            warpers.append(FlaxTopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        return warpers

    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        trace: bool = True,
        model_kwargs: Optional[Dict[str, jax_xla.DeviceArray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        model = self

        # initialize model specific kwargs
        model_kwargs = model.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            current_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model(state.current_token, **state.model_kwargs)
            next_token = jnp.argmax(model_outputs.logits[:, -1], axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = model.update_inputs_for_generation(model_outputs, model_kwargs)

            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                current_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return FlaxGreedySearchOutput(sequences=state.sequences)

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jax_xla.DeviceArray] = None,
        model_kwargs: Optional[Dict[str, jax_xla.DeviceArray]] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
    ):
        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id)
        pad_token_id = jnp.array(pad_token_id)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        model = self

        # initialize model specific kwargs
        model_kwargs = model.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            current_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.current_token, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply top_k, top_k, temperature
            logits = logits_warper(state.sequences, logits)

            next_token = jax.random.categorical(prng_key, model_outputs.logits[:, -1], axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = model.update_inputs_for_generation(model_outputs, model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                current_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)
