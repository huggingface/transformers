# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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
from jax import lax

from .utils import logging

from .generation_flax_logits_process import (
    FlaxLogitsProcessorList,
    FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper,
)


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.DeviceArray
    sequences: jnp.DeviceArray
    current_token: jnp.DeviceArray
    is_sent_finished: jnp.DeviceArray
    model_kwargs: Dict[str, jnp.DeviceArray]


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.DeviceArray
    sequences: jnp.DeviceArray
    current_token: jnp.DeviceArray
    is_sent_finished: jnp.DeviceArray
    prng_key: jnp.DeviceArray
    model_kwargs: Dict[str, jnp.DeviceArray]


class FlaxGenerationMixin:
    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    def generate(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
        prng_key: Optional[jnp.DeviceArray] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        trace: bool = True,
        **model_kwargs,
    ):
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        do_sample = do_sample if do_sample is not None else self.config.do_sample

        #        model_kwargs["attention_mask"] = attention_mask
        if do_sample:
            logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, temperature=temperature)
            return self._sample(
                input_ids, max_length, pad_token_id, eos_token_id, prng_key, logits_warper=logits_warper, model_kwargs=model_kwargs, trace=trace
            )
        else:
            return self._greedy_search(
                input_ids, max_length, pad_token_id, eos_token_id, trace=trace, model_kwargs=model_kwargs
            )

    def _get_logits_warper(self, top_k: int = None, top_p: float = None, temperature: float = None) -> FlaxLogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
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
        model_kwargs: Optional[Dict[str, jnp.DeviceArray]] = None,
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

        return state.sequences

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: jnp.DeviceArray = jax.random.PRNGKey(0),
        model_kwargs: Optional[Dict[str, jnp.DeviceArray]] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
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

        return state.sequences
