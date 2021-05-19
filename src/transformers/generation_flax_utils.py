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


from typing import Any, Optional

import flax
import jax.numpy as jnp
from jax import lax

from .utils import logging


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.DeviceArray
    sequences: jnp.DeviceArray
    current_token: jnp.DeviceArray
    is_sent_finished: jnp.DeviceArray
    cache: Any = None


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
        trace: bool = True,
    ):
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        return self.greedy_search(input_ids, max_length, pad_token_id, eos_token_id, trace=trace)

    def greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
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
        init_sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        init_sequences = lax.dynamic_update_slice(init_sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        model = self

        # initializing the cache
        past_key_values = model.init_cache(batch_size, max_length)

        import ipdb

        ipdb.set_trace()
        # first step
        model_outputs = model(input_ids, past_key_values=past_key_values)
        next_token = jnp.argmax(logits[:, -1:], axis=-1)

        state = GreedyState(
            cur_len=cur_len,
            sequences=init_sequences,
            current_token=next_token,
            is_sent_finished=is_sent_finished,
            cache=past_key_values,
        )

        def greedy_search_cond_fn(state):
            """Sampling loop termination condition."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """Sampling loop state update."""
            is_sent_finished = state.is_sent_finished | (state.current_token == eos_token_id)
            current_token = state.current_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished

            next_sequences = lax.dynamic_update_slice(state.sequences, current_token, (0, state.cur_len))

            next_logits, next_cache = model(current_token, past_key_values=state.cache).logits[:, -1]
            next_token = jnp.argmax(next_logits, axis=-1)

            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                current_token=next_token,
                is_sent_finished=is_sent_finished,
                cache=next_cache,
            )

        # Run greedy search and collect final state.
        if not trace:
            final_state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            final_state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        sequences = final_state.sequences

        return sequences
