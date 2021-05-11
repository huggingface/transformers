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
        debug=False,
    ):
        # set init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        return self.greedy_search(input_ids, max_length, pad_token_id, eos_token_id, debug=debug)

    def greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        debug=False,
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

        state = GreedyState(cur_len=cur_len, sequences=sequences, is_sent_finished=is_sent_finished)

        def greedy_search_cond_fn(state):
            """Sampling loop termination condition."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """Sampling loop state update."""
            input_ids = state.sequences
            generated_token_index = lax.cond(
                state.cache is not None, lambda _: 1, lambda _: state.cur_len, operand=None
            )
            next_logits = model(input_ids)[0][:, generated_token_index]

            next_tokens = jnp.argmax(next_logits, axis=-1)

            next_tokens = next_tokens * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_sequences = lax.dynamic_update_slice(state.sequences, next_tokens[:, None], (0, state.cur_len))

            is_sent_finished = state.is_sent_finished | (next_sequences[:, state.cur_len] == eos_token_id)

            return GreedyState(cur_len=state.cur_len + 1, sequences=next_sequences, is_sent_finished=is_sent_finished)

        # Run greedy search and collect final state.
        if debug:
            final_state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            final_state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        sequences = final_state.sequences

        return sequences
