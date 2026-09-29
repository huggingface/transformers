# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Interface for the stateful core of linear attention layers.

A linear attention layer keeps its projections, gating and output projection in the modeling file and dispatches only
its stateful core (causal convolution, recurrence and reading/writing of the recurrent state) through
`ALL_LINEAR_ATTENTION_FUNCTIONS`. This lets an external runtime, which manages the recurrent state itself, swap the core
without re-implementing the rest of the layer.

Functions are registered under `"<implementation>|<mechanism>"` and selected with `config._linear_attn_implementation`.
The signature of a function depends on its mechanism:

- `"ssd"` (Mamba2's state space duality):
  `(module, hidden_states_B_C, dt, cache_params=None, attention_mask=None, **kwargs) -> scan_output`, where
    - `module` is the mixer, from which the function reads `conv1d`, `A_log`, `D`, `dt_bias` and the layer geometry.
    - `hidden_states_B_C` has shape `(batch_size, seq_len, conv_dim)` and is the input to the convolution.
    - `dt` has shape `(batch_size, seq_len, num_heads)` and is the time step before `dt_bias` and softplus.
    - `scan_output` has shape `(batch_size, seq_len, num_heads * head_dim)` and is the output of the scan, including
      the `D` skip connection but before the gated normalization.
"""

from __future__ import annotations

from collections.abc import Callable

from ..utils import logging
from ..utils.generic import GeneralInterface


logger = logging.get_logger(__name__)


class LinearAttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of the functions that implement the stateful core of linear attention layers. Keys
    are `"<implementation>|<mechanism>"`, e.g. `"vllm|ssd"`, because the signature of the core differs between
    mechanisms. See the module docstring for the signature of each mechanism.
    """

    _global_mapping = {}

    def get_interface(self, linear_attn_implementation: str | None, mechanism: str, default: Callable) -> Callable:
        """Return the function registered for `linear_attn_implementation` and `mechanism`, or `default` for
        `"eager"`. Raise if `linear_attn_implementation` does not implement `mechanism`."""
        if linear_attn_implementation is None:
            logger.warning_once(
                "You tried to access the `LinearAttentionInterface` with a `config._linear_attn_implementation` set "
                "to `None`. This is expected if you use a linear attention module as a standalone module. If this is "
                "not the case, something went wrong with the dispatch of `config._linear_attn_implementation`"
            )
            return default
        if linear_attn_implementation == "eager":
            return default
        key = f"{linear_attn_implementation}|{mechanism}"
        if key not in self:
            raise KeyError(
                f"`{linear_attn_implementation}` does not implement the `{mechanism}` linear attention mechanism. "
                "Register a function for it in the `LinearAttentionInterface` under "
                f'`"{key}"`, or use `linear_attn_implementation="eager"`.'
            )
        return self[key]

    def implementations(self) -> set[str]:
        """Every implementation with at least one registered mechanism."""
        return {key.split("|", 1)[0] for key in self}


ALL_LINEAR_ATTENTION_FUNCTIONS = LinearAttentionInterface()
