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
Interfaces for the stateful core of linear attention layers.

A linear attention layer keeps its projections, gating and output projection in the modeling file and dispatches only
its stateful core (causal convolution, recurrence and reading/writing of the recurrent state) through the interface of
its mechanism. This lets an external runtime, which manages the recurrent state itself, swap the core without
re-implementing the rest of the layer.

Each mechanism has its own interface because the signature of its core differs. All of them are selected with
`config._linear_attn_implementation`.
"""

from __future__ import annotations

from collections.abc import Callable

from ..utils import logging
from ..utils.generic import GeneralInterface


logger = logging.get_logger(__name__)


class LinearAttentionInterface(GeneralInterface):
    """Base class for the per-mechanism linear attention interfaces. Subclasses must define their own
    `_global_mapping` so that registering a function for one mechanism does not register it for the others."""

    def get_interface(self, linear_attn_implementation: str | None, default: Callable) -> Callable:
        """Return the requested `linear_attn_implementation`, or `default` for `"eager"`. Raise if the requested
        implementation is not registered for this mechanism."""
        if linear_attn_implementation is None:
            logger.warning_once(
                f"You tried to access the `{type(self).__name__}` with a `config._linear_attn_implementation` set to "
                "`None`. This is expected if you use a linear attention module as a standalone module. If this is "
                "not the case, something went wrong with the dispatch of `config._linear_attn_implementation`"
            )
        elif linear_attn_implementation != "eager" and linear_attn_implementation not in self:
            raise KeyError(
                f"`{linear_attn_implementation}` is not a valid linear attention implementation registered in the "
                f"`{type(self).__name__}`"
            )
        return super().get(linear_attn_implementation, default)


class SSDInterface(LinearAttentionInterface):
    """
    Interface for the core of Mamba2 (state space duality, SSD): causal convolution, chunked scan and recurrent state
    handling.

    A registered function has the signature
    `(module, hidden_states_B_C, dt, cache_params=None, attention_mask=None, **kwargs) -> scan_output`, where:

    - `module` is the mixer, from which the function reads `conv1d`, `A_log`, `D`, `dt_bias` and the layer geometry.
    - `hidden_states_B_C` has shape `(batch_size, seq_len, conv_dim)` and is the input to the convolution.
    - `dt` has shape `(batch_size, seq_len, num_heads)` and is the time step before `dt_bias` and softplus.
    - `scan_output` has shape `(batch_size, seq_len, num_heads * head_dim)` and is the output of the selective scan,
      including the `D` skip connection but before the gated normalization.
    """

    _global_mapping = {}


ALL_SSD_FUNCTIONS = SSDInterface()

ALL_LINEAR_ATTENTION_INTERFACES: tuple[LinearAttentionInterface, ...] = (ALL_SSD_FUNCTIONS,)
"""Every linear attention interface, used to validate `linear_attn_implementation`."""
