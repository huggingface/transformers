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
Interfaces for the stateful pieces of linear attention layers.

A linear attention layer keeps its projections, gating, masking and output projection in the modeling file and
dispatches only the pieces that read or write the recurrent state through an interface. Each such piece has its own
interface, so that an external runtime, which manages the recurrent state itself, can swap them without
re-implementing the rest of the layer. For example, a Mamba2 mixer dispatches its causal convolution (which owns the
conv state) through the `CausalConv1dInterface` and its selective scan (which owns the recurrent state) through the
`SSDInterface`.

All of them are selected with `config._linear_attn_implementation`.

Outside the interfaces, a layer must read the sizes it computes with (e.g. to split its projections) from attributes of
the module at forward time, not from its config or from values computed at init. A runtime that shards the layer for
tensor parallelism then only has to set those attributes to their rank-local values.
"""

from __future__ import annotations

from collections.abc import Callable

from ..utils import logging
from ..utils.generic import GeneralInterface


logger = logging.get_logger(__name__)


class LinearAttentionInterface(GeneralInterface):
    """Base class for the linear attention interfaces. Subclasses must define their own `_global_mapping` so that
    registering a function for one piece does not register it for the others."""

    def get_interface(self, linear_attn_implementation: str | None, default: Callable) -> Callable:
        """Return the requested `linear_attn_implementation`, or `default` for `"eager"`. Raise if the requested
        implementation is not registered for this interface."""
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


class CausalConv1dInterface(LinearAttentionInterface):
    """
    Interface for the causal depthwise convolution of a linear attention layer and the handling of its conv state.

    A registered function has the signature
    `(module, hidden_states, cache_params=None, use_precomputed_states=False, **kwargs) -> hidden_states`, where:

    - `module` is the mixer, from which the function reads `conv1d`, `activation`, `conv_kernel_size` and `layer_idx`.
    - `hidden_states` has shape `(batch_size, seq_len, conv_dim)` and is the input to the convolution.
    - `use_precomputed_states` is whether `cache_params` held a previous state for this layer before the forward pass.
    - The returned `hidden_states` has the same shape and is the output of the convolution, after the activation.
    """

    _global_mapping = {}


class SSDInterface(LinearAttentionInterface):
    """
    Interface for the selective scan of Mamba2 (state space duality, SSD) and the handling of its recurrent state.

    A registered function has the signature
    `(module, hidden_states, B, C, dt, cache_params=None, use_precomputed_states=False, **kwargs) -> scan_output`,
    where:

    - `module` is the mixer, from which the function reads `A_log`, `D`, `dt_bias`, `time_step_limit`, `layer_idx` and
      the layer geometry.
    - `hidden_states` has shape `(batch_size, seq_len, num_heads * head_dim)`.
    - `B` and `C` have shape `(batch_size, seq_len, n_groups * ssm_state_size)`.
    - `dt` has shape `(batch_size, seq_len, num_heads)` and is the time step before `dt_bias` and softplus.
    - `use_precomputed_states` is whether `cache_params` held a previous state for this layer before the forward pass.
      It cannot be read from `cache_params` here because the convolution has already updated it.
    - `scan_output` has shape `(batch_size, seq_len, num_heads * head_dim)` and is the output of the selective scan,
      including the `D` skip connection but before the gated normalization.
    """

    _global_mapping = {}


ALL_CAUSAL_CONV1D_FUNCTIONS = CausalConv1dInterface()
ALL_SSD_FUNCTIONS = SSDInterface()

ALL_LINEAR_ATTENTION_INTERFACES: tuple[LinearAttentionInterface, ...] = (
    ALL_CAUSAL_CONV1D_FUNCTIONS,
    ALL_SSD_FUNCTIONS,
)
"""Every linear attention interface, used to validate `linear_attn_implementation`."""
