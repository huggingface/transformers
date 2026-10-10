# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Structured forward state with independently evolving tensor streams."""

from dataclasses import dataclass, field

import torch
from torch.utils._pytree import register_pytree_node, tree_flatten


@dataclass
class LayerExecutionState:
    """The main hidden stream and optional named streams; all tensors remain connected to autograd.

    Args:
        hidden_states (`torch.Tensor`): Main decoder state.
        streams (`dict[str, torch.Tensor]`, *optional*): Named independently evolving tensor states.
    """

    hidden_states: torch.Tensor
    streams: dict[str, torch.Tensor] = field(default_factory=dict)


register_pytree_node(
    LayerExecutionState,
    lambda state: ([state.hidden_states, state.streams], None),
    lambda children, context: LayerExecutionState(*children),
)


def state_signature(state):
    """Validate every stream, including its pytree structure, rather than silently dropping auxiliary state."""
    leaves, structure = tree_flatten(state)
    if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
        raise ValueError("Layer state must contain only tensor streams.")
    return structure, tuple(value.shape for value in leaves)
