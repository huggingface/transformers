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

from __future__ import annotations

from ..modeling_outputs import MoERouting
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch


def normalize_moe_routing(
    moe_routing: MoERouting | dict[int, torch.LongTensor] | None,
) -> dict[int, torch.LongTensor] | None:
    """
    Normalize the accepted MoE routing inputs into a layer-indexed dictionary.

    `MoERouting` is the preferred public replay payload. A dictionary keyed by layer index is kept as a supported
    internal/backward-compatible shape for model implementations that already use it.
    """
    if moe_routing is None:
        return None
    if isinstance(moe_routing, MoERouting):
        return dict(enumerate(moe_routing.selected_experts))
    if isinstance(moe_routing, dict):
        return moe_routing
    raise TypeError(
        "`moe_routing` must be a `MoERouting` instance or a dict keyed by layer index with expert index tensors."
    )


def validate_forced_selected_experts(
    selected_experts: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    model_name: str,
) -> torch.LongTensor:
    """
    Validate the selected experts tensor used for replay and return it cast to `torch.long`.
    """
    selected_experts = selected_experts.to(dtype=torch.long)
    expected_shape = (num_tokens, top_k)
    if selected_experts.shape != expected_shape:
        raise ValueError(
            f"Forced {model_name} routing must match the flattened token shape and configured top-k. "
            f"Expected {expected_shape}, got {tuple(selected_experts.shape)}."
        )
    if selected_experts.numel() > 0 and (
        int(selected_experts.min().item()) < 0 or int(selected_experts.max().item()) >= num_experts
    ):
        raise ValueError(
            f"Forced {model_name} routing indices must be in [0, {num_experts}), "
            f"got min={int(selected_experts.min().item())}, max={int(selected_experts.max().item())}."
        )
    return selected_experts


def gather_forced_routing_scores(
    router_probs: torch.Tensor,
    selected_experts: torch.LongTensor,
    *,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.LongTensor]:
    """
    Gather the current router probabilities for the forced expert indices.

    This mirrors Megatron's minimal replay contract: preserve the expert path exactly while recomputing current router
    weights for those experts.
    """
    routing_scores = torch.gather(router_probs, dim=-1, index=selected_experts)
    if renormalize:
        routing_scores = routing_scores / routing_scores.sum(dim=-1, keepdim=True)
    return routing_scores.to(router_probs.dtype), selected_experts
