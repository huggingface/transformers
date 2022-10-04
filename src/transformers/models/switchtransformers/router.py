# coding=utf-8
# Copyright 2022 Mesh TensorFlow authors, SwitchTransformers Authors and HuggingFace Inc. team.
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
from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn

# Output classes

RouterOutput = Any

@dataclass
class RouterIndices:
    r"""
    Dispatch indices and combine weights for scatter/gather-based routing.

    Attributes:
    dispatch_indices: <int32>[num_groups, tokens_per_group,
        num_selected_experts, 2] dispatch indices indicating, for each token, its preferred expert and its priority in
        that expert's buffer.
    combine_weights: <float>[num_groups, tokens_per_group, num_selected_experts]
        combine weights used for scaling expert outputs with the router's dispatch probability/confidence.
    auxiliary_loss: Load balancing loss for router.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
        effort to improve stability.
    """
    dispatch_indices: torch.Tensor
    combine_weights: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.

@dataclass
class RouterMask:
    r"""
    Dispatch and combine torch.Tensors for expert routing with masked matmuls.

    Attributes:
    dispatch_mask: <bool>[num_groups, tokens_per_group, num_experts,
        expert_capacity] dispatch torch.Tensor that is 1 if the token gets routed to the corresponding expert, and 0
        otherwise.
    combine_torch.Tensor: <float>[num_groups, tokens_per_group, num_experts,
        expert_capacity] combine torch.Tensor used for combining expert outputs and scaling with router probability.
    auxiliary_loss: Load balancing loss for router.
    router_z_loss: Router z-loss. Encourages router logits to remain small in an
        effort to improve stability.
    """
    dispatch_mask: torch.Tensor
    combine_array: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.

# Router loss

def _router_z_loss(router_logits: torch.Tensor) -> float:
    r"""
    Compute router z-loss implemented in PyTorch.

    The router z-loss was introduced in Designing Effective Sparse Expert Models (https://arxiv.org/abs/2202.08906). It
    encourages router logits to remain small in an effort to improve stability.

    Args:
    router_logits: <float>[num_groups, tokens_per_group, num_experts] router
        logits.

    Returns:
    Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def _load_balancing_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function implements the loss function presented in
    equations (4) - (6). It aims to penalize those cases where the routing between experts is unbalanced.

    Args:
    router_probs: Probability assigned to each expert per token. Shape:
        <float32>[num_groups, tokens_per_group, num_experts].
    expert_indices: <int>[num_groups, tokens_per_group, num_selected_experts]
        indices identifying the top num_selected_experts for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    
    # For a given token, determine if it was routed to a given expert.
    # Shape: [num_groups, tokens_per_group, num_experts]
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)

# Router classes

class Router(nn.Module):
    """
    Abstract base router class, defining router API and inner workings.

    Attributes:
    router_weights: Configurable module used to compute router logits from token
        inputs.
    jitter_noise: Amplitude of jitter noise applied to router logits.
    dtype: Numeric float type for returned combine torch.Tensor. All actual
        computations are performed in float32 of the input for stability.
    ignore_padding_tokens: Whether to ignore padding tokens during routing. Note
        that some routers (e.g. TokensChooseMaskedRouter) will completely ignore padding tokens, while others (e.g.
        TokensChooseScatterRouter and ExpertsChooseMaskedRouter) will simply down-weight the probability of selecting
        padding tokens.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_experts = config.num_experts
        self.router_weights = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
    
    def _compute_router_probabilities(self, token_inputs: torch.Tensor, num_experts: int, apply_jitter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input tokens.

        Args:
        token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] from which
            router probabilities are computed.
        num_experts: Number of experts.
        apply_jitter: If true, apply jitter noise.

        Returns:
        - <float32>[num_groups, tokens_per_group, num_experts] probabilities for
            each token and expert. Used for routing tokens to experts.
        - <float>[num_groups, tokens_per_group, num_experts] raw router logits.
            Used for computing router z-loss.
        """
        # For remainder of routing computation we use float32 to ensure stability.
        # See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        token_inputs = token_inputs.to(torch.float32)

        if apply_jitter and self.jitter_noise > 0:
            token_inputs *= torch.random.uniform(
                token_inputs.shape,
                token_inputs.dtype,
                minval=1.0 - self.jitter_noise,
                maxval=1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        router_logits = self.router_weights(token_inputs, num_experts)

        router_probabilities = torch.nn.softmax(router_logits, axis=-1)

        return router_probabilities, router_logits


    def forward(self, token_inputs: torch.Tensor, expert_capacity: int, apply_jitter: bool = True) -> RouterOutput:
        r"""
        Args:
        Computes dispatch and combine torch.Tensors for routing to experts.
            token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to
            send to experts.
            num_experts: Number of experts.
            expert_capacity: Each group will send this many tokens to each expert.
            apply_jitter: If true, apply jitter noise during routing.
        Returns:
            Router indices or mask torch.Tensors (depending on router type).
        """
        router_probs, router_logits = self._compute_router_probabilities(token_inputs, self.num_experts, apply_jitter)


        if self.ignore_padding_tokens:
            # To identify non-padding tokens, we rely on the fact that padding tokens
            # in the inputs have already been masked in the default T5 architecture.
            # See
            # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
            # and
            # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
            padding_mask = jnp.torch.Tensor((jnp.sum(jnp.abs(token_inputs), axis=-1) > 0),
                                    dtype=token_inputs.dtype)
            router_logits *= jnp.expand_dims(padding_mask, axis=-1)
        else:
            padding_mask = None

        instructions = self._compute_routing_instructions(router_probs,
                                                            padding_mask,
                                                            expert_capacity)

        return instructions.replace(router_z_loss=_router_z_loss(router_logits))