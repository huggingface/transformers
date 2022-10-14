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
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


# from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig


# Output classes
RouterOutput = Any


def _jax_one_hot(tensor, num_classes, axis=-1, dtype=torch.bool):
    r"""
    This function mimics the behavior of jax.nn.functional.one_hot in PyTorch. It takes a tensor of indices, the number
    of desired classes and the axis to apply the one-hot encoding. If any value is outside the range [0, num_classes),
    it will be set to zeros.

    Args:
        tensor (`torch.Tensor`):
            Input tensor
        num_classes (`int`):
            Number of classes to process for one hot encoding
        axis (`int`, *optional*):
            The lookup axis to check for one-hot encoding
        dtype (`torch.dtype`, *optional*):
            Output `dtype`. The one hot encoded vector will be casted to this dtype
    """
    if tensor.is_floating_point():
        raise "Input tensor for one hot encoding must be an `int32` or `int64`"

    if axis >= len(tensor.shape):
        raise "Axis is out of bounds"

    if axis == -1:
        axis = len(tensor.shape)
    elif axis < -1:
        raise "Axis must be greater than -1"
    else:
        axis = axis + 1

    # Get the final output shape
    output_shape = list(tensor.shape)
    output_shape.insert(axis, num_classes)

    # Create an empty output of zeros
    out = torch.zeros(tuple(output_shape), dtype=dtype)

    # Mask out the places where it is outside the range [0, num_classes)
    # kudos to GitHub copilot for this line
    mask = (tensor >= 0) & (tensor < num_classes)
    out[mask, tensor[mask]] = 1

    return out


@dataclass
class RouterIndices:
    r"""
    A dataclass wrapper to store the dispatch indices and combine weights for scatter/gather-based routing.

    Attributes:
        dispatch_indices (`torch.Tensor`):
            A tensor of size [`num_groups`, `tokens_per_group`, `num_selected_experts`, 2] dispatch indices indicating,
            for each token, its preferred expert and its priority in that expert's buffer.
        combine_weights (`torch.Tensor`):
            A tensor of size [`num_groups`, `tokens_per_group`, `num_selected_experts`] combine weights used for
            scaling expert outputs with the router's dispatch probability/confidence.
        auxiliary_loss (`float`):
            Load balancing loss for router.
        router_z_loss (`float`):
            Router z-loss. Encourages router logits to remain small in an effort to improve stability.
    """
    dispatch_indices: torch.Tensor
    combine_weights: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.0

    def to(self, device):
        return replace(
            self, dispatch_mask=self.dispatch_indices.to(device), combine_array=self.combine_weights.to(device)
        )


@dataclass
class RouterMask:
    r"""
    Dispatch and combine torch.Tensors for expert routing with masked matmuls.

    Attributes:
        dispatch_mask (`torch.Tensor`):
            A mask tensor of shape [num_groups, tokens_per_group, num_experts, expert_capacity] that is 1 if the token
            gets routed to the corresponding expert, and 0 otherwise.
        combine_array (`torch.Tensor`):
            A tensor of shape [num_groups, tokens_per_group, num_experts, expert_capacity] combine torch.Tensor used
            for combining expert outputs and scaling with router probability.
        auxiliary_loss (`float`):
            Load balancing loss for router.
        router_z_loss (`float`):
            Router z-loss. Encourages router logits to remain small in an effort to improve stability.
    """
    dispatch_mask: torch.Tensor
    combine_array: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.0

    def to(self, device):
        return replace(self, dispatch_mask=self.dispatch_mask.to(device), combine_array=self.combine_array.to(device))


# Router loss


def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute router z-loss implemented in PyTorch.

    The router z-loss was introduced in Designing Effective Sparse Expert Models (https://arxiv.org/abs/2202.08906). It
    encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [num_groups, tokens_per_group, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961). This function implements the loss function presented in
    equations (4) - (6). It aims to penalize those cases where the routing between experts is unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [num_groups, tokens_per_group, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [num_groups, tokens_per_group, num_selected_experts] identifying the top
            num_selected_experts for a given token.

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
        router_weights (`torch.nn.Module`):
            Configurable module used to compute router logits from token inputs.
        jitter_noise (`float`):
            Amplitude of jitter noise applied to router logits.
        dtype (`torch.dtype`):
            Numeric float type for returned combine torch.Tensor. All actual computations are performed in float32 of
            the input for stability.
        ignore_padding_tokens (`bool`):
            Whether to ignore padding tokens during routing. Note that some routers (e.g. `TokensChooseMaskedRouter`)
            will completely ignore padding tokens, while others (e.g. `TokensChooseScatterRouter` and
            `ExpertsChooseMaskedRouter`) will simply down-weight the probability of selecting padding tokens.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_experts = config.num_experts
        self.router_weights = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(
        self, token_inputs: torch.Tensor, num_experts: int, apply_jitter: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input tokens.

        Args:
            token_inputs (`torch.Tensor`):
                [num_groups, tokens_per_group, hidden_dim] from which router probabilities are computed.
            num_experts (`int`):
                Number of experts.
            apply_jitter (`bool`):
                If true, apply jitter noise.

        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape [num_groups, tokens_per_group, num_experts] corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape [num_groups, tokens_per_group, num_experts] corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # For remainder of routing computation we use float32 to ensure stability.
        # See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        self.input_tokens_dtype = token_inputs.dtype
        token_inputs = token_inputs.to(self.dtype)

        if apply_jitter and self.jitter_noise > 0:
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = (
                torch.rand(token_inputs.shape) * (distrib_lower_bound - distrib_upper_bound)
            ) + distrib_upper_bound

            # Multiply the token inputs by the uniform distribution - adding some noise
            token_inputs *= uniform_distrib

        # Shape: [num_groups, tokens_per_group, num_experts]
        router_logits = self.router_weights(token_inputs)

        router_probabilities = torch.nn.Softmax(dim=-1)(router_logits)

        return router_probabilities, router_logits

    def forward(
        self, token_inputs: torch.Tensor, expert_capacity: int, apply_jitter: bool = True, **kwargs
    ) -> RouterOutput:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`token_inputs`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
        Computes dispatch and combine torch.Tensors for routing to experts.
            token_inputs: <float>[num_groups, tokens_per_group, hidden_dim] inputs to send to experts. num_experts:
            Number of experts. expert_capacity: Each group will send this many tokens to each expert. apply_jitter: If
            true, apply jitter noise during routing.
        Returns:
            Router indices or mask torch.Tensors (depending on router type).
        """
        router_probs, router_logits = self._compute_router_probabilities(token_inputs, self.num_experts, apply_jitter)

        # Flax code for reference
        if self.ignore_padding_tokens:
            # To identify non-padding tokens, we rely on the fact that padding tokens
            # in the inputs have already been masked in the default T5 architecture.
            # See
            # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L315
            # and
            # https://github.com/google/flaxformer/blob/9712a16/flaxformer/architectures/t5/t5_architecture.py#L603.
            padding_mask = torch.Tensor((torch.sum(torch.abs(token_inputs), axis=-1) > 0)).to(token_inputs.dtype)
            router_logits *= padding_mask.unsqueeze(-1)
        else:
            padding_mask = None

        instructions = self._compute_routing_instructions(router_probs, padding_mask, expert_capacity, **kwargs)
        # We cast back the output to the previous dtype
        instructions = instructions.to(self.input_tokens_dtype)

        return replace(instructions, router_z_loss=router_z_loss_func(router_logits))

    def _compute_routing_instructions(self, router_probs, padding_mask, expert_capacity):
        raise NotImplementedError(
            """
                The forward function cannot be called from the `Router` super-class. Please call an appropriate Router
                class that inherits from the `Router` class (for example `ExpertsChooseMaskedRouter`)
            """
        )


class MaskedRouter(Router):
    """
    Abstract base router class for masked matmul dispatch routers.

    MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine array for sending and receiving (via
    masked matmuls) inputs and outputs to and from experts.

    Routing using masked matmuls is generally faster than scatter-based routing on TPUs.
    """

    def _compute_routing_instructions(
        self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int
    ) -> RouterMask:
        """
        Computes masks for the top-k experts per token. This has to be implemented for each subclass of MaskedRouter
        routers.

        Args:
            router_probs (`torch.Tensor`):
                Input router probabilities of shape [num_groups, tokens_per_group, num_experts] this corresponds to the
                probabilities used to determine the routing of tokens to the experts.
            padding_mask (`torch.Tensor`):
                Padding mask tensor of shape [num_groups, tokens_per_group] a mask used to identify padding tokens that
                should be ignored by the router.
            expert_capacity (`int`):
                Each group will send this many tokens to each expert.

        Returns:
            Router mask arrays.
        """
        raise NotImplementedError("MaskedRouter is an abstract class that should be subclassed.")


class ExpertsChooseMaskedRouter(MaskedRouter):
    """
    Masked matmul router using experts choose tokens assignment.

    This router uses the same mechanism as in Mixture-of-Experts with Expert Choice (https://arxiv.org/abs/2202.09368):
    each expert selects its top expert_capacity tokens. An individual token may be processed by multiple experts or
    none at all.

    Note: "experts choose routing" should not be used in decoder blocks because it breaks the autoregressive behavior
    -- the model will learn to cheat by using future token information to improve current token predictions.
    """

    def _compute_routing_instructions(
        self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int
    ) -> RouterMask:
        """Computes masks for the highest probability token per expert.

        Args:
            router_probs (`torch.Tensor`):
                Raw router probabilities of shape [num_groups, tokens_per_group, num_experts] used to determine the
                routing of tokens to the experts.
            padding_mask (`torch.Tensor`):
                padding mask tensor of shape [num_groups, tokens_per_group] used to identify padding tokens that should
                be down-weighted by the router.
            expert_capacity (`int`):
                Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        tokens_per_group = router_probs.shape[1]

        if padding_mask is not None:
            # Because experts choose tokens, we mask probabilities corresponding to
            # tokens before the top-k operation. Note that, unlike for masked-based
            # tokens-choose routing, the experts here may still choose to select the
            # (down-weighted) padding tokens.
            router_probs *= padding_mask.unsqueeze(-1)

        # vmap over group dimension.
        # router_probs_t = router_probs.t()
        router_probs_t = router_probs.permute(0, 2, 1)

        # Top expert_capacity router probability and corresponding token indices for
        # each expert. Shapes: [num_groups, num_experts, expert_capacity].
        expert_gate, expert_index = torch.topk(router_probs_t, k=expert_capacity)

        # Convert to one-hot mask of expert indices for each token in each group.
        # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
        dispatch_mask = _jax_one_hot(expert_index, tokens_per_group, dtype=torch.int32)

        # Move axes to conform with shape expected by MoeLayer API.
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        dispatch_mask = torch.moveaxis(dispatch_mask, 3, 1)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, num_experts, tokens_per_group,
        # expert_capacity].
        combine_array = torch.einsum("...ec,...tec->...tec", expert_gate, dispatch_mask)

        # Each expert is choosing tokens until it reaches full capacity, so we don't
        # need an auxiliary loading balancing loss for expert choice routing.
        auxiliary_loss = 0.0

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


class TokensChooseMaskedRouter(MaskedRouter):
    """
    Masked matmul router using tokens choose top-k experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. There is no guarantee that each
    token is processed by an expert, or that each expert receives at least one token.

    Attributes:
    num_selected_experts (`int`):
        Maximum number of experts to which each token is routed. Tokens may be routed to fewer experts if particular
        experts are oversubscribed / reach capacity.
    batch_prioritized_routing (`bool`):
        Whether or not to use Batch Prioritized Routing (BPR), originally introduced in V-MoE
        (https://arxiv.org/abs/2106.05974). With BPR, we prioritize routing those top-k tokens with the highest router
        probability, rather than simply using each tokens left-to-right ordering in the batch. This prioritization is
        important because the experts have limited capacity.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_selected_experts = config.num_selected_experts
        self.batch_prioritized_routing = config.batch_prioritized_routing

    def _compute_routing_instructions(
        self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int
    ) -> RouterMask:
        """
        Computes masks for the top-k experts per token.

        Args:
            router_probs (`torch.Tensor`):
                Router raw probabilities tensor of shape [num_groups, tokens_per_group, num_experts] used to determine
                the routing of tokens to the experts.
            padding_mask (`torch.Tensor`):
                Padding mask of shape [num_groups, tokens_per_group] used to identify padding tokens that should be
                ignored by the router.
            expert_capacity (`int`):
                Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        num_groups, _, num_experts = router_probs.shape

        # Top-k router probability and corresponding expert indices for each token.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        expert_gate, expert_index = torch.topk(router_probs, k=self.num_selected_experts)

        if padding_mask is not None:
            # Mask applied to gate. Exclude choices corresponding to padding tokens.
            gate_mask = padding_mask.unsqueeze(-1).to(expert_index.dtype)
            expert_gate *= gate_mask

            # Set `expert_index` elements corresponding to padding to negative
            # numbers. Negative `expert_index` elements will ultimately be dropped in
            # the one_hot conversion to the `expert_mask`.
            # First convert nonzero padding elements to negative values.
            expert_index *= (2 * gate_mask) - 1
            # Handle zero padding elements by negatively shifting all padding.
            expert_index += (gate_mask - 1).repeat(1, 1, self.num_selected_experts)

            # To correctly compute load balancing loss, we also mask out probs.
            router_probs *= gate_mask

        auxiliary_loss = load_balancing_loss_func(router_probs, expert_index)

        if self.batch_prioritized_routing:
            # Sort tokens according to their routing probability per group, so that
            # the highest probability tokens are routed first.
            permutation = torch.argsort(-expert_gate[..., 0], dim=-1)
            # Shape: [num_groups, tokens_per_group, num_selected_experts]
            expert_index = torch.take_along_dim(expert_index, permutation.unsqueeze(-1), dim=-2)

        # Make num_selected_experts the leading axis to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3 choices,
        # etc.
        expert_index = expert_index.permute((0, 2, 1))
        # Shape: [num_groups, num_selected_experts * tokens_per_group]
        expert_index = expert_index.reshape(num_groups, -1)

        # Create mask out of indices.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        expert_mask = torch.nn.functional.one_hot(expert_index, num_experts)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        token_priority = torch.cumsum(expert_mask, axis=1) * expert_mask - 1.0
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape((num_groups, self.num_selected_experts, -1, num_experts))
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = token_priority.permute((0, 2, 1, 3))
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [num_groups, tokens_per_group, num_experts].
        token_priority = torch.max(token_priority, axis=2).values

        if self.batch_prioritized_routing:
            # Place token priorities in original ordering of tokens.
            inv_permutation = torch.argsort(permutation, dim=-1)
            token_priority = torch.take_along_dim(token_priority, inv_permutation.unsqueeze(-1), dim=-2)

        # Token T can only be routed to expert E if its priority is positive and
        # less than the expert capacity. One-hot matrix will ignore indices outside
        # the range [0, expert_capacity).
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
        # token_priority = token_priority * (token_priority > 0)

        # dispatch_mask = torch.nn.functional.one_hot(token_priority.long(), expert_capacity + 1)[..., 1:]
        dispatch_mask = _jax_one_hot(token_priority.long(), expert_capacity, axis=-1)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
        # expert_capacity].
        combine_array = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
        # combine_array = torch.einsum("...te,...te->...te", router_probs, dispatch_mask)

        # Return to default dtype now that router computation is complete.
        combine_array = combine_array.to(torch.float32)

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)

class ScatterRouter(Router):
    """
    Abstract base router class for scatter dispatch routers.

    ScatterRouter(s) return RouterIndices containing dispatch indices and combine weights for sending token inputs (via
    scatter) and receiving outputs (via gather) to and from experts.

    Scatter-based routing is generally faster than masked matmul routing on CPUs and GPUs.
    """

    def _compute_routing_instructions(
        self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int
    ) -> RouterIndices:
        """Computes instructions for routing inputs to experts.

        Args:
            router_probs (`torch.Tensor`):
                Router raw probabilities tensor of shape [num_groups, tokens_per_group, num_experts] used to determine
                the routing of tokens to the experts.
            padding_mask (`torch.Tensor`):
                Padding mask of shape [num_groups, tokens_per_group] used to identify padding tokens that should be
                ignored by the router.
            expert_capacity (`int`):
                Each group will send this many tokens to each expert.

        Returns:
            Router indices containing dispatch indices and combine weights.
        """
        raise NotImplementedError("ScatterRouter is an abstract class that should be subclassed.")


class TokensChooseScatterRouter(ScatterRouter):
    """
    Scatter router using tokens choose top-k experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. There is no guarantee that each
    token is processed by an expert, or that each expert receives at least one token.

    Attributes:
        num_selected_experts (`int`):
            Maximum number of experts to which each token is routed. Tokens may be routed to fewer experts if
            particular experts are oversubscribed / reach capacity.
        batch_prioritized_routing (`bool`):
            Whether or not to use Batch Prioritized Routing BPR), originally introduced in V-MoE
            (https://arxiv.org/abs/2106.05974). With BPR, we prioritize routing those top-k tokens with the highest
            router probability, rather than simply using each tokens left-to-right ordering in the batch. This
            prioritization is important because the expert's have limited capacity.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_selected_experts = config.num_selected_experts
        self.batch_prioritized_routing = config.batch_prioritized_routing

    def _compute_routing_instructions(
        self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity: int
    ) -> RouterIndices:
        """Computes dispatch indices and combine weights for the top-k experts.

        Args:
            router_probs (`torch.Tensor`):
                Router raw probabilities tensor of shape [num_groups, tokens_per_group, num_experts] used to determine
                the routing of tokens to the experts.
            padding_mask (`torch.Tensor`):
                Padding mask of shape [num_groups, tokens_per_group] used to identify padding tokens that should be
                ignored by the router.
            expert_capacity (`int`):
                Each group will send this many tokens to each expert.

        Returns:
            Dispatch indices and combine weights for scatter/gather-based routing.
        """
        num_groups, tokens_per_group, num_experts = router_probs.shape

        if padding_mask is not None:
            # Because experts choose tokens, we mask probabilities corresponding to
            # tokens before the top-k operation. Note that, unlike for masked-based
            # tokens-choose routing, the experts here may still choose to select the
            # (down-weighted) padding tokens.
            router_probs *= padding_mask.unsqueeze(-1)

        # Top-k router probability and corresponding expert indices for each token.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        combine_weights, expert_indices = torch.topk(router_probs, k=self.num_selected_experts)

        auxiliary_loss = load_balancing_loss_func(router_probs, expert_indices)

        if self.batch_prioritized_routing:
            # Sort tokens according to their routing probability per group, so that
            # the highest probability tokens are routed first.
            token_ordering = torch.argsort(-combine_weights[..., 0], dim=-1)
            # Shape: [num_groups, tokens_per_group, num_selected_experts]
            expert_indices = torch.take_along_dim(expert_indices, token_ordering.unsqueeze(-1), dim=-2)

        # Identify each token's preferred expert.
        # Make num_selected_experts the leading axis to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3
        # choices...
        preferred_experts = expert_indices.permute(0, 2, 1)
        # Shape: [num_groups, num_selected_experts * tokens_per_group]
        preferred_experts = preferred_experts.reshape(num_groups, -1)

        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        expert_mask = _jax_one_hot(preferred_experts, num_experts, dtype=torch.int32)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        token_priority = torch.cumsum(expert_mask, axis=1) * expert_mask - 1.0
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape((num_groups, self.num_selected_experts, -1, num_experts))
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = token_priority.permute((0, 2, 1, 3))
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [num_groups, tokens_per_group, num_experts].
        token_priority = torch.max(token_priority, axis=-1).values

        # Return to original index shape.
        preferred_experts = preferred_experts.reshape(num_groups, self.num_selected_experts, tokens_per_group)
        # Shape: [num_groups, tokens_per_group, num_selected_experts]
        preferred_experts = preferred_experts.permute(0, 2, 1)

        if self.batch_prioritized_routing:
            # Place token priorities in original ordering of tokens.
            inv_permutation = torch.argsort(token_ordering, dim=-1)
            preferred_experts = torch.take_along_dim(
                preferred_experts.unsqueeze(-1), inv_permutation.unsqueeze(-1), dim=-2
            )
            token_priority = torch.take_along_dim(token_priority.unsqueeze(-1), inv_permutation.unsqueeze(-1), dim=-2)

        # Mask out tokens that overflow the maximum expert capacities.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        combine_weights *= token_priority < expert_capacity

        # Expert index and priority within the expert capacity buffer.
        # Shape: [num_groups, tokens_per_group, num_selected_experts, 2].
        dispatch_indices = torch.stack([preferred_experts, token_priority], dim=-1)

        # Return to default dtype now that router computation is complete.
        dispatch_indices = dispatch_indices.to(torch.int32)

        return RouterIndices(dispatch_indices, combine_weights, auxiliary_loss)


# num_groups = 2
# tokens_per_group = 4
# hidden_dim = 3
# num_experts = 2
# expert_capacity = 2 # Total capacity = 2*2*1 = 4 < num_tokens
# jitter_noise = 0.0

# input_tokens = torch.Tensor(
#     [[[0.6433916 , 0.18188512, 0.02240455],
#     [0.563781  , 0.5526401 , 0.0958724 ],
#     [0.34253013, 0.03644359, 0.08744538],
#     [0.7909105 , 0.35205448, 0.53364205]],

#     [[0.02900076, 0.4168595 , 0.5802449 ],
#     [0.91486526, 0.27414513, 0.14991808],
#     [0.9383501 , 0.5209162 , 0.51207185],
#     [0.90618336, 0.7309413 , 0.95533276]]]
# )

# config = SwitchTransformersConfig(
#     num_experts=num_experts,
#     hidden_size=hidden_dim,
#     router_jitter_noise=jitter_noise,
#     expert_capacity=expert_capacity,
#     batch_prioritized_routing=False,
# )
# # model = TokensChooseMaskedRouter(config)
# model = ExpertsChooseMaskedRouter(config)

# model.router_weights.weight = torch.nn.Parameter(
#     torch.Tensor([[-0.00107201,  0.01544739],
#         [-0.0087319 ,  0.01314363],
#         [ 0.03530733,  0.03709853]]).t()
# )

# model(input_tokens, expert_capacity=expert_capacity)


# hidden_dim = 4
# num_experts = 2
# num_selected_experts = 1  # Switch routing case
# expert_capacity = 1  # Total capacity = 2*2*1 = 4 < num_tokens
# jitter_noise = 0.0

# input_tokens = torch.Tensor(
# [
#     [
#         [0.6433916, 0.18188512, 0.02240455, 0.563781],
#         [0.5526401, 0.0958724, 0.34253013, 0.03644359],
#         [0.08744538, 0.7909105, 0.35205448, 0.53364205],
#     ],
#     [
#         [0.02900076, 0.4168595, 0.5802449, 0.91486526],
#         [0.27414513, 0.14991808, 0.9383501, 0.5209162],
#         [0.51207185, 0.90618336, 0.7309413, 0.95533276],
#     ],
# ]
# )

# config = SwitchTransformersConfig(
# num_experts=num_experts,
# hidden_size=hidden_dim,
# num_selected_experts=num_selected_experts,
# router_jitter_noise=jitter_noise,
# expert_capacity=expert_capacity,
# batch_prioritized_routing=False,
# )
# model = TokensChooseMaskedRouter(config)

# model.router_weights.weight = torch.nn.Parameter(
# torch.Tensor(
#     [
#         [0.02008116, 0.00620062],
#         [-0.00811031, -0.00031623],
#         [-0.03542127, 0.02703803],
#         [0.02335377, -0.02971946],
#     ],
# ).t()
# )

# output = model(input_tokens, expert_capacity=expert_capacity)


# num_groups = 2
# tokens_per_group = 4
# hidden_dim = 3
# num_experts = 3
# num_selected_experts = 1
# expert_capacity = 2
# jitter_noise = 0.0

# input_tokens = torch.Tensor(
#     [[[0.6433916 , 0.18188512, 0.02240455],
#               [0.563781  , 0.5526401 , 0.0958724 ],
#               [0.34253013, 0.03644359, 0.08744538],
#               [0.7909105 , 0.35205448, 0.53364205]],

#              [[0.02900076, 0.4168595 , 0.5802449 ],
#               [0.91486526, 0.27414513, 0.14991808],
#               [0.9383501 , 0.5209162 , 0.51207185],
#               [0.90618336, 0.7309413 , 0.95533276]]]
# )

# config = SwitchTransformersConfig(
#     num_experts=num_experts,
#     hidden_size=hidden_dim,
#     num_selected_experts=num_selected_experts,
#     router_jitter_noise=jitter_noise,
#     expert_capacity=expert_capacity,
#     batch_prioritized_routing=False,
# )
# model = TokensChooseScatterRouter(config)

# model.router_weights.weight = torch.nn.Parameter(
#     torch.Tensor(
#         [[ 0.02736656, -0.00253537,  0.04682618],
#         [ 0.00928149,  0.04933621, -0.00275501],
#         [ 0.00751786,  0.04295348, -0.00503795]],
#     ).t()
# )

# output = model(input_tokens, expert_capacity=expert_capacity)
