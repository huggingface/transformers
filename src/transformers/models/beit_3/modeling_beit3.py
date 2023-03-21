import copy

import math
import time
from typing import Optional, Tuple, Dict, Any, cast

import numpy as np
from torch import nn, Tensor
import torch
from torch.autograd.grad_mode import F
from torch.nn import LayerNorm, ModuleList

from transformers.activations import get_activation
from ...utils import logging

EVAL_CAPACITY_TOKEN_FRACTION = 0.25
SAMPLE_FRACTION = 0.2
logger = logging.get_logger(__name__)

def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module

def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(
        indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype
    )
    output.scatter_(len(output.shape) - 1, indices, 1)
    return output


def fused_cumsum_sub_one(mask):
    return torch.cumsum(mask, dim=0) - 1

def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    use_xmoe=False,
    gate_obj=None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()

    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = capacity_factor * S/E
        capacity = int(capacity_factor * math.ceil(num_tokens / num_experts))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = (
        100
        * torch.histc(
            (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = (
        torch.sort(expert1_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    gates1_s = (gates * mask1).sum(dim=1)

    # Compute locations in capacity buffer
    locations1 = fused_cumsum_sub_one(mask1)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)

    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    dispatch_mask = combine1_sec.bool()
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata



def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    one = torch.tensor(1.0, device=device)
    zero = torch.tensor(0.0, device=device)
    gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
    return gumbel(shape)


def top2gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    second_expert_policy="sampling",
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    batch_prioritized_routing=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = 2 * math.ceil(num_tokens / num_experts)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)
    if second_expert_policy == "sampling":
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    mask2 = one_hot(indices2_s, num_experts)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)

    if normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s

    if second_expert_policy == "random":
        sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # Compute locations in capacity buffer
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
        mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    if batch_prioritized_routing:
        # if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = fused_cumsum_sub_one(sorted_mask1) * sorted_mask1
        importance_sorted_locations1 = sorted_cumsum1[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = fused_cumsum_sub_one(sorted_mask2) * sorted_mask2
        importance_sorted_locations2 = sorted_cumsum2[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = (
            importance_sorted_locations1,
            importance_sorted_locations2,
        )
    else:
        locations1 = fused_cumsum_sub_one(mask1)
        locations2 = fused_cumsum_sub_one(mask2)
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    # for logging purposes
    metadata["overflow_expert1"] = (
        100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    )
    metadata["overflow_expert2"] = (
        100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)
    )

    # Remove locations outside capacity from mask
    mask1_, mask2_ = mask1, mask2
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = (
        100
        * torch.histc(
            (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = (
        torch.sort(expert1_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    expert2_hist = (
        100
        * torch.histc(
            (indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = (
        torch.sort(expert2_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()

    if not normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    combine2_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates2.unsqueeze(-1),
        locations2_sc.to(gates2.dtype).unsqueeze(1),
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    if use_fp32:
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_weights, dispatch_mask, metadata


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Beit3DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1


class VisionEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        contain_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if contain_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                batch_size, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class TextEmbedding(nn.Embedding):
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class PositionalEmbedding(nn.Embedding):
    def forward(
        self,
        x,
        positions=None,
        **kwargs,
    ):
        if positions is None:
            # being consistent with Fairseq, which starts from 2.
            positions = (
                torch.arange(2, x.size(1) + 2, device=x.device).long().unsqueeze(0)
            )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = self.get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_rng_state(self):
        state = {"torch_rng_state": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        return state

    def set_rng_state(self, state):
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.set_rng_state(self.rng_state)


def make_experts(args, embed_dim, expert_ffn_dim):
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    expert_list = []
    ddp_rank = args.ddp_rank
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert (
            args.moe_expert_count % world_size == 0
        ), f"{args.moe_expert_count}, {world_size}"
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(
                    FeedForwardNetwork(
                        embed_dim,
                        expert_ffn_dim,
                        args.activation_fn,
                        args.dropout,
                        args.activation_dropout,
                        args.layernorm_eps,
                        args.subln,
                    )
                )
    else:
        assert (
            world_size % args.moe_expert_count == 0
        ), f"{world_size}, {args.moe_expert_count}"

        with set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(
                FeedForwardNetwork(
                    embed_dim,
                    expert_ffn_dim,
                    args.activation_fn,
                    args.dropout,
                    args.activation_dropout,
                    args.layernorm_eps,
                    args.subln,
                )
            )
    experts = nn.ModuleList(expert_list)
    return experts

# def _find_my_group_index(grouped_ranks):
#     my_rank = dist.get_rank()
#     for i, group in enumerate(grouped_ranks):
#         if my_rank in group:
#             return i
#     raise RuntimeError


# def get_moe_group(moe_expert_count):
#     if dist.is_initialized():
#         if not hasattr(get_moe_group, "_moe_groups"):
#             world_size = dist.get_world_size()
#
#             if world_size <= moe_expert_count:
#                 assert moe_expert_count % world_size == 0
#                 moe_groups = [[i] for i in range(world_size)]
#
#             else:
#                 assert world_size % moe_expert_count == 0
#                 ranks_per_group = world_size // moe_expert_count
#                 moe_groups = [
#                     [i + j * moe_expert_count for j in range(ranks_per_group)]
#                     for i in range(moe_expert_count)
#                 ]
#
#             get_moe_group._moe_group_idx = moe_groups
#             get_moe_group._moe_groups = [dist.new_group(g) for g in moe_groups]
#
#         my_group_idx = _find_my_group_index(get_moe_group._moe_group_idx)
#         return get_moe_group._moe_groups[my_group_idx]


# def get_all2all_group(moe_expert_count):
#     if dist.is_initialized():
#         if not hasattr(get_all2all_group, "_all2all_groups"):
#             world_size = dist.get_world_size()
#
#             # more experts than world size
#             if world_size <= moe_expert_count:
#                 assert moe_expert_count % world_size == 0
#                 all2all_groups = [[i for i in range(world_size)]]
#
#             # larger world than num experts
#             else:
#                 assert world_size % moe_expert_count == 0
#                 ranks_per_group = world_size // moe_expert_count
#                 all2all_groups = [
#                     [i * moe_expert_count + j for j in range(moe_expert_count)]
#                     for i in range(ranks_per_group)
#                 ]
#
#             get_all2all_group._all2all_group_idx = all2all_groups
#             get_all2all_group._all2all_groups = [
#                 dist.new_group(g) for g in all2all_groups
#             ]
#
#         my_group_idx = _find_my_group_index(get_all2all_group._all2all_group_idx)
#         return get_all2all_group._all2all_groups[my_group_idx]


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation(activation_fn)
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::
        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        use_xmoe=False,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        if not use_xmoe:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = torch.nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            torch.nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", torch.nn.Parameter(wg))

        self.use_xmoe = use_xmoe
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction

    def forward(self, input, mask=None):  # type: ignore
        if self.use_xmoe:
            input = self.wg_reduction(input)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input)

        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            capacity_factor=self.capacity_factor,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            use_xmoe=self.use_xmoe,
            gate_obj=self,
        )

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

    def _get_gating_temperature(self, eps=1e-4):
        if self.gating_t.data.item() < eps:
            return eps
        return self.gating_t

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        # if torch.distributed.is_initialized():
        #     dist.all_to_all_single(output, input, group=group)
        # else:
        assert group is None
        output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::
        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        second_expert_policy="sampling",
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=0.25,
        batch_prioritized_routing=False,
        use_xmoe=False,
    ) -> None:
        super().__init__()
        if not use_xmoe:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = torch.nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            torch.nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", torch.nn.Parameter(wg))
        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.use_xmoe = use_xmoe

    def forward(self, input, mask=None):  # type: ignore
        if self.use_xmoe:
            input = self.wg_reduction(input)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input)
        return top2gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            batch_prioritized_routing=self.batch_prioritized_routing,
        )

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


class MOELayer(nn.Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::
        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        gate (torch.nn.Module):
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate, experts, args):
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        # self.expert_group = get_moe_group(args.moe_expert_count)
        # self.all2all_group = get_all2all_group(args.moe_expert_count)
        # self.world_size = dist.get_world_size(group=self.expert_group)
        # self.all2all_size = dist.get_world_size(group=self.all2all_group)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert (
            len(input.shape) == 3
        ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert (
                len(input_padding_mask.shape) == 2
            ), "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            logger.warning(
                f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
            )
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (
                    expected_bsz,
                    input_shape[1],
                ),
                dtype=torch.bool,
                device=input.device,
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[: input_shape[0], :] = input_padding_mask
            else:
                padded_input_padding_mask[: input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask

        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = (
            input_padding_mask.reshape(-1) if input_padding_mask is not None else None
        )

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = reshaped_input_shape[0] * torch.ones(
                (1,), dtype=torch.long, device=input.device
            )
            dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
            expected_dim = int(expected_dim.item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[
                    : reshaped_input_shape[0]
                ] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[: reshaped_input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask

        l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
            reshaped_input, reshaped_input_padding_mask
        )

        dispatch_mask = dispatch_mask.to(input.dtype).permute(
            1, 2, 0
        )  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        M = reshaped_input.size(1)
        assert reshaped_input.size() == (S, M)
        # einsum("sec,sm->ecm")
        dispatched_input = torch.mm(
            dispatch_mask.view(E * C, S), reshaped_input
        )  # -> (E*C),M

        # if self.all2all_size > 1:
        #     dispatched_input = self.all_to_all_wrapper(dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            self.all2all_size, self.num_local_experts, -1, d_model
        )
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)

        if self.all2all_size > 1:
            expert_output = self.all_to_all_wrapper(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * self.num_local_experts, -1, d_model
        )

        # einsum("sec,ecm->sm")
        combined_output = combine_weights.view(S, E * C).mm(
            expert_output.view(E * C, M)
        )

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[: reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[: input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, "dummy_a2a", False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += cpu_end - cpu_start
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, "record_a2a_perf_stats", False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

class XPOS(nn.Module):
    def __init__(
            self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        return attn, attn_weights

def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn

class EncoderLayer(nn.Module):
    def __init__(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
        self.dropout_module = torch.nn.Dropout(args.dropout)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.encoder_layers)[
                depth
            ]
            self.drop_path = Beit3DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.normalize_before = args.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.encoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = MultiwayWrapper(
                args,
                self.build_ffn(
                    self.embed_dim,
                    self.args,
                ),
            )
        else:
            assert not self.args.multiway
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)
        self.final_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    * 0.81
                )
            else:
                self.alpha = math.pow(2.0 * args.encoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, x, encoder_padding_mask, attn_mask=None, rel_pos=None, multiway_split_position=None, incremental_state=None):
        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x = x.transpose(0, 1)
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux

class RelativePositionBias(nn.Module):
    def __init__(
        self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        step = 0 if step is None else step
        context_position = torch.arange(
            step,
            step + qlen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device,
        )[:, None]
        memory_position = torch.arange(
            klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        # shape (batch * num_heads, qlen, klen)
        return (
            self.compute_bias(qlen, klen, step)
            .repeat(batch_size, 1, 1, 1)
            .view(-1, qlen, klen)
        )

def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if isinstance(module.q_proj, MultiwayNetwork):
            normal_(module.q_proj.A.weight.data)
            normal_(module.q_proj.B.weight.data)
            normal_(module.k_proj.A.weight.data)
            normal_(module.k_proj.B.weight.data)
            normal_(module.v_proj.A.weight.data)
            normal_(module.v_proj.B.weight.data)
        else:
            normal_(module.q_proj.weight.data)
            normal_(module.k_proj.weight.data)
            normal_(module.v_proj.weight.data)

class Encoder(nn.Module):
    def __init__(
        self,
        args,
        embed_tokens=None,
        embed_positions=None,
        output_projection=None,
        is_encoder_decoder=False,
        **kwargs
    ):
        self.args = args
        super().__init__(**kwargs)

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.encoder_embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if (
            output_projection is None
            and not is_encoder_decoder
            and not args.no_output_layer
            and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = MultiwayWrapper(
                args, LayerNorm(embed_dim, eps=args.layernorm_eps), dim=1
            )
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.encoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_encoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before and args.normalize_output:
            self.layer_norm = MultiwayWrapper(args, LayerNorm(embed_dim, eps=args.layernorm_eps))
        else:
            self.layer_norm = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.encoder_attention_heads,
            )
        else:
            self.relative_position = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    / 1.15
                )
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(
                    math.log(3 * args.decoder_layers)
                    * math.log(2 * args.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        if args.share_encoder_input_output_embed:
            assert args.encoder_embedding_type == "language"
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.encoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.encoder_embed_dim**-0.5
            )
        return output_projection

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = EncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        src_tokens,
        token_embedding=None,
        positions=None,
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(src_tokens, positions=positions)
            else:
                x = embed + self.embed_positions(x, positions=positions)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        **kwargs
    ):
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(
                    src_tokens, device=src_tokens.device
                ).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
            )

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        l_aux = []
        for idx, layer in enumerate(self.layers):
            x, l_aux_i = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                attn_mask=attn_mask,
                rel_pos=rel_pos_bias,
                multiway_split_position=multiway_split_position,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only and self.output_projection is not None:
            x = self.output_projection(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
        }


class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embed(visual_tokens, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out