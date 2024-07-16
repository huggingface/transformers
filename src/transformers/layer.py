# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple

import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import Module


def single_all_to_all(input, scatter_idx, gather_idx, group):
    if dist.get_world_size(group) <= 1:
        return input
    seq_world_size = dist.get_world_size(group)

    input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    # TODO: use all_to_all_single instead
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_idx).contiguous()


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm

        support and only support shape [b, s, h, d]
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,  # scatter dim, commonly the head dim
        gather_idx: int = 1,  # gather dim, commonly the sequence dim
    ) -> None:
        super(DistributedAttention, self).__init__()

        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [b,s/p:h:]
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        # out shape : e.g., [b,s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [b,s/p::h]
        return output
