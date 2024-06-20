# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

from torch import distributed as dist


def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:  # scatter_idx == 1, scatter sequence dim
        input_t = input.reshape(
            inp_shape[: scatter_idx] +  # the batch size dim
            [seq_world_size, inp_shape[scatter_idx]] +  # scatter the sequence dim
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()
    else:  # scatter_idx == 2, scatter heads dim
        input_t = input.reshape(
            [-1] +  # flatten batch size and sequence dim
            [seq_world_size, inp_shape[scatter_idx]] +  # scatter the heads
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    if scatter_idx < 2:
        output = output.reshape(
            [
                seq_world_size,
                -1,  # batch size dim
                inp_shape[scatter_idx]  # sequence dim (scattered)
            ] + inp_shape[gather_idx:]  # heads dim, and the rest
        ).permute(
            1, 2, 0, *list(range(3, len(output.shape)))
        )
    else:
        output = output.reshape(
            [
                seq_world_size,
                -1,  # batch size dim
                inp_shape[gather_idx]  # sequence dim
            ] + inp_shape[scatter_idx:]  # heads dim (scattered), and the rest
        ).transpose(
            0, 1
        )

    return output.reshape(
        inp_shape[:gather_idx] +
        [seq_world_size * inp_shape[gather_idx]] +
        inp_shape[gather_idx + 1:]
    ).contiguous()


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

        assert scatter_idx == 2 and gather_idx == 1, 'Only support shape [b, s, h, ...]'

        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """ forward

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
