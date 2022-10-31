# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from .misc import Attention, Dropout, PairToSequence, ResidueMLP, SequenceToPair
from .openfold_triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from .openfold_triangular_multiplicative_update import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing


# TODO Matt: This module can be left unchanged and does not receive any config objects


class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        dropout=0,
    ):
        super().__init__()

        if sequence_state_dim % sequence_head_width != 0:
            raise ValueError(
                f"`sequence_state_dim` (here {sequence_state_dim} needs to be round multiple of `sequence_head_dim`"
                f" (here {sequence_head_width})."
            )
        if pairwise_state_dim % pairwise_head_width != 0:
            raise ValueError(
                f"`pairwise_state_dim` (here {pairwise_state_dim} needs to be round multiple of `pairwise_head_width`"
                f" (here {pairwise_head_width})."
            )
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width

        if sequence_state_dim != sequence_num_heads * sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {sequence_state_dim} != {sequence_num_heads} * {sequence_head_width}."
            )
        if pairwise_state_dim != pairwise_num_heads * pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {pairwise_state_dim} != {pairwise_num_heads} * {pairwise_head_width}."
            )
        if pairwise_state_dim % 2 != 0:
            raise ValueError(f"`pairwise_state_dim` should be even, got {pairwise_state_dim}.")

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = SequenceToPair(sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim)
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

        self.seq_attention = Attention(sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True)
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        if dropout >= 0.4:
            raise ValueError(f"`dropout` should not be greater than 0.4, got {dropout}.")
        self.drop = nn.Dropout(dropout)
        self.row_drop = Dropout(dropout * 2, 2)
        self.col_drop = Dropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim mask: B x L boolean
          tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim
        """
        if len(sequence_state.shape) != 3:
            raise ValueError(f"`sequence_state` should be a 3d-tensor, got {len(sequence_state.shape)} dims.")
        if len(pairwise_state.shape) != 4:
            raise ValueError(f"`pairwise_state` should be a 4d-tensor, got {len(pairwise_state.shape)} dims.")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]

        if sequence_state_dim != self.sequence_state_dim:
            raise ValueError(
                "`sequence_state` last dimension should be equal to `self.sequence_state_dim`. Got"
                f"{sequence_state_dim} != {self.sequence_state_dim}."
            )
        if pairwise_state_dim != self.pairwise_state_dim:
            raise ValueError(
                "`pairwise_state` last dimension should be equal to `self.pairwise_state_dim`. Got "
                f"{pairwise_state_dim} != {self.pairwise_state_dim}."
            )
        if batch_dim != pairwise_state.shape[0]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent batch size: {batch_dim} != "
                f"{pairwise_state.shape[0]}."
            )
        if seq_dim != pairwise_state.shape[1] or seq_dim != pairwise_state.shape[2]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent sequence length: {seq_dim} != "
                f"{pairwise_state.shape[1]} or {pairwise_state.shape[2]}."
            )

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(self.tri_mul_out(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.col_drop(self.tri_mul_in(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state
