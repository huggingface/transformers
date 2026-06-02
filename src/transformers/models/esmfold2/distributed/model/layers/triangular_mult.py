# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Distributed TriangleMultiplicativeBlock for ESMFold2's pair representation.

The pair tensor z has shape (B, N, N, d_pair) and is distributed across a 2D
CP grid as DTensor with placements (Shard(0), Shard(1), Shard(2)).  GPU (i, j)
owns the shard z[..., i_start:i_end, j_start:j_end, :].

Triangle multiplication patterns:
  Outgoing: contracted[b,n,m,d] = sum_k  a[b,n,k,d] * b[b,m,k,d]   (einsum "bnkd,bmkd->bnmd")
  Incoming: contracted[b,n,m,d] = sum_k  a[b,k,n,d] * b[b,k,m,d]   (einsum "bknd,bkmd->bnmd")

The distributed BMM uses ring communication to accumulate partial results:
  - One operand is transposed across the 2D grid (so (i,j) gets the chunk from (j,i))
  - Both operands are ring-shifted (row-wise and column-wise respectively)
  - Each step computes a local matmul; results are accumulated
"""

from enum import Enum, auto
from typing import Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from projects.huggingface.transformers.models.esmfold2.distributed.comm import (
    Ring2DComm,
)
from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.linear import (
    LinearParamsReplicated,
)
from projects.huggingface.transformers.models.esmfold2.distributed.utils import (
    update_exhaustive_strides,
)
from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
    TriangleMultiplicativeBlock as SerialTriangleMultiplicativeBlock,
)


class _Direction(Enum):
    Outgoing = auto()
    Incoming = auto()


# ---------------------------------------------------------------------------
# Core distributed batch matmul
# ---------------------------------------------------------------------------


class _XposeArgs(Enum):
    lhs = auto()
    rhs = auto()


def _distributed_bmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    comm: Ring2DComm,
    permute_lhs: tuple[int, ...] | None = None,
    permute_rhs: tuple[int, ...] | None = None,
    permute_out: tuple[int, ...] | None = None,
    xpose_args: _XposeArgs | None = None,
) -> torch.Tensor:
    """Distributed batch matmul using ring communication on a 2D process grid.

    See boltz-cp's comm.py diagrams for the full algorithm description.

    Parameters
    ----------
    lhs, rhs:
        Local tensor shards.  Shape: (B, ..., N, K) after optional permute.
    comm:
        Ring2DComm object providing all the communication handles.
    permute_lhs, permute_rhs:
        Optional permutations applied before computation.
    permute_out:
        Optional permutation applied to the output.
    xpose_args:
        Which operand to transpose across the grid first.
    """
    if permute_lhs is not None:
        lhs = lhs.permute(permute_lhs)
    lhs = lhs.clone(memory_format=torch.contiguous_format)
    if permute_rhs is not None:
        rhs = rhs.permute(permute_rhs)
    rhs = rhs.clone(memory_format=torch.contiguous_format)

    if xpose_args == _XposeArgs.lhs:
        lhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(lhs)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    elif xpose_args == _XposeArgs.rhs:
        rhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(rhs)
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
    elif xpose_args is None:
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    else:
        raise ValueError(f"Invalid xpose_args: {xpose_args}")

    i_ready = 0
    i_recv = i_ready ^ 1
    lhs_buffer = [lhs_recv, lhs]
    rhs_buffer = [rhs_recv, rhs]

    if xpose_args is not None:
        comm.comm_2d_trans.wait_until_finished()

    lhs_buffer[i_recv] = comm.comm_row_init.enqueue_to_dispatch(
        lhs_buffer[i_ready], lhs_buffer[i_recv]
    )
    rhs_buffer[i_recv] = comm.comm_col_init.enqueue_to_dispatch(
        rhs_buffer[i_ready], rhs_buffer[i_recv]
    )

    i_ready ^= 1
    i_recv ^= 1

    out = torch.zeros_like(lhs_buffer[i_ready])

    comm.comm_row_init.wait_until_finished()
    comm.comm_col_init.wait_until_finished()

    for k_step in range(comm.group_layout.shape[1]):
        lhs_ready = lhs_buffer[i_ready]
        rhs_ready = rhs_buffer[i_ready]
        if k_step < comm.group_layout.shape[1] - 1:
            lhs_buffer[i_recv] = comm.comm_row.enqueue_to_dispatch(
                lhs_ready, lhs_buffer[i_recv]
            )
            rhs_buffer[i_recv] = comm.comm_col.enqueue_to_dispatch(
                rhs_ready, rhs_buffer[i_recv]
            )
        out = out + torch.matmul(lhs_ready, rhs_ready)
        if k_step < comm.group_layout.shape[1] - 1:
            comm.comm_row.wait_until_finished()
            comm.comm_col.wait_until_finished()
            i_ready ^= 1
            i_recv ^= 1

    if permute_out is not None:
        out = out.permute(permute_out)
    return out


# ---------------------------------------------------------------------------
# Autograd function for distributed triangle multiplication
# ---------------------------------------------------------------------------


class _TriangleMultiplicativeBlockImpl(torch.autograd.Function):
    """Distributed triangle multiplication autograd function.

    Forward
    -------
    Input x has shape (B, N_local_row, N_local_col, 2*d) and is already:
        x = signal * sigmoid(gate_logits) * visibility_mask  (pre-combined inner gate + mask)

    The function splits x into a = x[..., :d] and b = x[..., d:], then computes
    the triangle multiplication using ring communication.

    Backward
    --------
    Propagates gradients back through the distributed BMM.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x: DTensor, comm: Ring2DComm, direction: _Direction) -> DTensor:
        if not isinstance(x, DTensor):
            raise TypeError(f"x must be DTensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"x must be 4D, got {x.ndim}D")
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"Last dim of x must be even, got {x.shape[-1]}")
        placements = x.placements
        if placements != (Shard(0), Shard(1), Shard(2)):
            raise ValueError(
                f"x must have placements (Shard(0), Shard(1), Shard(2)), got {placements}"
            )

        x_local = x.to_local()
        a_local, b_local = torch.chunk(x_local, 2, dim=-1)
        a_local = a_local.clone(memory_format=torch.contiguous_format)
        b_local = b_local.clone(memory_format=torch.contiguous_format)

        if x.requires_grad:
            ctx.save_for_backward(a_local, b_local)
            ctx.comm = comm
            ctx.shape_x = x.shape
            ctx.stride_x = x.stride()
            ctx.placements = placements
            ctx.device_mesh = x.device_mesh
            ctx.direction = direction

        if direction == _Direction.Outgoing:
            # contracted[b,n,m,d] = sum_k a[b,n,k,d] * b[b,m,k,d]
            # a: (B,n,k,d) → permute to (B,D,n,k)
            # b: (B,m,k,d) → permute to (B,D,k,m) (needs transpose of (B,D,n,k))
            permute_lhs = (0, 3, 1, 2)
            permute_rhs = (0, 3, 2, 1)
            permute_out = (0, 2, 3, 1)
            xpose_args = _XposeArgs.rhs
        elif direction == _Direction.Incoming:
            # contracted[b,n,m,d] = sum_k a[b,k,n,d] * b[b,k,m,d]
            # a: (B,k,n,d) → permute to (B,D,n,k)  => need (0,3,2,1)
            # b: (B,k,m,d) → permute to (B,D,k,m)  => need (0,3,1,2)
            permute_lhs = (0, 3, 2, 1)
            permute_rhs = (0, 3, 1, 2)
            permute_out = (0, 2, 3, 1)
            xpose_args = _XposeArgs.lhs
        else:
            raise ValueError(f"Invalid direction: {direction}")

        out_local = _distributed_bmm(
            a_local,
            b_local,
            comm,
            permute_lhs=permute_lhs,
            permute_rhs=permute_rhs,
            permute_out=permute_out,
            xpose_args=xpose_args,
        ).contiguous()

        # Output has shape (B, N_local_row, N_local_col, d)
        shape_output = x.shape[:-1] + (out_local.shape[-1],)
        stride_output = update_exhaustive_strides(x.shape, x.stride(), shape_output)
        return DTensor.from_local(
            out_local,
            device_mesh=x.device_mesh,
            placements=placements,
            shape=shape_output,
            stride=stride_output,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, d_out: DTensor) -> Tuple[DTensor, None, None]:
        if not isinstance(d_out, DTensor):
            raise TypeError(f"d_out must be DTensor, got {type(d_out)}")

        a, b = ctx.saved_tensors
        comm = ctx.comm
        direction = ctx.direction
        d_out_local = d_out.to_local().to(dtype=a.dtype)

        if direction == _Direction.Outgoing:
            # d_a: d_out[b,n,m,d] * b[b,m,k,d] -> d_a[b,n,k,d]
            # permute: d_out (B,n,m,D)->(B,D,n,m); b (B,m,k,D)->(B,D,m,k); out (B,D,n,k)->(B,n,k,D)
            lhs_da, rhs_da = d_out_local, b
            permute_lhs_da = (0, 3, 1, 2)
            permute_rhs_da = (0, 3, 1, 2)
            permute_out_da = (0, 2, 3, 1)
            xpose_da = None

            # d_b: d_out[b,n,m,d] * a[b,n,k,d] -> d_b[b,m,k,d]
            # permute: d_out (B,n,m,D)->(B,D,m,n); a (B,n,k,D)->(B,D,n,k); out (B,D,m,k)->(B,m,k,D)
            lhs_db, rhs_db = d_out_local, a
            permute_lhs_db = (0, 3, 2, 1)
            permute_rhs_db = (0, 3, 1, 2)
            permute_out_db = (0, 2, 3, 1)
            xpose_db = _XposeArgs.lhs

        elif direction == _Direction.Incoming:
            # d_a: d_out[b,n,m,d] * b[b,k,m,d] -> d_a[b,k,n,d]
            lhs_da, rhs_da = b, d_out_local
            permute_lhs_da = (0, 3, 1, 2)
            permute_rhs_da = (0, 3, 2, 1)
            permute_out_da = (0, 2, 3, 1)
            xpose_da = _XposeArgs.rhs

            # d_b: d_out[b,n,m,d] * a[b,k,n,d] -> d_b[b,k,m,d]
            lhs_db, rhs_db = a, d_out_local
            permute_lhs_db = (0, 3, 1, 2)
            permute_rhs_db = (0, 3, 1, 2)
            permute_out_db = (0, 2, 3, 1)
            xpose_db = None
        else:
            raise ValueError(f"Invalid direction: {direction}")

        da_local = _distributed_bmm(
            lhs_da,
            rhs_da,
            comm,
            permute_lhs=permute_lhs_da,
            permute_rhs=permute_rhs_da,
            permute_out=permute_out_da,
            xpose_args=xpose_da,
        ).contiguous()
        db_local = _distributed_bmm(
            lhs_db,
            rhs_db,
            comm,
            permute_lhs=permute_lhs_db,
            permute_rhs=permute_rhs_db,
            permute_out=permute_out_db,
            xpose_args=xpose_db,
        ).contiguous()

        dab_local = torch.cat([da_local, db_local], dim=-1)
        dx = DTensor.from_local(
            dab_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.placements,
            shape=ctx.shape_x,
            stride=ctx.stride_x,
        )
        return dx, None, None


# ---------------------------------------------------------------------------
# Public distributed module
# ---------------------------------------------------------------------------


class TriangleMultiplicativeBlockDistributed(nn.Module):
    """Distributed TriangleMultiplicativeBlock for ESMFold2's pair representation.

    Replaces the serial layer in a model by:
    1. Replacing all parameters with DTensor replicated parameters.
    2. Implementing the forward pass using distributed ring-communication BMM.

    The pair tensor z is expected as a DTensor with placements
    (Shard(0), Shard(1), Shard(2)) on a 3D mesh (dp, cp_axis_0, cp_axis_1).

    Parameters
    ----------
    layer:
        The serial TriangleMultiplicativeBlock to distribute.
    device_mesh:
        The device mesh (should be the subgroups mesh: dp × cp_axis_0 × cp_axis_1).
    comm:
        Ring2DComm for the CP group.
    """

    def __init__(
        self,
        layer: SerialTriangleMultiplicativeBlock,
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialTriangleMultiplicativeBlock):
            raise TypeError(
                f"layer must be TriangleMultiplicativeBlock, got {type(layer).__name__}"
            )
        self.device_mesh = device_mesh
        self.ring_comm = comm

        self._direction = (
            _Direction.Outgoing if layer.flow == "outgoing" else _Direction.Incoming
        )
        self._latent_channels = layer.latent_channels

        self.norm_start = LayerNormParamsReplicated(layer.norm_start, device_mesh)
        self.norm_mix = LayerNormParamsReplicated(layer.norm_mix, device_mesh)
        self.proj_bundle = LinearParamsReplicated(layer.proj_bundle, device_mesh)
        self.proj_emit = LinearParamsReplicated(layer.proj_emit, device_mesh)
        self.proj_gate = LinearParamsReplicated(layer.proj_gate, device_mesh)

    def forward(self, pair: DTensor, mask: DTensor | None = None) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        pair:
            Pair tensor (B, N, N, d_pair) as DTensor(Shard(0), Shard(1), Shard(2)).
        mask:
            Visibility mask (B, N, N) as DTensor(Shard(0), Shard(1), Shard(2)).
            If None, no masking is applied.

        Returns
        -------
        DTensor of same shape and placements as pair.
        """
        # 1. Layer-normalise input
        normalized = self.norm_start(pair)

        # 2. Compute bundled projection: (d → 4*d)
        bundled = self.proj_bundle(normalized)

        # 3. Split into signal (2*d) and inner gate logits (2*d); apply inner gate
        latent = self._latent_channels
        signal, gate_logits = bundled.split(2 * latent, dim=-1)  # DTensor ops
        x = signal * gate_logits.sigmoid()

        # 4. Apply visibility mask
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # 5. Distributed triangle multiplication
        contracted = _TriangleMultiplicativeBlockImpl.apply(
            x, self.ring_comm, self._direction
        )

        # 6. Norm + output projection
        out = self.proj_emit(self.norm_mix(contracted))

        # 7. Output gate (applied to pre-norm input)
        output_gate = self.proj_gate(normalized).sigmoid()
        return out * output_gate
