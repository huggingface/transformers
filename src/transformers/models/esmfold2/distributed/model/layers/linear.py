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

"""Distributed linear layer with parameters replicated across the device mesh."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from projects.huggingface.transformers.models.esmfold2.distributed.utils import (
    update_exhaustive_strides,
)


class _LinearParamsReplicatedImpl(torch.autograd.Function):
    """Linear layer with replicated parameters and arbitrary input placements.

    Parameters are replicated on all mesh dimensions; inputs can be sharded
    (e.g. Shard(0), Shard(1), Shard(2)) or replicated.  Backward uses
    all-reduce over the cp group to accumulate parameter gradients.
    """

    @staticmethod
    def forward(
        ctx,
        x: DTensor,
        weight: DTensor,
        bias: Optional[DTensor],
        reduce_group: dist.ProcessGroup,
        avg_reduce: bool,
    ) -> DTensor:
        if not isinstance(x, DTensor):
            raise TypeError(f"x must be DTensor, got {type(x)}")
        if not isinstance(weight, DTensor):
            raise TypeError(f"weight must be DTensor, got {type(weight)}")

        ctx.reduce_group = reduce_group
        ctx.avg_reduce = avg_reduce

        x_local = x.to_local()
        weight_local = weight.to_local()
        bias_local = bias.to_local() if bias is not None else None

        ctx.save_for_backward(x_local, weight_local)
        ctx.x_shape = x.shape
        ctx.x_stride = x.stride()
        ctx.x_placements = x.placements
        ctx.device_mesh = x.device_mesh

        out_local = F.linear(x_local, weight_local, bias_local)

        shape_output = x.shape[:-1] + (weight.shape[0],)
        stride_output = update_exhaustive_strides(x.shape, x.stride(), shape_output)
        return DTensor.from_local(
            out_local,
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=shape_output,
            stride=stride_output,
        )

    @staticmethod
    def backward(ctx, d_out: DTensor):
        if not isinstance(d_out, DTensor):
            raise TypeError(f"d_out must be DTensor, got {type(d_out)}")

        x_saved, weight_saved = ctx.saved_tensors
        d_out_local = d_out.to_local()

        dw: Optional[Tensor] = None
        dw_work = None
        if ctx.needs_input_grad[1]:
            # Aggregate over all but the last two dims (batch + seq dims)
            dw = torch.einsum("...i,...j->ij", d_out_local, x_saved)
            dw = dw.contiguous()  # type: ignore[union-attr]
            op = dist.ReduceOp.AVG if ctx.avg_reduce else dist.ReduceOp.SUM
            dw_work = dist.all_reduce(dw, op=op, group=ctx.reduce_group, async_op=True)

        db: Optional[Tensor] = None
        db_work = None
        if ctx.needs_input_grad[2]:
            reduce_dims = list(range(d_out_local.ndim - 1))
            db = d_out_local.sum(dim=reduce_dims).contiguous()
            op = dist.ReduceOp.AVG if ctx.avg_reduce else dist.ReduceOp.SUM
            db_work = dist.all_reduce(db, op=op, group=ctx.reduce_group, async_op=True)

        dx_dtensor: Optional[DTensor] = None
        if ctx.needs_input_grad[0]:
            dx_local = F.linear(d_out_local, weight_saved.t())
            shape_dx = ctx.x_shape
            stride_dx = ctx.x_stride
            dx_dtensor = DTensor.from_local(
                dx_local,
                device_mesh=ctx.device_mesh,
                placements=ctx.x_placements,
                shape=shape_dx,
                stride=stride_dx,
            )

        # Wrap parameter gradients as DTensors with Replicate placement
        replicate = [Replicate()] * ctx.device_mesh.ndim
        dw_dtensor: Optional[DTensor] = None
        if dw_work is not None:
            dw_work.wait()
            dw_dtensor = DTensor.from_local(
                dw,  # type: ignore[arg-type]
                device_mesh=ctx.device_mesh,
                placements=replicate,
                shape=dw.shape,  # type: ignore[union-attr]
                stride=dw.stride(),  # type: ignore[union-attr]
            )

        db_dtensor: Optional[DTensor] = None
        if db_work is not None:
            db_work.wait()
            db_dtensor = DTensor.from_local(
                db,  # type: ignore[arg-type]
                device_mesh=ctx.device_mesh,
                placements=replicate,
                shape=db.shape,  # type: ignore[union-attr]
                stride=db.stride(),  # type: ignore[union-attr]
            )

        return dx_dtensor, dw_dtensor, db_dtensor, None, None


class LinearParamsReplicated(nn.Module):
    """nn.Linear wrapper with parameters replicated over the device mesh.

    Accepts DTensor inputs with arbitrary placements and outputs DTensors
    with the same placements.  Parameter gradients are all-reduced across
    the CP group so every rank accumulates the full gradient.

    Parameters
    ----------
    layer_local:
        The serial nn.Linear layer whose parameters to replicate.
    device_mesh:
        The device mesh for distributing tensors.
    avg_reduce:
        If True, use AVG instead of SUM for all-reduce (useful when the
        effective batch is already averaged).
    """

    def __init__(
        self, layer_local: nn.Linear, device_mesh: DeviceMesh, avg_reduce: bool = False
    ) -> None:
        super().__init__()
        if not isinstance(layer_local, nn.Linear):
            raise TypeError(f"layer_local must be nn.Linear, got {type(layer_local)}")

        self.device_mesh = device_mesh
        self.avg_reduce = avg_reduce
        replicate_placements = [Replicate()] * device_mesh.ndim

        self.weight = nn.Parameter(
            distribute_tensor(layer_local.weight, device_mesh, replicate_placements)
        )
        if layer_local.bias is not None:
            self.bias = nn.Parameter(
                distribute_tensor(layer_local.bias, device_mesh, replicate_placements)
            )
        else:
            self.bias = None

        # Choose reduce group: use cp group if present, otherwise world
        if "cp" in device_mesh.mesh_dim_names:  # type: ignore[operator]
            self._reduce_group = device_mesh.get_group("cp")
        else:
            self._reduce_group = dist.group.WORLD

    def forward(self, x: DTensor) -> DTensor:
        return _LinearParamsReplicatedImpl.apply(  # type: ignore[return-value]
            x, self.weight, self.bias, self._reduce_group, self.avg_reduce
        )
