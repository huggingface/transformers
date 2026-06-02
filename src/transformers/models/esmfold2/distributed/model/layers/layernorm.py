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

"""Distributed LayerNorm with parameters replicated across the device mesh."""

from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, distribute_tensor

_shape_t = Union[int, list[int], torch.Size]


class _LayerNormParamsReplicatedImpl(torch.autograd.Function):
    """LayerNorm with replicated parameters and arbitrary DTensor input placement."""

    @staticmethod
    def forward(
        ctx,
        x: DTensor,
        normalized_shape: list[int],
        weight: Optional[DTensor],
        bias: Optional[DTensor],
        eps: float,
        reduce_group: dist.ProcessGroup,
    ) -> DTensor:
        if not isinstance(x, DTensor):
            raise TypeError(f"x must be DTensor, got {type(x)}")

        x_local = x.to_local()
        weight_local = weight.to_local() if weight is not None else None
        bias_local = bias.to_local() if bias is not None else None

        ctx.reduce_group = reduce_group
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        ctx.x_shape = x.shape
        ctx.x_stride = x.stride()
        ctx.x_placements = x.placements
        ctx.device_mesh = x.device_mesh
        ctx.save_for_backward(x_local, weight_local)

        out_local = F.layer_norm(
            x_local, normalized_shape, weight_local, bias_local, eps
        )
        return DTensor.from_local(
            out_local,
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )

    @staticmethod
    def backward(ctx, d_out: DTensor):
        if not isinstance(d_out, DTensor):
            raise TypeError(f"d_out must be DTensor, got {type(d_out)}")

        x_saved, weight_saved = ctx.saved_tensors
        d_out_local = d_out.to_local()
        eps = ctx.eps
        normalized_shape = ctx.normalized_shape

        dx_dtensor: Optional[DTensor] = None
        dw_dtensor: Optional[DTensor] = None
        db_dtensor: Optional[DTensor] = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            dims = tuple(-(i + 1) for i in range(len(normalized_shape)))
            mean = x_saved.mean(dim=dims, keepdim=True)
            var = x_saved.var(dim=dims, unbiased=False, keepdim=True)
            x_norm = (x_saved - mean) / torch.sqrt(var + eps)

        if ctx.needs_input_grad[0]:
            if weight_saved is not None:
                dy = d_out_local * weight_saved.view(
                    *([1] * (d_out_local.ndim - len(normalized_shape))),
                    *weight_saved.shape,
                )
            else:
                dy = d_out_local
            dims = tuple(-(i + 1) for i in range(len(normalized_shape)))
            dy_mean = dy.mean(dim=dims, keepdim=True)
            dy_x_norm_mean = (dy * x_norm).mean(dim=dims, keepdim=True)
            dx_local = (dy - dy_mean - x_norm * dy_x_norm_mean) / torch.sqrt(var + eps)
            dx_dtensor = DTensor.from_local(
                dx_local,
                device_mesh=ctx.device_mesh,
                placements=ctx.x_placements,
                shape=ctx.x_shape,
                stride=ctx.x_stride,
            )

        if ctx.needs_input_grad[2]:
            reduce_dims = list(range(d_out_local.ndim - len(normalized_shape)))
            dw = (d_out_local * x_norm).sum(dim=reduce_dims).contiguous()
            dw_work = dist.all_reduce(
                dw, op=dist.ReduceOp.SUM, group=ctx.reduce_group, async_op=True
            )

        if ctx.needs_input_grad[3]:
            reduce_dims = list(range(d_out_local.ndim - len(normalized_shape)))
            db = d_out_local.sum(dim=reduce_dims).contiguous()
            db_work = dist.all_reduce(
                db, op=dist.ReduceOp.SUM, group=ctx.reduce_group, async_op=True
            )

        replicate = [Replicate()] * ctx.device_mesh.ndim
        if ctx.needs_input_grad[2]:
            dw_work.wait()  # type: ignore[union-attr]
            dw_dtensor = DTensor.from_local(
                dw,
                device_mesh=ctx.device_mesh,
                placements=replicate,
                shape=dw.shape,
                stride=dw.stride(),
            )
        if ctx.needs_input_grad[3]:
            db_work.wait()  # type: ignore[union-attr]
            db_dtensor = DTensor.from_local(
                db,
                device_mesh=ctx.device_mesh,
                placements=replicate,
                shape=db.shape,
                stride=db.stride(),
            )

        return dx_dtensor, None, dw_dtensor, db_dtensor, None, None


class LayerNormParamsReplicated(nn.Module):
    """nn.LayerNorm wrapper with parameters replicated over the device mesh.

    Accepts DTensor inputs with arbitrary placements and outputs DTensors
    with the same placements.

    Parameters
    ----------
    layer_local:
        The serial nn.LayerNorm layer whose parameters to replicate.
    device_mesh:
        The device mesh for distributing tensors.
    """

    def __init__(self, layer_local: nn.LayerNorm, device_mesh: DeviceMesh) -> None:
        super().__init__()
        if not isinstance(layer_local, nn.LayerNorm):
            raise TypeError(
                f"layer_local must be nn.LayerNorm, got {type(layer_local)}"
            )

        self.device_mesh = device_mesh
        self.normalized_shape = list(layer_local.normalized_shape)
        self.eps = layer_local.eps
        replicate_placements = [Replicate()] * device_mesh.ndim

        if layer_local.weight is not None:
            self.weight = nn.Parameter(
                distribute_tensor(layer_local.weight, device_mesh, replicate_placements)
            )
        else:
            self.weight = None

        if layer_local.bias is not None:
            self.bias = nn.Parameter(
                distribute_tensor(layer_local.bias, device_mesh, replicate_placements)
            )
        else:
            self.bias = None

        if "cp" in device_mesh.mesh_dim_names:  # type: ignore[operator]
            self._reduce_group = device_mesh.get_group("cp")
        else:
            self._reduce_group = dist.group.WORLD

    def forward(self, x: DTensor) -> DTensor:
        return _LayerNormParamsReplicatedImpl.apply(  # type: ignore[return-value]
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            self._reduce_group,
        )
