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

"""Distributed PairUpdateBlock and Pairformer for ESMFold2.

ESMFold2's Pairformer is pair-only (no single/sequence track), with each block:
    PairUpdateBlock:
        pair = pair + tri_mul_out(pair)
        pair = pair + tri_mul_in(pair)
        pair = pair_transition(pair)

All three operations are distributed across the 2D CP grid.
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from projects.huggingface.transformers.models.esmfold2.distributed.comm import (
    Ring2DComm,
)
from projects.huggingface.transformers.models.esmfold2.distributed.manager import (
    DistributedManager,
)
from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.linear import (
    LinearParamsReplicated,
)
from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.triangular_mult import (
    TriangleMultiplicativeBlockDistributed,
)
from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
    FoldingTrunk as SerialFoldingTrunk,
)
from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
    PairUpdateBlock as SerialPairUpdateBlock,
)
from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
    Transition as SerialTransition,
)


class TransitionDistributed(nn.Module):
    """Distributed Transition block (LayerNorm + SwiGLU FFN).

    Parameters are replicated; the LayerNorm and both Linear layers
    use DTensor with replicated params.
    """

    def __init__(self, layer: SerialTransition, device_mesh: DeviceMesh) -> None:
        super().__init__()
        if not isinstance(layer, SerialTransition):
            raise TypeError(f"layer must be Transition, got {type(layer).__name__}")
        self.norm = LayerNormParamsReplicated(layer.norm, device_mesh)
        # SwiGLUMLP has w12 and w3
        self.w12 = LinearParamsReplicated(layer.ffn.w12, device_mesh)
        self.w3 = LinearParamsReplicated(layer.ffn.w3, device_mesh)
        self.hidden_features = layer.ffn.hidden_features

    def forward(self, x: DTensor) -> DTensor:
        normed = self.norm(x)
        x12 = self.w12(normed)
        x1, x2 = x12.split(self.hidden_features, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)
        return x + out


class PairUpdateBlockDistributed(nn.Module):
    """Distributed PairUpdateBlock.

    Computes:
        pair = pair + tri_mul_out(pair, mask)
        pair = pair + tri_mul_in(pair, mask)
        pair = pair_transition(pair)

    All pair operations are distributed via 2D CP ring communication.

    Parameters
    ----------
    layer:
        Serial PairUpdateBlock to distribute.
    dist_manager:
        DistributedManager with the CP group and subgroups set up.
    """

    def __init__(
        self, layer: SerialPairUpdateBlock, dist_manager: DistributedManager
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialPairUpdateBlock):
            raise TypeError(
                f"layer must be PairUpdateBlock, got {type(layer).__name__}"
            )

        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh_subgroups

        ring_comm_out = Ring2DComm(
            dist_manager.group["cp"],
            dist_manager.subgroups["cp"][0],
            dist_manager.layout_subgroups["cp"],
        )
        ring_comm_in = Ring2DComm(
            dist_manager.group["cp"],
            dist_manager.subgroups["cp"][0],
            dist_manager.layout_subgroups["cp"],
        )

        self.tri_mul_out = TriangleMultiplicativeBlockDistributed(
            layer.tri_mul_out._engine, self.device_mesh, ring_comm_out
        )
        self.tri_mul_in = TriangleMultiplicativeBlockDistributed(
            layer.tri_mul_in._engine, self.device_mesh, ring_comm_in
        )
        self.pair_transition = TransitionDistributed(
            layer.pair_transition, self.device_mesh
        )

    def forward(
        self, pair: DTensor, pair_attention_mask: Optional[DTensor] = None
    ) -> DTensor:
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class FoldingTrunkDistributed(nn.Module):
    """Distributed Pairformer: ModuleList of PairUpdateBlockDistributed.

    Wraps the serial Pairformer by distributing each block.

    Parameters
    ----------
    pairformer:
        Serial Pairformer module.
    dist_manager:
        DistributedManager with the CP group and subgroups set up.
    """

    def __init__(
        self, trunk: SerialFoldingTrunk, dist_manager: DistributedManager
    ) -> None:
        super().__init__()
        if not isinstance(trunk, SerialFoldingTrunk):
            raise TypeError(f"trunk must be FoldingTrunk, got {type(trunk).__name__}")

        self.blocks = nn.ModuleList(
            [PairUpdateBlockDistributed(block, dist_manager) for block in trunk.blocks]  # type: ignore[arg-type]
        )

    def forward(
        self, pair: DTensor, pair_attention_mask: Optional[DTensor] = None
    ) -> DTensor:
        for block in self.blocks:
            pair = block(pair, pair_attention_mask=pair_attention_mask)
        return pair
