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

"""Layout / sharding helpers and the user-facing model-wrap entry point for
2D context parallelism."""

from math import isqrt, lcm
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import Shard, distribute_tensor


class LayoutMap:
    """Bijective mapping between multi-dimensional indices and flat indices.

    Analogous to C++ std::layout_stride::mapping.

    Parameters
    ----------
    strides:
        Per-dimension strides (must be positive integers).
    shape:
        Per-dimension sizes (must be positive integers).
    offset:
        Offset added to every flat index (default 0).
    """

    def __init__(
        self, strides: tuple[int, ...], shape: tuple[int, ...], offset: int = 0
    ):
        if not all(isinstance(s, (int, np.int64)) and s > 0 for s in strides):  # type: ignore[arg-type]
            raise ValueError(f"Strides must be positive integers: {strides}")
        if any(s < 0 for s in shape):
            raise ValueError(f"Shape must be non-negative: {shape}")
        if any(s == 0 for s in shape):
            raise ValueError(f"Shape must not contain zeros: {shape}")

        self._strides = strides
        self._n_axes = len(strides)
        if len(shape) != self._n_axes:
            raise ValueError(
                f"Shape {shape} and strides {strides} must have the same length"
            )

        self._shape = shape
        self._numel = int(np.prod(self._shape))
        self._offset = offset

        shape_and_strides = np.array(
            list(zip(self._shape, self._strides)),
            dtype=np.dtype([("shape", int), ("strides", int)]),
        )
        argsort_ascend = np.argsort(shape_and_strides, order=["strides", "shape"])

        self.is_unique = self._is_unique(argsort_ascend)
        self.is_exhaustive = self._is_exhaustive(argsort_ascend)

        if not self.is_unique:
            raise ValueError(
                f"Strides {strides} and shape {shape} do not give a unique layout."
            )

        self._required_span_size = self._compute_required_span_size()
        self._argsort_descend_strides = argsort_ascend[::-1]
        self._argsort_ascend_strides = argsort_ascend

    def _compute_required_span_size(self) -> int:
        if self._n_axes == 0:
            return 1
        return 1 + sum(
            (self._shape[i] - 1) * self._strides[i] for i in range(self._n_axes)
        )

    def _strides_exhaustive(self, permutation: np.ndarray):
        strides = np.array(self._strides)
        shape = np.array(self._shape)
        shape_permuted = shape[permutation]
        strides_permuted = strides[permutation]
        shape_shifted = np.concatenate([[1], shape_permuted[:-1]])
        strides_shifted = np.concatenate([[1], strides_permuted[:-1]])
        return strides_permuted, strides_shifted * shape_shifted

    def _is_unique(self, permutation: np.ndarray) -> bool:
        if self._n_axes == 0:
            return True
        strides, strides_exhaustive = self._strides_exhaustive(permutation)
        return bool(np.all(strides >= strides_exhaustive))

    def _is_exhaustive(self, permutation: np.ndarray) -> bool:
        if self._n_axes == 0:
            return True
        strides, strides_exhaustive = self._strides_exhaustive(permutation)
        return bool(np.all(strides == strides_exhaustive))

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def required_span_size(self) -> int:
        return self._required_span_size

    @property
    def numel(self) -> int:
        return self._numel

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    def __call__(self, ids: tuple[int, ...]) -> int:
        if len(ids) != self._n_axes:
            raise ValueError(
                f"Expected {self._n_axes} elements in ids but got {len(ids)}"
            )
        if len(ids) == 0:
            return self._offset
        if self._shape is not None:
            for axis, idx in enumerate(ids):
                if idx < 0 or idx >= self._shape[axis]:
                    raise ValueError(
                        f"ids[{axis}] == {idx} out of range [0, {self._shape[axis] - 1}]"
                    )
        return int(np.dot(ids, self._strides)) + self._offset

    def unravel(self, flat_index: int) -> tuple[int, ...]:
        if not self.is_unique:
            raise ValueError(f"Layout is not unique, cannot unravel {flat_index}")
        if not isinstance(flat_index, (int, np.integer)):
            raise TypeError(f"Expected int, got {type(flat_index)}")
        remaining = flat_index - self._offset
        if remaining < 0 or remaining >= self._required_span_size:
            raise ValueError(
                f"flat_index {flat_index} out of range [{self._offset}, "
                f"{self._offset + self._required_span_size - 1}]"
            )
        indices = [0] * self._n_axes
        for i_dim in self._argsort_descend_strides:
            stride = self._strides[i_dim]
            size = self._shape[i_dim]
            indices[i_dim] = (remaining // stride) % size
            remaining -= indices[i_dim] * stride
        if remaining != 0:
            raise ValueError(f"flat_index {flat_index} is out of the valid span range.")
        return tuple(indices)

    def __getitem__(self, slices) -> "LayoutMap":
        if not isinstance(slices, tuple) and isinstance(slices, (slice, int)):
            slices = (slices,)
        if len(slices) < self._n_axes:
            slices = slices + (slice(None),) * (self._n_axes - len(slices))

        new_shape = []
        new_strides = []
        new_offset = self.offset

        for axis, s in enumerate(slices):
            if isinstance(s, (int, np.int64)):  # type: ignore[arg-type]
                new_offset += s * self.strides[axis]  # type: ignore[operator]
            elif isinstance(s, slice):
                start, stop, step = s.indices(self.shape[axis])
                if step <= 0:
                    raise ValueError("Unsupported slicing: negative or zero steps")
                if start >= stop:
                    raise ValueError("Unsupported slicing: start not smaller than stop")
                dim_len = max(0, (stop - start + step - 1) // step)
                new_shape.append(dim_len)
                new_strides.append(self.strides[axis] * step)
                new_offset += start * self.strides[axis]
            else:
                raise TypeError(f"Unsupported slice type: {type(s)}")

        return LayoutMap(tuple(new_strides), tuple(new_shape), new_offset)


class LayoutRightMap(LayoutMap):
    """Row-major (C-contiguous) layout."""

    def __init__(self, shape: tuple[int, ...]):
        strides = np.ones_like(shape)
        strides[1:] = shape[:0:-1]
        strides = np.cumprod(strides)[::-1]
        super().__init__(tuple(strides), shape=shape)


class LayoutLeftMap(LayoutMap):
    """Column-major (Fortran-contiguous) layout."""

    def __init__(self, shape: tuple[int, ...]):
        strides = np.ones_like(shape)
        strides[1:] = shape[:-1]
        strides = np.cumprod(strides)
        super().__init__(tuple(strides), shape=shape)


def get_group_rank_from_axial_shift(
    coord: tuple[int, ...], axis: int, delta: int, layout_group: LayoutMap
) -> int:
    """Return the rank obtained by shifting coord along axis by delta (wrapping)."""
    if len(coord) != len(layout_group.shape):
        raise ValueError(
            f"Incompatible coord {coord} and layout_group shape {layout_group.shape}"
        )
    if axis >= len(coord):
        raise ValueError(f"Axis {axis} out of range for coord {coord}")
    coord_shifted = list(coord)
    coord_shifted[axis] = (coord_shifted[axis] + delta) % layout_group.shape[axis]
    return layout_group(coord_shifted)  # type: ignore[arg-type]


def update_exhaustive_strides(
    shape_original: Sequence[int],
    strides_original: Sequence[int],
    shape_new: Sequence[int],
) -> tuple[int, ...]:
    """Compute strides for shape_new that preserve the same axis-ordering as
    the exhaustive layout (shape_original, strides_original)."""
    layout_original = LayoutMap(tuple(strides_original), tuple(shape_original))
    if not layout_original.is_exhaustive:
        raise ValueError(
            f"Layout (shape={shape_original}, strides={strides_original}) is not exhaustive"
        )
    shape_new_ascending = np.array(shape_new)[layout_original._argsort_ascend_strides]
    argsort_output = np.argsort(layout_original._argsort_ascend_strides)
    strides_new_ascending = np.concatenate(([1], shape_new_ascending[:-1])).cumprod()
    strides_new = strides_new_ascending[argsort_output]
    return tuple(strides_new.tolist())


def slice_repr_mask(
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    pair_mask: torch.Tensor,
    n_ranks: int,
    layout_group: LayoutMap,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    """Slice s, z, mask, pair_mask into n_ranks shards for 2D CP distribution."""
    if z.shape[-2] != z.shape[-3]:
        raise ValueError(f"z is not square in the middle two axes: {z.shape}")
    if s.shape[-2] != z.shape[-3]:
        raise ValueError(f"Incompatible s {s.shape} and z {z.shape}")
    if mask.shape != s.shape[:-1]:
        raise ValueError(f"Incompatible s {s.shape} and mask {mask.shape}")
    if pair_mask.shape != z.shape[:-1]:
        raise ValueError(f"Incompatible z {z.shape} and pair_mask {pair_mask.shape}")

    n_tokens = s.shape[-2]
    coords = [layout_group.unravel(rank) for rank in range(n_ranks)]
    n_ranks_axis = isqrt(n_ranks)
    if n_ranks_axis * n_ranks_axis != n_ranks:
        raise ValueError(f"n_ranks is not a perfect square: {n_ranks}")
    if n_tokens % n_ranks_axis:
        raise ValueError(
            f"Token dim {n_tokens} not divisible by sqrt(n_ranks) = {n_ranks_axis}"
        )
    stride = n_tokens // n_ranks_axis
    s_slices, z_slices, mask_slices, pair_mask_slices = [], [], [], []
    for i_row, j_col in coords:
        i0, i1 = i_row * stride, (i_row + 1) * stride
        j0, j1 = j_col * stride, (j_col + 1) * stride
        s_slices.append(s[..., i0:i1, :].contiguous())
        mask_slices.append(mask[..., i0:i1].contiguous())
        z_slices.append(z[..., i0:i1, j0:j1, :].contiguous())
        pair_mask_slices.append(pair_mask[..., i0:i1, j0:j1].contiguous())
    return s_slices, z_slices, mask_slices, pair_mask_slices


def tiled_softmax_attention_update(
    o_chunk: torch.Tensor,
    lse_m_chunk: torch.Tensor,
    amax_chunk: torch.Tensor | None,
    o: torch.Tensor | None = None,
    lse_m: torch.Tensor | None = None,
    amax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Numerically-stable online softmax accumulation for ring attention.

    Updates running (o, lse_m, amax) with a new (o_chunk, lse_m_chunk, amax_chunk).
    When amax_chunk is None the function operates without amax tracking.
    """
    if not ((o is None) == (lse_m is None)):
        raise ValueError("o and lse_m must both be None or both provided")

    has_amax = amax_chunk is not None
    is_initial_chunk = o is None

    if lse_m_chunk.shape[-1] != 1:
        raise ValueError("lse_m_chunk must have shape (..., 1)")
    if o_chunk.ndim != lse_m_chunk.ndim:
        raise ValueError("o_chunk and lse_m_chunk must have the same ndim")
    if lse_m_chunk.shape[:-1] != o_chunk.shape[:-1]:
        raise ValueError("o_chunk and lse_m_chunk must match except in the last dim")

    if is_initial_chunk:
        return o_chunk, lse_m_chunk, amax_chunk

    if has_amax:
        d_lse_m = lse_m - lse_m_chunk  # type: ignore[operator]
        amax_next = torch.maximum(amax_chunk, amax)  # type: ignore[arg-type]
        delta_lse = amax_chunk - amax - d_lse_m  # type: ignore[operator]
        o_new = o - torch.sigmoid(delta_lse) * (o - o_chunk)
        lse_m_new = lse_m_chunk + torch.logsumexp(
            torch.cat([(amax - amax_next) + d_lse_m, amax_chunk - amax_next], dim=-1),  # type: ignore[operator]
            dim=-1,
            keepdim=True,
        ).to(dtype=lse_m_chunk.dtype)
        return o_new, lse_m_new, amax_next
    else:
        d_lse_m = lse_m - lse_m_chunk  # type: ignore[operator]
        o_new = o - torch.sigmoid(-d_lse_m) * (o - o_chunk)
        lse_m_new = lse_m_chunk + torch.log1p(torch.exp(d_lse_m)).to(
            dtype=lse_m_chunk.dtype
        )
        return o_new, lse_m_new, None


# ---------------------------------------------------------------------------
# End-to-end CP runtime: drop-in replacement for the serial ``FoldingTrunk``.
# ---------------------------------------------------------------------------


class TrunkCPWrapper(nn.Module):
    """Drop-in replacement for ``FoldingTrunk`` that runs distributed.

    Accepts and returns plain tensors so the rest of an ESMFold2 model
    (LM, MSA encoder, diffusion sampler) is untouched. Pair tensors whose
    ``N`` is not a multiple of the CP axes are zero-padded; the gathered
    output is sliced back to the original length, and the mask is padded
    the same way so padded rows/cols contribute nothing.

    Typical use on an ``N×N`` CP grid (e.g. 4 ranks via
    ``torch.multiprocessing.spawn``)::

        from collections import OrderedDict

        from transformers.models.esmc import ESMFold2Model
        from transformers.models.esmc.distributed import (
            DistributedManager,
            wrap_model_with_cp_trunks,
        )

        DistributedManager.initialize(OrderedDict([("dp", 1), ("cp", (2, 2))]))
        dm = DistributedManager()

        model = ESMFold2Model.from_pretrained(...).cuda().eval()
        wrap_model_with_cp_trunks(model, dm)
        # ``model.forward`` (and ``processor.fold``) now run the Pairformer
        # across the CP grid; everything else stays serial per rank.
    """

    def __init__(self, serial_trunk: nn.Module, dist_manager) -> None:
        super().__init__()
        # Lazy imports: this module is imported by manager.py and the
        # distributed layers, so importing pairformer / model_common at
        # module level would create a cycle.
        from projects.huggingface.transformers.models.esmfold2.distributed.manager import (
            DistributedManager,
        )
        from projects.huggingface.transformers.models.esmfold2.distributed.model.layers.pairformer import (
            FoldingTrunkDistributed,
        )
        from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
            FoldingTrunk as SerialFoldingTrunk,
        )

        if not isinstance(serial_trunk, SerialFoldingTrunk):
            raise TypeError(f"expected FoldingTrunk, got {type(serial_trunk).__name__}")
        if not isinstance(dist_manager, DistributedManager):
            raise TypeError(
                f"expected DistributedManager, got {type(dist_manager).__name__}"
            )

        # ``FoldingTrunkDistributed`` requires the serial trunk's chunking
        # disabled (it composes its own ring loop). ``serial_trunk`` is typed
        # ``nn.Module`` (the SerialFoldingTrunk symbol is imported lazily inside
        # the function to avoid a circular import with pairformer.py); pyright
        # can't narrow through the lazy-import isinstance check.
        serial_trunk.set_chunk_size(None)  # type: ignore[operator]

        self.dist_trunk = FoldingTrunkDistributed(serial_trunk, dist_manager)
        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1)
        self.cp_axis_0 = self.device_mesh.size(1)
        self.cp_axis_1 = self.device_mesh.size(2)
        self.shard_factor = lcm(self.cp_axis_0, self.cp_axis_1)

    # The serial trunk exposes this knob to the parent model. The distributed
    # path doesn't support chunking, but the parent ``set_chunk_size`` call
    # still needs a no-op hook so it doesn't blow up.
    def set_chunk_size(self, _chunk_size: int | None) -> None:
        return

    def forward(
        self, pair: torch.Tensor, pair_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        N = pair.shape[1]
        pad = (self.shard_factor - N % self.shard_factor) % self.shard_factor
        if pad:
            # F.pad pads from the last dim backward; pair is (B, N, N, d_pair).
            pair = F.pad(pair, (0, 0, 0, pad, 0, pad))
            if pair_attention_mask is not None:
                pair_attention_mask = F.pad(pair_attention_mask, (0, pad, 0, pad))

        pair_dt = distribute_tensor(
            pair.contiguous(), self.device_mesh, [Shard(0), Shard(1), Shard(2)]
        )
        mask_dt = None
        if pair_attention_mask is not None:
            mask_dt = distribute_tensor(
                pair_attention_mask.contiguous(),
                self.device_mesh,
                [Shard(0), Shard(1), Shard(2)],
            )

        out_dt = self.dist_trunk(pair_dt, pair_attention_mask=mask_dt)
        out = out_dt.full_tensor()
        if pad:
            out = out[:, :N, :N, :]
        return out


def wrap_model_with_cp_trunks(model: nn.Module, dist_manager) -> list[str]:
    """Replace every ``FoldingTrunk`` submodule with ``TrunkCPWrapper``.

    Walks ``model.named_modules()`` and rebinds each attribute that points
    at a serial ``FoldingTrunk``. Returns the list of replaced submodule
    paths so callers (e.g. spawned workers) can log what got wrapped.
    """
    from projects.huggingface.transformers.models.esmfold2.modeling_esmfold2_common import (
        FoldingTrunk as SerialFoldingTrunk,
    )

    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, SerialFoldingTrunk):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                targets.append((full, parent, child_name, child))

    replaced: list[str] = []
    for full, parent, child_name, child in targets:
        wrapped = TrunkCPWrapper(child, dist_manager).to(
            device=dist_manager.device, dtype=next(child.parameters()).dtype
        )
        setattr(parent, child_name, wrapped)
        replaced.append(full)
    return replaced
