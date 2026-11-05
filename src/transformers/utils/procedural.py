# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Procedural weights for `from_pretrained(..., dummy=True)`.

A `ProceduralTensor` is only a shape, a dtype, a device and a seed. Any op that reads its values generates them
(`randn * std`, or ones / zeros), uses them and drops them, so a dummy model has the real module tree, names, shapes
and dtypes and a working forward pass without storing any weights. Each parameter's seed mixes the global seed with
the parameter's position, so a given (repo, seed) always gives the same model.

Memory stays small whatever the model size:
- each block of `SEED_ROWS` rows of a matrix has its own seed, so `w[i]` or `w[a:b]` only generates the blocks it
  touches; matmuls (`x @ W.T`) and embedding lookups go in chunks of about `CHUNK_BYTES`, so a full matrix is never
  in memory at once;
- tensors with 3 or more dims (fused MoE experts) are generated per slice of the first dim, so `experts[i]` only
  generates expert `i`, and dummy models use the per-expert ("eager") MoE path.
"""

import contextlib

import torch
from torch.utils._pytree import tree_map


CHUNK_BYTES = 64 << 20
SEED_ROWS = 16  # each block of 16 rows of a matrix has its own seed, so row slices only generate their blocks
_MASK = (1 << 64) - 1
aten = torch.ops.aten


def _mix(*xs: int) -> int:
    """splitmix64 over the inputs, as a 63-bit seed."""
    h = 0x9E3779B97F4A7C15
    for x in xs:
        h = (h ^ (x & _MASK)) * 0xBF58476D1CE4E5B9 & _MASK
        h = (h ^ (h >> 31)) * 0x94D049BB133111EB & _MASK
        h ^= h >> 29
    return h >> 1


class ProceduralTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype, seed: int, kind: str = "randn", std: float = 0.02, device="cpu", transposed=False):
        shape = tuple(shape)
        shown = (shape[1], shape[0]) if transposed else shape
        t = torch.Tensor._make_wrapper_subclass(cls, shown, dtype=dtype, device=device, requires_grad=False)
        t.seed, t.kind, t.std, t.base_shape, t.transposed = seed, kind, std, shape, transposed
        return t

    def __init__(self, *args, **kwargs):
        pass

    def _like(self, seed=None, shape=None, dtype=None, device=None, transposed=None) -> "ProceduralTensor":
        return ProceduralTensor(
            self.base_shape if shape is None else shape,
            self.dtype if dtype is None else dtype,
            self.seed if seed is None else seed,
            self.kind,
            self.std,
            self.device if device is None else device,
            self.transposed if transposed is None else transposed,
        )

    # -- generation --------------------------------------------------------------------------------------------
    def rows(self, r0: int, r1: int) -> torch.Tensor:
        """Rows r0:r1 of the (untransposed) 2-D tensor, generated from the seed blocks they touch only."""
        cols = self.base_shape[1]
        if self.kind != "randn":
            return (torch.ones if self.kind == "ones" else torch.zeros)(
                r1 - r0, cols, dtype=self.dtype, device=self.device
            )
        b0, b1 = r0 // SEED_ROWS, (r1 + SEED_ROWS - 1) // SEED_ROWS
        out = torch.empty((b1 - b0) * SEED_ROWS, cols, dtype=torch.float32, device=self.device)
        g = torch.Generator(device=self.device)
        for b in range(b0, b1):
            g.manual_seed(_mix(self.seed, b))
            torch.randn(
                SEED_ROWS,
                cols,
                generator=g,
                device=self.device,
                out=out[(b - b0) * SEED_ROWS : (b - b0 + 1) * SEED_ROWS],
            )
        start = r0 - b0 * SEED_ROWS
        return out[start : start + r1 - r0].mul_(self.std).to(self.dtype)

    def chunks(self):
        """(first row, values) for chunks of about CHUNK_BYTES of rows of the (untransposed) 2-D tensor."""
        rows, cols = self.base_shape
        step = max(SEED_ROWS, CHUNK_BYTES // max(1, cols * 4) // SEED_ROWS * SEED_ROWS)
        for r0 in range(0, rows, step):
            yield r0, self.rows(r0, min(rows, r0 + step))

    def materialize(self) -> torch.Tensor:
        """The full values of this tensor, generated now and not kept (avoided by matmuls and embeddings)."""
        if self.kind != "randn" or len(self.base_shape) == 1:
            if self.kind == "randn":  # 1-D randn
                g = torch.Generator(device=self.device).manual_seed(self.seed)
                return torch.randn(self.base_shape, generator=g, device=self.device).mul_(self.std).to(self.dtype)
            values = (torch.ones if self.kind == "ones" else torch.zeros)(
                self.base_shape, dtype=self.dtype, device=self.device
            )
        elif len(self.base_shape) >= 3:
            values = torch.stack([self._slice(i).materialize() for i in range(self.base_shape[0])])
        else:
            values = self.rows(0, self.base_shape[0])
        return values.t() if self.transposed else values

    def _slice(self, i: int) -> "ProceduralTensor":
        return self._like(_mix(self.seed, i % self.base_shape[0]), self.base_shape[1:], transposed=False)

    def __repr__(self, *, tensor_contents=None):
        return repr(self.materialize())

    # -- dispatch ----------------------------------------------------------------------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        a0 = args[0] if args else None
        if func in (aten.detach.default, aten.alias.default):
            return a0._like()
        # `.to(device / dtype)` keeps the recipe: the values are then generated there
        if func is aten._to_copy.default and isinstance(a0, cls):
            return a0._like(dtype=kwargs.get("dtype"), device=kwargs.get("device"))
        if (
            func in (aten.t.default, aten.transpose.int, aten.permute.default)
            and isinstance(a0, cls)
            and a0.dim() == 2
        ):
            dims = args[1:] if func is not aten.permute.default else tuple(args[1])
            if func is aten.t.default or sorted(d % 2 for d in dims) == [0, 1] and dims != (0, 1):
                return a0._like(transposed=not a0.transposed)
        # `experts[i]`: only slice i is generated
        if isinstance(a0, cls) and len(a0.base_shape) >= 3:
            if func is aten.select.int and args[1] == 0:
                return a0._slice(args[2])
            if func is aten.index.Tensor and len(args[1]) == 1 and args[1][0] is not None and args[1][0].dim() == 0:
                return a0._slice(int(args[1][0]))
        # w[a:b], w[i] (and their transposed forms): only the rows needed are generated
        if isinstance(a0, cls) and len(a0.base_shape) == 2 and a0.kind == "randn":
            row_dim = 1 if a0.transposed else 0
            if func is aten.slice.Tensor:
                dim = args[1] if len(args) > 1 else 0
                start, end, step = (list(args[2:]) + [None, None, 1])[:3]
                n = a0.base_shape[0]
                start, end, _ = slice(start, end, step).indices(n) if dim % 2 == row_dim else (0, n, 1)
                values = a0.rows(start, max(start, end))
                values = values.t() if a0.transposed else values
                if dim % 2 == row_dim:
                    return values[::step] if not a0.transposed else values[:, ::step]
                return func(values, *args[1:], **kwargs)
            if func is aten.select.int and args[1] % 2 == row_dim:
                i = args[2] % a0.base_shape[0]
                return a0.rows(i, i + 1)[0]
        # x @ W.T, chunk by chunk over the rows of W (the output columns)
        if func in (aten.mm.default, aten.addmm.default):
            x, w = (args[1], args[2]) if func is aten.addmm.default else (args[0], args[1])
            if isinstance(w, cls) and w.transposed and w.kind == "randn" and not isinstance(x, cls):
                out = torch.cat([x @ v.t() for _, v in w.chunks()], dim=1)
                if func is aten.addmm.default:
                    beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
                    out = beta * (args[0].materialize() if isinstance(args[0], cls) else args[0]) + alpha * out
                return out
        if func is aten.embedding.default and isinstance(a0, cls) and a0.kind == "randn" and not a0.transposed:
            idx = args[1]
            out = torch.empty(*idx.shape, a0.base_shape[1], dtype=a0.dtype, device=idx.device)
            for r0, v in a0.chunks():
                hit = (idx >= r0) & (idx < r0 + v.shape[0])
                if hit.any():
                    out[hit] = v[idx[hit] - r0].to(out.device)
            return out
        if func._schema.is_mutable and any(isinstance(x, cls) for x in args):
            raise RuntimeError(f"Weights of a dummy model are generated on the fly and can't be modified ({func}).")

        def unwrap(x):
            return x.materialize() if isinstance(x, cls) else x

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


def make_procedural(model: torch.nn.Module, seed: int = 0, device_map: dict | None = None) -> None:
    """Replaces every parameter of `model` (usually on the meta device) by a `ProceduralTensor`, in place. Matrices
    get `randn * config.initializer_range`, 1-D weights (norms) ones and biases zeros; tied parameters stay tied."""
    from ..integrations.accelerate import get_device

    std = getattr(getattr(model, "config", None), "initializer_range", None) or 0.02
    new = {}
    for index, (name, param) in enumerate(model.named_parameters()):
        leaf = name.rsplit(".", 1)[-1]
        kind = "randn" if param.dim() >= 2 else ("zeros" if leaf == "bias" else "ones")
        device = get_device(device_map, name, valid_torch_device=True) if device_map else "cpu"
        value = ProceduralTensor(param.shape, param.dtype, _mix(seed, index), kind, std, device)
        new[id(param)] = torch.nn.Parameter(value, requires_grad=False)
        new[id(param)]._is_hf_initialized = True
    for module in model.modules():
        for leaf, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[leaf] = new[id(param)]
    # Per-expert MoE path: only the experts that tokens are routed to get generated
    if hasattr(model, "set_experts_implementation"):
        with contextlib.suppress(Exception):  # models without experts
            model.set_experts_implementation("eager")
