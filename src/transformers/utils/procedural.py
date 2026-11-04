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
(`randn(seed) * std`, or ones / zeros), uses them and drops them, so a dummy model has the real module tree, names,
shapes and dtypes and a working forward pass, without storing any weights. The seed of each parameter mixes the
global seed with the parameter's position, so a given (repo, seed) always gives the same model. Tensors with 3 or
more dims (fused MoE experts) are generated per slice of the first dim, so `experts[i]` only generates expert `i`.
"""

import torch
from torch.utils._pytree import tree_map


_MASK = (1 << 64) - 1


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
    def __new__(cls, shape, dtype, seed: int, kind: str = "randn", std: float = 0.02, device="cpu"):
        t = torch.Tensor._make_wrapper_subclass(cls, shape, dtype=dtype, device=device, requires_grad=False)
        t.seed, t.kind, t.std = seed, kind, std
        return t

    def __init__(self, *args, **kwargs):
        pass

    def _like(self, seed=None, shape=None, dtype=None, device=None) -> "ProceduralTensor":
        return ProceduralTensor(
            self.shape if shape is None else shape,
            self.dtype if dtype is None else dtype,
            self.seed if seed is None else seed,
            self.kind,
            self.std,
            self.device if device is None else device,
        )

    def materialize(self) -> torch.Tensor:
        """The values of this tensor, generated now; they are not kept."""
        if self.kind == "ones":
            return torch.ones(self.shape, dtype=self.dtype, device=self.device)
        if self.kind == "zeros":
            return torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.dim() >= 3:
            return torch.stack([self._like(_mix(self.seed, i), self.shape[1:]).materialize() for i in range(self.shape[0])])
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        values = torch.randn(self.shape, generator=generator, device=self.device, dtype=torch.float32)
        return values.mul_(self.std).to(self.dtype)

    def __repr__(self, *, tensor_contents=None):
        return repr(self.materialize())

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten
        if func in (aten.detach.default, aten.alias.default):
            return args[0]._like()
        # `.to(device / dtype)` keeps the recipe: the values are then generated there
        if func is aten._to_copy.default and isinstance(args[0], cls):
            return args[0]._like(dtype=kwargs.get("dtype"), device=kwargs.get("device"))
        # `experts[i]`: only slice i is generated
        if func is aten.select.int and isinstance(args[0], cls) and args[0].dim() >= 3 and args[1] == 0:
            t = args[0]
            return t._like(_mix(t.seed, args[2] % t.shape[0]), t.shape[1:])
        if func._schema.is_mutable:
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
