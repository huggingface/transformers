# Copyright 2026 The HuggingFace Inc. team.
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

import functools
import re

from .utils import is_torch_available


if is_torch_available():
    import torch.nn as nn


class FusedModuleBase(nn.Module):
    def __init__(self, modules_to_fuse: list[nn.Module], source_names: list[str]):
        super().__init__()
        if len(modules_to_fuse) == 0:
            raise ValueError("At least one module must be provided for fusion.")
        if len(modules_to_fuse) != len(source_names):
            raise ValueError("Length of modules_to_fuse and source_names must match.")

        self._source_names = source_names

        for module in modules_to_fuse:
            attr_name = getattr(module, "kernel_layer_name", None)
            if attr_name is None:
                raise ValueError(f"Module {module} does not have a 'kernel_layer_name' attribute.")
            self.add_module(attr_name, module)

        self._fused_module_names = [m.kernel_layer_name for m in modules_to_fuse]

        # `kernelize` validates the kernel's forward signature against the class being replaced.
        # Since the fused container sits at the position of the first module in the chain, the
        # kernel's forward must match that module's signature. We patch the class-level forward
        # here (via `functools.wraps`) so the signature is correct when `kernelize` inspects it.
        # The body raises because this forward is always replaced by the kernel before any call.
        @functools.wraps(type(modules_to_fuse[0]).forward)
        def forward(self, *args, **kwargs):
            raise NotImplementedError("FusedModule is a placeholder and should not be called directly.")

        self.__class__.forward = forward

    def __repr__(self):
        names = ", ".join(self._fused_module_names)
        return f"{self.__class__.__name__}(fused=({names}))"


@functools.cache
def make_fused_module_class(source_layer_names: tuple[str, ...], kernel_layer_name: str) -> type:
    """
    Dynamically create and cache a `FusedModuleBase` subclass for a given fusion combination.

    Args:
        source_layer_names (`tuple[str, ...]`):
            Ordered tuple of `kernel_layer_name` values of the modules being fused
            (e.g. ``("RMSNorm", "MLP")``). Used as the cache key — the same combination
            always returns the same class object.
        kernel_layer_name (`str`):
            The name assigned to the fused class, used by `kernelize` to look up the
            kernel in the mapping (e.g. ``"RMSNormMLP"``).

    Returns:
        A subclass of `FusedModuleBase` with `kernel_layer_name` set as a class attribute.
    """
    return type(
        f"Fused_{'_'.join(source_layer_names)}",
        (FusedModuleBase,),
        {"kernel_layer_name": kernel_layer_name},
    )


def fuse_modules(
    model: nn.Module,
    module_names_to_fuse: list[str],
    kernel_layer_name: str,
) -> None:
    """
    Fuse a sequence of submodules into a single `FusedModuleBase` subclass in-place.

    For every parent module whose immediate children match all entries in
    ``module_names_to_fuse``, the function:

    - replaces the first module with a `FusedModuleBase` subclass instance that holds
      all source modules as named children,
    - replaces the remaining modules with `nn.Identity()` pass-throughs.

    The fused container's ``forward`` signature is patched to match the first source
    module's ``forward``, satisfying the ``kernelize`` signature check.

    Args:
        model (`nn.Module`):
            The model to modify in-place.
        module_names_to_fuse (`list[str]`):
            Glob-style paths of the modules to fuse, e.g.
            ``["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"]``.
            Integer indices are replaced with ``*`` so the same pattern applies to
            every repeated block.
        kernel_layer_name (`str`):
            The ``kernel_layer_name`` assigned to the fused class, used by ``kernelize``
            to look up the kernel in the mapping (e.g. ``"RMSNormMLP"``).

    Example::

        fuse_modules(
            model,
            ["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"],
            "RMSNormMLP",
        )
    """
    pattern = re.compile(r"\d+")
    for module_name, module in model.named_modules():
        generic_children = {
            re.sub(pattern, "*", f"{module_name}.{n}" if module_name else n): (n, child)
            for n, child in module.named_children()
        }
        if not all(p in generic_children for p in module_names_to_fuse):
            continue

        child_names = [generic_children[p][0] for p in module_names_to_fuse]
        modules_to_fuse = [generic_children[p][1] for p in module_names_to_fuse]

        source_layer_names = tuple(getattr(m, "kernel_layer_name") for m in modules_to_fuse)
        FusedClass = make_fused_module_class(source_layer_names, kernel_layer_name)
        fused_instance = FusedClass(modules_to_fuse, child_names)

        module.add_module(child_names[0], fused_instance)
        for child_name in child_names[1:]:
            module.add_module(child_name, nn.Identity())


def unfuse_modules(model: nn.Module) -> None:
    """
    Revert a previous `fuse_modules` call in-place, restoring the original modules.

    For each `FusedModuleBase` instance found in the model tree, the function:

    - restores the original first module at the fused container's position,
    - restores the remaining original modules at their original positions
      (replacing the `nn.Identity()` pass-throughs).

    Args:
        model (`nn.Module`): The model to restore in-place.

    Example::

        fuse_modules(model, ["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"], "RMSNormMLP")
        # ... kernelized forward pass ...
        unfuse_modules(model)  # back to original
    """
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, FusedModuleBase):
                continue
            orig_modules = [getattr(child, layer_name) for layer_name in child._fused_module_names]
            parent.add_module(name, orig_modules[0])
            for sibling_name, orig_module in zip(child._source_names[1:], orig_modules[1:]):
                parent.add_module(sibling_name, orig_module)
