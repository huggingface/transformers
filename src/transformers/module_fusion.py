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

import inspect
import re
from dataclasses import dataclass
from typing import Any

from .utils import is_torch_available


if is_torch_available():
    import torch.nn as nn


@dataclass
class ModuleSpec:
    """
    Describes the input and output tensor names for one module in a fusion chain.

    Args:
        inputs (`list[str]`):
            Names of the positional inputs to this module, in order.
        outputs (`list[str]`):
            Names of the outputs produced by this module, in order.
            Use ``"_"`` as a placeholder for outputs that should be ignored / not wired.
    """

    inputs: list[str]
    outputs: list[str]


class RegistryCollector(nn.Module):
    """
    Transparent pass-through module that captures its inputs into a shared registry.

    Placed at the position of every module in the fusion chain except the last one.
    The registry is later consumed by `FusedModule`.
    """

    def __init__(self, spec: ModuleSpec, index: int, registry: dict[str, Any], orig_module: nn.Module):
        super().__init__()
        self.spec = spec
        self.index = index
        self.orig_module = orig_module
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        # This module is a transparent pass-through, so we delegate all attribute access to the original module.
        # In particular, it allows to access the original parameters and buffers as if the collector was not there.
        try:
            return super().__getattr__(name)  # handles _modules, _parameters, _buffers
        except AttributeError:
            return getattr(self._modules["orig_module"], name)

    def _input_key(self, name: str) -> str:
        return f"in_{self.index}_{name}"

    def forward(self, *args, **kwargs):
        for name, arg in zip(self.spec.inputs, args):
            self._registry[self._input_key(name)] = arg
        self._registry.update({self._input_key(name): value for name, value in kwargs.items()})
        return args[0] if len(args) == 1 else args


class FusedModule(nn.Module):
    """
    Executes a chain of modules in a single forward call, using inputs previously
    captured by `RegistryCollector` instances.

    The registry uses two namespaces:
    - ``in_{i}_{name}``  — external inputs captured by collector ``i``
    - ``out_{i}_{name}`` — outputs produced by module ``i`` during fused execution

    For module ``i > 0``, inputs are resolved from ``out_{i-1}_{name}`` first,
    then fall back to ``in_{i}_{name}`` for external inputs not produced by the chain.
    """

    def __init__(self, modules: list[nn.Module], specs: list[ModuleSpec], registry: dict[str, Any]):
        super().__init__()
        self.modules_to_fuse = nn.ModuleList(modules)
        self.specs = specs
        self._registry = registry
        self._signatures = [inspect.signature(mod.forward) for mod in modules]
        self._validate_specs()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # handles _modules, _parameters, _buffers
        except AttributeError:
            return getattr(self._modules["modules_to_fuse"][-1], name)

    def _input_key(self, module_index: int, name: str) -> str:
        return f"in_{module_index}_{name}"

    def _output_key(self, module_index: int, name: str) -> str:
        return f"out_{module_index}_{name}"

    def _validate_specs(self):
        if len(self.modules_to_fuse) != len(self.specs):
            raise ValueError("Number of modules and specs must match.")

        # Build a mapping: out_{i}_{name} → i, for each module output.
        output_producers = {}
        for i, spec in enumerate(self.specs):
            for name in spec.outputs:
                if name != "_":
                    output_producers[self._output_key(i, name)] = i

        for i, (mod, spec, sig) in enumerate(zip(self.modules_to_fuse, self.specs, self._signatures)):
            n_required = sum(1 for p in sig.parameters.values() if p.default is inspect.Parameter.empty)
            if len(spec.inputs) < n_required or len(spec.inputs) > len(sig.parameters):
                raise ValueError(
                    f"Module of type {type(mod)} expects {len(sig.parameters)} inputs "
                    f"({n_required} required) but spec defines {len(spec.inputs)}."
                )
            if i == 0:
                continue  # module 0 inputs come from collectors, always externally provided
            for inp in spec.inputs:
                key = self._output_key(i - 1, inp)
                if key in output_producers and output_producers[key] > i - 1:
                    raise ValueError(
                        f"Module {i} requires '{inp}' but it is produced by module "
                        f"{output_producers[key]}, which comes later in the chain."
                    )

    def forward(self, *args, **kwargs):
        for name, arg in zip(self.specs[0].inputs, args):
            self._registry[self._input_key(0, name)] = arg
        self._registry.update({self._input_key(0, name): value for name, value in kwargs.items()})

        outputs = None
        for index, (mod, spec, sig) in enumerate(zip(self.modules_to_fuse, self.specs, self._signatures)):
            param_names = list(sig.parameters.keys())
            inputs = {}
            for spec_name, arg_name in zip(spec.inputs, param_names):
                if index == 0:
                    key = self._input_key(0, spec_name)
                else:
                    out_key = self._output_key(index - 1, spec_name)
                    key = out_key if out_key in self._registry else self._input_key(index, spec_name)
                inputs[arg_name] = self._registry[key]
            bound = sig.bind(**inputs)
            bound.apply_defaults()
            outputs = mod(**bound.arguments)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            self._registry.update(
                {self._output_key(index, name): output for name, output in zip(spec.outputs, outputs) if name != "_"}
            )

        self._registry.clear()

        if outputs is None:
            return None
        return outputs[0] if len(outputs) == 1 else outputs

    def __repr__(self):
        return f"FusedModule(fused={self.modules_to_fuse})"


def fuse_modules(
    model: nn.Module,
    module_names_to_fuse: list[str],
    module_specs: list[ModuleSpec],
) -> None:
    """
    Fuse a sequence of submodules into a single `FusedModule` in-place.

    The function traverses the model tree and, for every parent module whose
    immediate children match all entries in ``module_names_to_fuse``, replaces:

    - every module except the last with a `RegistryCollector` (transparent pass-through),
    - the last module with a `FusedModule` that re-executes the full chain.

    Args:
        model (`nn.Module`):
            The model to modify in-place.
        module_names_to_fuse (`list[str]`):
            Glob-style paths of the modules to fuse, e.g.
            ``["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"]``.
            Integer indices are replaced with ``*`` during matching so that the
            same spec applies to every repeated block.
        module_specs (`list[ModuleSpec]`):
            One `ModuleSpec` per entry in `module_names_to_fuse`,
            describing input/output tensor names for each module.

    Example:

        specs = [
            ModuleSpec(inputs=["hidden_states"], outputs=["hidden_states"]),
            ModuleSpec(inputs=["hidden_states"], outputs=["hidden_states"]),
        ]
        fuse_modules(
            model,
            ["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"],
            specs,
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
        registry = {}
        modules_to_fuse = [generic_children[p][1] for p in module_names_to_fuse]
        for index, (p, spec) in enumerate(zip(module_names_to_fuse[:-1], module_specs[:-1])):
            module.add_module(generic_children[p][0], RegistryCollector(spec, index, registry, modules_to_fuse[index]))
        last_p = module_names_to_fuse[-1]
        module.add_module(generic_children[last_p][0], FusedModule(modules_to_fuse, module_specs, registry))


def unfuse_modules(model: nn.Module) -> None:
    """
    Revert a previous `fuse_modules` call in-place, restoring the original modules.

    For each `FusedModule` found in the model tree, the function:

    - replaces each sibling `RegistryCollector` with the corresponding original module
      (recovered from `FusedModule.modules_to_fuse`),
    - replaces the `FusedModule` itself with the last original module.

    Collectors belonging to a given `FusedModule` are identified by sharing the same
    ``_registry`` object.

    Args:
        model (`nn.Module`): The model to restore in-place.

    Example::

        fuse_modules(model, ["model.layers.*.norm", "model.layers.*.mlp"], specs)
        # ... optimized forward pass ...
        unfuse_modules(model)  # back to original
    """
    for parent in model.modules():
        fused_children = {name: child for name, child in parent.named_children() if isinstance(child, FusedModule)}
        for fused_name, fused in fused_children.items():
            # Collectors belonging to this FusedModule share the same registry object.
            collectors = {
                name: child
                for name, child in parent.named_children()
                if isinstance(child, RegistryCollector) and child._registry is fused._registry
            }
            for col_name, collector in collectors.items():
                parent.add_module(col_name, fused.modules_to_fuse[collector.index])
            parent.add_module(fused_name, fused.modules_to_fuse[-1])
