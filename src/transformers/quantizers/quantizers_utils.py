# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import re
from contextlib import contextmanager
from typing import Any


# fused projection stacks of an experts module, and the tensors each holds beside it
FUSED_STACKS = ("gate_up_proj", "up_proj", "down_proj")
STACK_COMPANIONS = ("_scale_inv", "_weight_global_scale", "_activation_scale")


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def module_kind(module_name: str, module) -> str | None:
    """Return the layer kind a quantized-module registry is keyed by.

    The kinds are `"experts"`, `"grouped_linear"`, `"linear"` and `"embedding"`. Return `None` for a module no registry
    replaces. A grouped linear is block-diagonal. It is matched on the attribute a swap needs, not on its class name.
    """
    from torch import nn

    if module_name.endswith(".experts"):
        return "experts"
    if isinstance(module, nn.Linear) and hasattr(module, "n_groups"):
        return "grouped_linear"
    if type(module) is nn.Linear:
        return "linear"
    if type(module) is nn.Embedding:
        return "embedding"
    return None


# Matches the `experts` module and its parameters, not `shared_experts`
EXPERTS_PATTERN = r"\.experts($|\.)"


def embedding_pattern(name: str) -> str:
    """Return the `dtype_plan` key of a `modules_to_convert` entry. The entry names an embedding table by suffix."""
    return rf"{re.escape(name)}$"


def plan_entry_for(dtype_plan: dict, module_name: str, catch_all: bool = True):
    """Return the format config `dtype_plan` gives `module_name`, or `None` if no entry covers it.

    A targeted entry wins over the `".*"` catch-all in any order, since `config.json` does not keep the order. Two
    targeted entries on one module raise. Without `catch_all`, only a targeted entry counts.
    """
    claimed = [pattern for pattern in dtype_plan if pattern != ".*" and re.search(pattern, module_name)]
    if len(claimed) > 1:
        raise ValueError(f"{module_name!r} is claimed by more than one dtype_plan entry: {claimed}")
    if claimed:
        return dtype_plan[claimed[0]]
    return dtype_plan.get(".*") if catch_all else None


def replace_with_quantized_modules(
    model, dtype_plan: dict, modules_to_not_convert: list[str] | None, module_classes: dict[str, Any]
) -> bool:
    """Swap every module `dtype_plan` gives a config for its quantized class in `module_classes`.

    The quantized module is built on `meta` with `from_module(module, config, model)`. A module is kept when the plan
    gives it no config, when its kind has no class, or when `modules_to_not_convert` names it. Return whether any
    module was replaced.
    """
    import torch

    replaced = False
    for module_name, module in model.named_modules():
        kind = module_kind(module_name, module)
        if kind is None or not should_convert_module(module_name, modules_to_not_convert):
            continue
        # the catch-all covers linears and experts. An embedding needs an entry that names it.
        config = plan_entry_for(dtype_plan, module_name, catch_all=kind != "embedding")
        quantized_class = module_classes.get(kind) if config is not None else None
        if quantized_class is None:
            continue
        with torch.device("meta"):
            quantized = quantized_class.from_module(module, config, model)
        model.set_submodule(module_name, quantized)
        replaced = True
    return replaced


def should_convert_module(full_name, patterns: list[str] | None = None):
    if patterns is None:
        return True

    # We should avoid converting in the following situations:
    # 1. The pattern matches the start of `full_name` up to a module boundary.
    #    The match ends on a ".", right before a ".", or at the end.
    #    (e.g., "model.decoder.layer.11." and "model.decoder.layer.11" match "model.decoder.layer.11.attn.weight";
    #    "model.decoder.layer.1" does not).
    # 2. The pattern matches `full_name` exactly or via regex, with the same boundary
    #    (e.g., "lm_head" matches "lm_head"; "model.decoder.layer.*" matches "model.decoder.layer.11.attn.weight";
    #    "mlp.gate" does not match "mlp.gate_proj").
    # 3. `full_name` ends with the pattern
    #    (e.g., "fc1" matches "model.decoder.layers.23.fc1").

    boundary = r"(?:(?<=\.)|(?=\.|$))"
    # Group the key so the boundary applies to every alternative.
    return not any(re.match(rf"(?:{key}){boundary}", full_name) or full_name.endswith(key) for key in patterns)


def checkpoint_module_namer(model):
    """Return a function mapping a module name of `model` to its name in the checkpoint.

    It applies the model's renames in reverse, as the loader does. Renames match keys, and some anchor on a parameter
    name such as `^embed.weight`. So the namer renames the module's `weight` key. A module rename moves all of its
    params alike, so `bias` would give the same name.
    """
    from ..conversion_mapping import get_model_conversion_mapping
    from ..core_model_loading import WeightRenaming

    # The legacy renames only rename params, e.g. `LayerNorm.gamma` to `LayerNorm.weight`, not modules
    reverse = [
        rule.reverse_transform()
        for rule in get_model_conversion_mapping(model, add_legacy=False)[::-1]
        if isinstance(rule, WeightRenaming)
    ]

    def checkpoint_name(module_name: str) -> str:
        key = f"{module_name}.weight"
        for rule in reverse:
            key, _ = rule.rename_source_key(key)
        return key.removesuffix(".weight")

    return checkpoint_name


@contextmanager
def on_device(device):
    """Align the current accelerator device with a tensor or device-like object."""
    from ..utils import is_torch_available

    if is_torch_available():
        import torch

        if isinstance(device, torch.Tensor):
            device = device.device
        elif isinstance(device, str):
            device = torch.device(device)

        device_type = getattr(device, "type", None)
        if device_type == "cuda":
            with torch.cuda.device(device):
                yield
                return
        if device_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(device):
                yield
                return

    yield
