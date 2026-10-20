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


# an experts module's fused projection stacks, and what each holds beside it
FUSED_STACKS = ("gate_up_proj", "up_proj", "down_proj")
STACK_COMPANIONS = ("_scale_inv", "_weight_global_scale", "_activation_scale")


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def module_kind(module_name: str, module) -> str | None:
    """The kind of layer a quantized-module registry is keyed by: `"experts"`, `"grouped_linear"` (a
    block-diagonal linear, matched on the attribute a swap needs rather than its class name), `"linear"` or
    `"embedding"`; `None` for anything a registry does not replace."""
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


EXPERTS_PATTERN = r"\.experts($|\.)"


def embedding_pattern(name: str) -> str:
    """The `dtype_plan` key of a `modules_to_convert` entry, which names an embedding table by suffix."""
    return rf"{re.escape(name)}$"


def plan_entry_for(dtype_plan: dict, module_name: str, catch_all: bool = True):
    """The format config `dtype_plan` gives `module_name`, `None` if none covers it. A targeted entry wins over the
    `".*"` catch-all whatever their order, which a round trip through `config.json` loses; two claiming the same
    module is an error. Without `catch_all`, only a targeted entry counts."""
    claimed = [pattern for pattern in dtype_plan if pattern != ".*" and re.search(pattern, module_name)]
    if len(claimed) > 1:
        raise ValueError(f"{module_name!r} is claimed by more than one dtype_plan entry: {claimed}")
    if claimed:
        return dtype_plan[claimed[0]]
    return dtype_plan.get(".*") if catch_all else None


def replace_with_quantized_modules(
    model, dtype_plan: dict, modules_to_not_convert: list[str] | None, module_classes: dict[str, Any]
) -> bool:
    """Swap every module `dtype_plan` gives a config for the class `module_classes` builds that kind of layer as, built
    on `meta` by its `from_module(module, config, model)`. A module the plan gives no config, of a kind with no class,
    or named by `modules_to_not_convert` is kept. Returns whether any module was replaced."""
    import torch

    replaced = False
    for module_name, module in model.named_modules():
        kind = module_kind(module_name, module)
        if kind is None or not should_convert_module(module_name, modules_to_not_convert):
            continue
        # the catch-all covers linears and experts; an embedding is quantized only where an entry names it
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
    """Whether no entry of the skip-list `patterns` covers the module `full_name`: one that matches it from its start
    (a name, its sub-tree, or a regex such as `"model.decoder.layer.*"`) or that it ends with (`"fc1"` covers
    `model.decoder.layers.23.fc1`). A match from the start must end at a module boundary: `"model.layers.1"` does not
    cover `model.layers.10`, nor `"mlp.gate"` cover `mlp.gate_proj`."""
    if patterns is None:
        return True
    # the match ends on a "." it consumed, or right before a "." or the end of the name
    boundary = r"(?:(?<=\.)|(?=\.|$))"
    return not any(re.match(rf"(?:{key}){boundary}", full_name) or full_name.endswith(key) for key in patterns)


def checkpoint_module_namer(model):
    """`model`'s module name -> the checkpoint's name for that module, through the model's renames reversed, as the
    loader applies them. They are written on keys, some anchored on a parameter name (DeepSeek-V4's `^embed.weight`),
    so a module's `weight` key is named back."""
    from ..conversion_mapping import get_model_conversion_mapping
    from ..core_model_loading import WeightRenaming

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


def modules_skipped_in_checkpoint(model, patterns: list[str] | None) -> list[str]:
    """The modules of `model` that the skip-list `patterns`, written in the checkpoint's module names, covers under
    those names but not under the model's own: a model renames its modules on load (a VLM's `language_model.model.*`
    becomes `model.language_model.*`). Each is taken with its sub-tree."""
    if not patterns:
        return []
    to_checkpoint = checkpoint_module_namer(model)
    covered = []
    for module_name, _ in model.named_modules():
        if any(module_name.startswith(f"{parent}.") for parent in covered):
            continue
        checkpoint_name = to_checkpoint(module_name)
        if checkpoint_name != module_name and not should_convert_module(checkpoint_name, patterns):
            covered.append(module_name)
    return covered


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
