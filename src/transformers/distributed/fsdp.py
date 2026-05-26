# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any, Literal

from ..utils import is_torch_available, is_torch_greater_or_equal, logging, strtobool
from ..utils.quantization_config import QuantizationMethod
from .tensor_parallel import replace_layer_number_by_wildcard


if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch

if is_torch_available() and is_torch_greater_or_equal("2.6"):
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy

logger = logging.get_logger(__name__)


def is_fsdp_enabled() -> bool:
    """Check if FSDP is active via Accelerate (env var based) — covers FSDP1 only."""
    if not is_torch_available():
        return False

    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_fsdp_managed_module(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP (1 or 2)."""
    if not is_torch_available():
        return False
    if not torch.distributed.is_available():
        return False

    # FSDP2: attribute set by apply_fsdp2()
    if getattr(module, "_is_fsdp_managed_module", False):
        return True
    # FSDP1: wrapped by FullyShardedDataParallel
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(module, FullyShardedDataParallel)


def _get_policy_kwargs(fsdp_plan: dict[str, Any]) -> dict[str, Any]:
    """Parse `cpu_offload` / `mixed_precision` flags from the user fsdp_plan into fully_shard kwargs."""
    policy_kwargs = {}
    if fsdp_plan.get("cpu_offload"):
        policy_kwargs["offload_policy"] = CPUOffloadPolicy()
    if fsdp_plan.get("mixed_precision"):
        policy_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
        )
    return policy_kwargs


def _parse_manual_plan_entry(
    entry: list[str],
) -> tuple[bool, MixedPrecisionPolicy | None, CPUOffloadPolicy | None]:
    """
    Returns:
        tuple[bool, MixedPrecisionPolicy | None, CPUOffloadPolicy | None]:
            - bool: whether to reshard after forward
            - MixedPrecisionPolicy | None: mixed precision policy
            - CPUOffloadPolicy | None: cpu offload policy
    """

    if not isinstance(entry, list):
        raise ValueError(
            f"Manual fsdp_plan values must be a list of strings combining strategy/policies, got {type(entry)}"
        )
    items = entry

    strategy: Literal["free_full_weight", "keep_full_weight"] | None = None
    offload_policy: CPUOffloadPolicy | None = None
    mp_policy: MixedPrecisionPolicy | None = None

    for item in items:
        if not isinstance(item, str):
            raise ValueError(
                f"fsdp_plan option must be a string, got {type(item)}. "
                "Supported: 'free_full_weight', 'keep_full_weight', 'cpu_offload', 'mixed_precision'."
            )
        token = item.lower()
        if token in {"free_full_weight", "keep_full_weight"}:
            strategy = token
        elif token == "cpu_offload":
            offload_policy = CPUOffloadPolicy()
        elif token == "mixed_precision":
            # TODO(3outeille): add support for different dtypes
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(
                "Unknown fsdp_plan option "
                f"{item!r}. Supported: 'free_full_weight', 'keep_full_weight', 'cpu_offload', 'mixed_precision'."
            )

    if strategy is None:
        strategy = "free_full_weight"

    return strategy != "keep_full_weight", mp_policy, offload_policy


def _iter_manual_plan_targets(model, pattern, name_to_module, already_sharded_names):
    if pattern in name_to_module:
        target = name_to_module[pattern]
        if isinstance(target, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)):
            # (ModuleList, ModuleDict, Sequential) don't have a forward() that gets called -
            # the model loops over their children directly. So when a pattern matches a
            # container, we shard each child instead.
            for child_name, child in target.named_children():
                yield f"{pattern}.{child_name}", child
        else:
            yield pattern, target
        return

    # Prefix match: "model.layers" matches "model.layers.0", etc.
    for name, module in model.named_modules():
        if name in already_sharded_names or isinstance(
            module, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)
        ):
            continue
        if name != pattern and not name.startswith(pattern + "."):
            continue
        if any(
            name.startswith(already_sharded_names_name + ".") for already_sharded_names_name in already_sharded_names
        ):
            continue
        yield name, module


def _get_manual_plan_modules(fsdp_plan: dict[str, Any]) -> dict[str, list[str]]:
    modules = fsdp_plan.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Manual fsdp_plan must define a 'modules' dict.")
    return modules


def is_tail_pair(entries) -> bool:
    """Match the canonical tail pair: one final norm + the output head (or tied embedding)."""
    if len(entries) != 2:
        return False
    names = [name for name, _ in entries]
    has_norm = any(n == "norm" or n.endswith(".norm") for n in names)
    has_head = any(n in {"lm_head", "embed_tokens"} or n.endswith((".lm_head", ".embed_tokens")) for n in names)
    return has_norm and has_head


def tied_source_path(model) -> str | None:
    """Return the dotted path of the input embedding module (the tied source)."""
    input_embed = getattr(model, "get_input_embeddings", lambda: None)()
    if input_embed is None:
        return None
    for name, module in model.named_modules():
        if module is input_embed:
            return name
    return None


def _resolve_plan_key(name_to_module: dict, key: str):
    """Resolve a plan key into the matching (name, module) pairs.

    Supports exact module names and tp_plan-style wildcards (via
    ``replace_layer_number_by_wildcard``).
    """
    if key in name_to_module:
        return [(key, name_to_module[key])]
    return [(name, mod) for name, mod in name_to_module.items() if replace_layer_number_by_wildcard(name) == key]


def _iter_plan_targets(model, plan, is_weights_tied: bool, tied_source: str | None):
    """Yield ``(name, module, strategy)`` for every module the plan applies to.

    Expands wildcards via ``_resolve_plan_key`` and pre-applies tying rules:
    skips the standalone tied-source entry, and rewrites a keep ``"lm_head"``
    entry to the tied source so the shared parameter is wrapped once.
    """
    name_to_module = dict(model.named_modules())
    for key, strategy in plan.items():
        if is_weights_tied and key == tied_source:
            continue
        if is_weights_tied and key == "lm_head" and strategy == "keep_full_weight" and tied_source is not None:
            yield tied_source, name_to_module[tied_source], strategy
            continue
        for name, module in _resolve_plan_key(name_to_module, key):
            yield name, module, strategy


def apply_fully_shard_data_parallel(
    model,
    fsdp_mesh,
    fsdp_plan: dict[str, Any] | None,
):
    """
    Apply FSDP2 (fully_shard) to a model.

    When ``fsdp_plan`` is ``None`` or doesn't contain a ``"modules"`` key, the
    model-declared ``model._fsdp_plan`` drives sharding. Policies (`cpu_offload`,
    `mixed_precision`) from ``fsdp_plan`` are applied on top.

    When ``fsdp_plan`` has a ``"modules"`` key, the user fully specifies the
    layout (manual mode).

    Examples:
        # Plan-driven (uses model._fsdp_plan).
        fsdp_plan = None
        fsdp_plan = {"cpu_offload": True, "mixed_precision": True}

        # Manual override.
        fsdp_plan = {
            "modules": {
                "model.embed_tokens": ["free_full_weight"],
                "model.layers.0.mlp": ["free_full_weight", "cpu_offload", "mixed_precision"],
                "model.norm": ["keep_full_weight"],
                "lm_head": ["keep_full_weight"],
            },
        }
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    if not is_torch_greater_or_equal("2.6"):
        raise OSError("FSDP2 requires torch>=2.6")

    if fsdp_plan is None:
        fsdp_plan = {}

    input_embed = getattr(model, "get_input_embeddings", lambda: None)()
    output_embed = getattr(model, "get_output_embeddings", lambda: None)()
    is_weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )

    if not isinstance(fsdp_plan, dict):
        raise ValueError(f"fsdp_plan must be a dict, got {type(fsdp_plan)}")
    is_manual = "modules" in fsdp_plan

    if not is_manual:
        policy_kwargs = _get_policy_kwargs(fsdp_plan)

        plan = getattr(model, "_fsdp_plan", None) or {}
        if not plan:
            raise ValueError(
                f"{type(model).__name__} has no `_fsdp_plan` declared. Either set "
                "`base_model_fsdp_plan` on the config and `_fsdp_plan` on the head class, "
                "or pass an explicit `fsdp_plan={'modules': {...}}` manual override."
            )

        tied_source = tied_source_path(model) if is_weights_tied else None
        keep_buffer: list[tuple[str, Any]] = []

        for name, module, strategy in _iter_plan_targets(model, plan, is_weights_tied, tied_source):
            if strategy == "keep_full_weight":
                keep_buffer.append((name, module))
                continue
            fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True, **policy_kwargs)
            logger.debug(f"Applied fully_shard to {name} (reshard=True)")

        # Optimization: when the keep buffer is exactly the (final_norm, lm_head/embed)
        # tail pair, bundle them into one fully_shard so that we dont need to do all-gather during backward pass.
        if is_tail_pair(keep_buffer):
            keep_names = [n for n, _ in keep_buffer]
            keep_modules = [m for _, m in keep_buffer]
            fully_shard(keep_modules, mesh=fsdp_mesh, reshard_after_forward=False, **policy_kwargs)
            logger.debug(f"Grouped tail {keep_names} (reshard=False)")
        else:
            for name, module in keep_buffer:
                fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=False, **policy_kwargs)
                logger.debug(f"Applied fully_shard to {name} (reshard=False)")

        # Shard root model
        fully_shard(model, mesh=fsdp_mesh, **policy_kwargs)

        logger.info(f"FSDP2 applied to model via _fsdp_plan: {len(plan)} entries")

    else:
        # fsdp_plan = {
        #     "modules": {
        #         "model.layers.0.self_attn": ["free_full_weight"],       # reshard_after_forward=True
        #         "model.norm": ["keep_full_weight"],                      # reshard_after_forward=False
        #         "model.layers.0.mlp": ["free_full_weight", "cpu_offload", "mixed_precision"],
        #     },
        # }

        name_to_module = dict(model.named_modules())
        already_sharded_names: set[str] = set()
        root_mp_policy = MixedPrecisionPolicy()
        root_offload_policy = OffloadPolicy()

        for pattern, entry in _get_manual_plan_modules(fsdp_plan).items():
            reshard, mp_policy, offload_policy = _parse_manual_plan_entry(entry)
            if mp_policy is not None:
                root_mp_policy = mp_policy
            if offload_policy is not None:
                root_offload_policy = offload_policy

            for name, module in _iter_manual_plan_targets(model, pattern, name_to_module, already_sharded_names):
                if name in already_sharded_names:
                    continue
                shard_kwargs = {"mesh": fsdp_mesh, "reshard_after_forward": reshard}
                if mp_policy is not None:
                    shard_kwargs["mp_policy"] = mp_policy
                if offload_policy is not None:
                    shard_kwargs["offload_policy"] = offload_policy
                fully_shard(module, **shard_kwargs)
                already_sharded_names.add(name)
                logger.debug(f"Applied fully_shard to {name}")

        # Shard root model with the same policies as sub-modules.
        # MixedPrecisionPolicy.output_dtype casting happens in post_forward
        # for every fully_shard-wrapped module, even with no direct parameters.
        fully_shard(model, mesh=fsdp_mesh, mp_policy=root_mp_policy, offload_policy=root_offload_policy)

    # Used by generation code to detect FSDP and enable synced_gpus.
    model._is_fsdp_managed_module = True

    if is_weights_tied and hasattr(model, "tie_weights"):
        # Re-tie weights.
        # fully_shard replaces nn.Parameter objects (swapping data for DTensor shards),
        # which breaks weight tying (e.g. lm_head.weight is no longer embed_tokens.weight).
        # Re-tying makes lm_head._parameters["weight"] point to the new DTensor parameter
        # so gradients accumulate correctly into a single buffer.
        model.tie_weights()

    return model


# ========================= PEFT compatibility =========================
# TODO(3outeille): make sure new FSDP works with PEFT
def get_fsdp_ckpt_kwargs():
    """
    Returns checkpoint kwargs for FSDP model saving.

    Checks if the `adapter_only` parameter is supported by `save_fsdp_model` from accelerate
    and returns the appropriate kwargs.
    """
    from accelerate.utils import save_fsdp_model

    if "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def update_fsdp_plugin_peft(model, accelerator):
    """
    Updates the FSDP plugin for PEFT LoRA/QLoRA compatibility.

    When using FSDP with PEFT LoRA, the auto wrap policy needs to be updated to additionally wrap
    LoRA trainable layers separately. When using FSDP with QLoRA, the mixed precision policy needs
    to be updated to use the quantization storage data type.
    """
    from peft import PeftConfig
    from peft.utils.other import fsdp_auto_wrap_policy

    if isinstance(model.active_peft_config, PeftConfig):
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    if (
        getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        and model.hf_quantizer.quantization_config.bnb_4bit_quant_storage.is_floating_point
    ):
        accelerator.state.fsdp_plugin.set_mixed_precision(
            model.hf_quantizer.quantization_config.bnb_4bit_quant_storage, override=True
        )
