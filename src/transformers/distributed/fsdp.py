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
from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

from ..integrations.tensor_parallel import replace_layer_number_by_wildcard
from ..utils import is_torch_available, is_torch_greater_or_equal, logging, strtobool
from ..utils.quantization_config import QuantizationMethod


if TYPE_CHECKING:
    import torch.nn as nn

    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch

if is_torch_available() and is_torch_greater_or_equal("2.6"):
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

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


def _get_fsdp_policy_kwargs(distributed_config: DistributedConfig | None) -> dict[str, Any]:
    """Build ``fully_shard`` policy kwargs from ``DistributedConfig`` runtime flags."""
    if distributed_config is None:
        return {}

    fsdp_policy_kwargs = {}
    if distributed_config.fsdp_cpu_offload:
        fsdp_policy_kwargs["offload_policy"] = CPUOffloadPolicy()
    if distributed_config.fsdp_mixed_precision:
        fsdp_policy_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
        )
    return fsdp_policy_kwargs


def is_norm_and_head_pair(modules: list[tuple[str, Any]]) -> bool:
    """Match the canonical tail pair: one final norm + the output head (or tied embedding)."""
    if len(modules) != 2:
        return False
    module_names = [name for name, _ in modules]
    has_final_norm = any(name == "norm" or name.endswith(".norm") for name in module_names)
    has_output_head = any(
        name in {"lm_head", "embed_tokens"} or name.endswith((".lm_head", ".embed_tokens")) for name in module_names
    )
    return has_final_norm and has_output_head


def _resolve_tied_embed_lm_head_plan(
    fsdp_plan: dict[str, str],
    *,
    tie_word_embeddings: bool,
) -> dict[str, str]:
    """
    Rewrite the plan so tied embed/lm_head weights are wrapped once.
    Example:
        {"model.embed_tokens": "free_full_weight",
        "model.layers.*": "free_full_weight",
        "model.norm": "keep_full_weight",
        "lm_head": "keep_full_weight"}
    ->
        {"model.layers.*": "free_full_weight",
        "model.norm": "keep_full_weight",
        "model.embed_tokens": "keep_full_weight"}
    """
    if not tie_word_embeddings:
        return fsdp_plan

    embed_module_key = next(
        (key for key in fsdp_plan if key == "embed_tokens" or key.endswith(".embed_tokens")),
        None,
    )
    if embed_module_key is None:
        return fsdp_plan

    adapted_plan = {}
    for key, sharding_strategy in fsdp_plan.items():
        if key == embed_module_key:
            continue
        if key == "lm_head" and sharding_strategy == "keep_full_weight":
            adapted_plan[embed_module_key] = sharding_strategy
        else:
            adapted_plan[key] = sharding_strategy
    return adapted_plan


def expand_fsdp_plan(model, fsdp_plan: dict[str, str]) -> list[tuple[str, nn.Module, str]]:
    """Expand plan keys into ``(module_name, module, sharding_strategy)`` shard targets."""
    module_lookup = dict(model.named_modules())
    shard_targets: list[tuple[str, nn.Module, str]] = []

    for plan_key, sharding_strategy in fsdp_plan.items():
        if plan_key in module_lookup:
            shard_targets.append((plan_key, module_lookup[plan_key], sharding_strategy))
            continue

        for module_name, module in module_lookup.items():
            if replace_layer_number_by_wildcard(module_name) == plan_key:
                shard_targets.append((module_name, module, sharding_strategy))

    return shard_targets


def verify_fsdp_plan(module_names: list[str], fsdp_plan: dict[str, str] | None) -> None:
    """
    Verify the FSDP plan of the model, log a warning if plan keys were not applied or strategies are invalid.
    """
    if not fsdp_plan:
        return

    name_lookup = dict.fromkeys(module_names)
    unused_rules: dict[str, str] = {}
    invalid_strategies: dict[str, str] = {}

    for key, strategy in fsdp_plan.items():
        if strategy not in {"free_full_weight", "keep_full_weight"}:
            invalid_strategies[key] = strategy
            continue
        if key not in name_lookup and not any(replace_layer_number_by_wildcard(name) == key for name in name_lookup):
            unused_rules[key] = strategy

    if invalid_strategies:
        logger.warning(f"The following FSDP entries have unknown strategies: {invalid_strategies}")
    if unused_rules:
        logger.warning(f"The following FSDP rules were not applied to any module: {unused_rules}")


def apply_fully_shard_data_parallel(model, fsdp_mesh):
    """
    Apply FSDP2 (fully_shard) to a model.
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    if not is_torch_greater_or_equal("2.6"):
        raise OSError("FSDP2 requires torch>=2.6")

    fsdp_plan = dict(getattr(model, "_fsdp_plan", None) or {})
    if not fsdp_plan:
        raise ValueError(
            f"{type(model).__name__} does not have a FSDP2 plan declared. Set "
            "`base_model_fsdp_plan` on the config and `_fsdp_plan` on the head class."
        )

    distributed_config = getattr(model.config, "distributed_config", None)
    fsdp_policy_kwargs = _get_fsdp_policy_kwargs(distributed_config)
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    adapted_fsdp_plan = _resolve_tied_embed_lm_head_plan(fsdp_plan, tie_word_embeddings=tie_word_embeddings)
    shard_targets = expand_fsdp_plan(model, adapted_fsdp_plan)

    reshard_targets = []
    no_reshard_targets = []
    for module_name, module, sharding_strategy in shard_targets:
        if sharding_strategy == "keep_full_weight":
            no_reshard_targets.append((module_name, module))
        else:
            reshard_targets.append((module_name, module))

    for module_name, module in reshard_targets:
        fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)
        logger.debug(f"Applied fully_shard to {module_name} (reshard=True)")

    # Optimization: when the keep buffer is exactly the (final_norm, lm_head/embed)
    # tail pair, bundle them into one fully_shard so that we dont need to do all-gather during backward pass.
    if is_norm_and_head_pair(no_reshard_targets):
        module_names = [name for name, _ in no_reshard_targets]
        modules = [module for _, module in no_reshard_targets]
        fully_shard(modules, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
        logger.debug(f"Grouped tail {module_names} (reshard=False)")
    else:
        for module_name, module in no_reshard_targets:
            fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
            logger.debug(f"Applied fully_shard to {module_name} (reshard=False)")

    # Apply FSDP2 to the root module
    fully_shard(model, mesh=fsdp_mesh, **fsdp_policy_kwargs)

    logger.info(f"FSDP2 applied to model via _fsdp_plan: {len(fsdp_plan)} entries")

    # Used by generation code to detect FSDP and enable synced_gpus.
    model._is_fsdp_managed_module = True

    if tie_word_embeddings and hasattr(model, "tie_weights"):
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
