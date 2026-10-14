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

from ..utils import is_torch_available, is_torch_distributed_available, logging, strtobool
from ..utils.quantization_config import QuantizationMethod
from .utils import _get_input_output_embeddings, _is_torch_distributed_initialized


if TYPE_CHECKING:
    import torch.nn as nn

    from .configuration_utils import DistributedConfig
    from .utils import TransformersDeviceMesh

if is_torch_available():
    import torch

if is_torch_distributed_available():
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

logger = logging.get_logger(__name__)


def is_fsdp_enabled() -> bool:
    """Check if FSDP is active via Accelerate (env var based) — covers FSDP1 only."""
    return (
        _is_torch_distributed_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_fsdp_managed_module(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP (1 or 2)."""
    if not is_torch_distributed_available():
        return False

    # FSDP2: attribute set by apply_fsdp2()
    if getattr(module, "_is_fsdp_managed_module", False):
        return True
    # FSDP1: wrapped by FullyShardedDataParallel
    from torch.distributed.fsdp import FullyShardedDataParallel

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


def is_norm_and_head_pair(no_reshard_targets: list[tuple[str, nn.Module]], model: nn.Module) -> bool:
    if len(no_reshard_targets) != 2:
        return False
    input_embed, output_head = _get_input_output_embeddings(model)
    head_modules = {module for module in (input_embed, output_head) if module is not None}

    names, modules = [], []
    for name, module in no_reshard_targets:
        names.append(name)
        modules.append(module)

    has_final_norm = any(name == "norm" or name.endswith(".norm") for name in names)
    has_output_head = any(module in head_modules for module in modules)
    return has_final_norm and has_output_head


def apply_fully_sharded_data_parallelism(
    model: nn.Module,
    mesh_manager: TransformersDeviceMesh,
    fsdp_plan: dict[str, str],
    ep_plan: dict[str, str],
) -> nn.Module:
    """
    Apply FSDP2 (fully_shard) to a model: dispatched experts on `efsdp` and the rest of the modules on `fsdp`.

    Torch availability, distributed initialization and the version requirement
    are asserted upstream by `initialize_distributed_mesh`.
    """
    distributed_config = getattr(model.config, "distributed_config", None)
    fsdp_mesh = mesh_manager.get_mesh("fsdp")
    fsdp_policy_kwargs = _get_fsdp_policy_kwargs(distributed_config)

    if distributed_config is not None and distributed_config.ep_size > 1 and ep_plan:
        expert_mesh = mesh_manager.get_mesh("efsdp")
        for module_name, module in model.named_modules():
            if ep_plan.get(module_name) == "ep_dispatch_experts":
                fully_shard(module, mesh=expert_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)
                # an EP group holds ep_size / tp_size distinct batches
                # the efsdp reduce the sums on efsdp_size copies of that expert.
                # The expert gradient therefore covers ep_size / tp_size * efsdp_size = fsdp_size batches
                # not efsdp_size batches (the relationship is efsdp_size = fsdp_size * tp_size / ep_size)
                module.set_gradient_divide_factor(float(distributed_config.fsdp_size))
                if torch.distributed.get_backend(expert_mesh.get_group()) != "nccl":
                    # Non-NCCL backends need to sum first, then apply the division otherwise it runtime error.
                    module.set_force_sum_reduction_for_comms(True)

    reshard_targets, no_reshard_targets = [], []
    for name, style in fsdp_plan.items():
        if style == "keep_full_weight":
            no_reshard_targets.append((name, model.get_submodule(name)))
        else:
            reshard_targets.append((name, model.get_submodule(name)))

    for name, module in reshard_targets:
        fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)
        logger.debug(f"Applied fully_shard to {name} (reshard=True)")

    # Optimization: when the keep buffer is exactly the (final_norm, lm_head/embed)
    # tail pair, bundle them into one fully_shard so that we dont need to do all-gather during backward pass.
    if is_norm_and_head_pair(no_reshard_targets, model):
        names, modules = zip(*no_reshard_targets)
        fully_shard(list(modules), mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
        logger.debug(f"Grouped tail {names} (reshard=False)")
    else:
        for name, module in no_reshard_targets:
            fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
            logger.debug(f"Applied fully_shard to {name} (reshard=False)")

    # Apply FSDP2 to the root module
    fully_shard(model, mesh=fsdp_mesh, **fsdp_policy_kwargs)

    logger.info(f"FSDP2 applied to model via _fsdp_plan: {len(fsdp_plan)} shard targets")

    # Used by generation code to detect FSDP and enable synced_gpus.
    model._is_fsdp_managed_module = True

    # NOTE(3outeille): No need to tie the word embeddings here, it will be done _finalize_model_loading in modeling_utils.py

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
