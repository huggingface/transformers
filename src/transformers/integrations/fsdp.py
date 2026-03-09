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
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal, logging, strtobool
from ..utils.quantization_config import QuantizationMethod


if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.get_logger(__name__)

def is_fsdp_managed_module(module: nn.Module) -> bool:
    if not is_torch_available():
        return False
    import torch
    if not torch.distributed.is_available():
        return False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(module, FullyShardedDataParallel) or getattr(module, "_is_fsdp_managed_module", False)


def is_fsdp_enabled():
    if not is_torch_available():
        return False
    import torch
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def initialize_fsdp(
    fsdp_plan: str | list | tuple | dict[str, str | list[object] | tuple[object, ...]] | None,
    device_mesh=None,
    device_map=None,
):
    """
    Sets up the device mesh for FSDP2 (Fully Sharded Data Parallel).
    This function is called when the model is loaded and fsdp_plan is set.

    Args:
        fsdp_plan: Either "auto" for automatic sharding or a dict mapping module names to sharding strategies.
        device_mesh: Optional pre-created DeviceMesh for FSDP.
        device_map: Optional device map.

    Returns:
        Tuple of (device_map, device_mesh, fsdp_size)
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    import torch
    import torch.distributed as dist

    if fsdp_plan is None:
        return device_map, device_mesh, None

    if not is_torch_greater_or_equal("2.5"):
        raise OSError("FSDP2 is only supported for `torch>=2.5`.")

    if device_mesh is None:
        # Detect the accelerator on the machine
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            device_type = "cpu"  # fallback
        current_device = getattr(torch, device_type)

        if not dist.is_initialized():
            try:
                rank = int(os.environ["RANK"])
                local_rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ["WORLD_SIZE"])

                backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
                backend = backend_map.get(device_type)
                if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", "0")):
                    backend = "ccl"
                if device_type == "xpu" and not is_torch_greater_or_equal("2.8", accept_dev=True):
                    backend = "ccl"

                dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                if device_type != "cpu":
                    current_device.set_device(local_rank)

            except Exception as e:
                raise OSError(
                    "We tried to initialize torch.distributed for you, but it failed. Make "
                    "sure you init torch distributed in your script to use `fsdp_plan`."
                ) from e

        if device_type != "cpu":
            current_device.set_device(int(os.environ["LOCAL_RANK"]))
            index = current_device.current_device()
            fsdp_device = torch.device(device_type, index)
            device_map = fsdp_device
        else:
            fsdp_device = torch.device(device_type)
            device_map = device_type or {}

        fsdp_size = dist.get_world_size()
        device_mesh = torch.distributed.init_device_mesh(fsdp_device.type, (fsdp_size,), mesh_dim_names=("dp_shard",))
    else:
        # Use provided device mesh
        if device_mesh.ndim > 1:
            if "dp_shard" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `fsdp_plan` with n-d `device_mesh`, it must contain an 'fsdp' dimension. "
                    "Please provide a valid `device_mesh`."
                )
            device_mesh = device_mesh["dp_shard"]
        fsdp_size = device_mesh.size()
        device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

    return device_map, device_mesh, fsdp_size


def get_transformer_block_classes(model):
    """
    Identifies transformer block classes in a model for FSDP wrapping.
    These are typically the repeated layers that benefit from FSDP sharding.

    Returns a set of module classes that should be wrapped with fully_shard().
    """
    block_classes = set()

    # Common transformer block class names
    block_names = {
        "DecoderLayer",
        "EncoderLayer",
        "TransformerBlock",
        "Block",
        "Layer",
    }

    for module in model.modules():
        class_name = module.__class__.__name__
        # Use endswith to avoid false positives (e.g. "Layer" matching "LayerNorm")
        for block_name in block_names:
            if class_name.endswith(block_name):
                block_classes.add(type(module))
                break

    # Filter out nested block classes (e.g. SparseMoeBlock inside DecoderLayer).
    # We only want to FSDP-wrap the outermost block classes. If a class like
    # MoeBlock only ever appears inside a DecoderLayer, we skip it.
    if len(block_classes) > 1:
        # Collect the dotted module paths for each candidate class.
        # i.e: {DecoderLayer: ["layers.0", "layers.1"],
        #        MoeBlock: ["layers.0.moe", "layers.1.moe"]}
        paths_by_class = {}
        for name, module in model.named_modules():
            cls = type(module)
            if cls in block_classes:
                paths_by_class.setdefault(cls, []).append(name)

        def _is_nested_inside_other_class(cls):
            # A class is "inner" if every one of its instances lives under
            # an instance of a different candidate class in the module tree.
            paths = paths_by_class.get(cls, [])
            if not paths:
                return False
            for path in paths:
                has_parent = any(
                    path.startswith(parent_path + ".")
                    for other_cls, parent_paths in paths_by_class.items()
                    if other_cls is not cls
                    for parent_path in parent_paths
                )
                if not has_parent:
                    return False
            return True

        # Keep only the outer (non-nested) classes.
        block_classes = {cls for cls in block_classes if not _is_nested_inside_other_class(cls)}

    return block_classes


def _find_final_norm(model, decoder_layer_names):
    # Find the final normalization layer before the output head.
    final_norm = None
    for name, module in model.named_modules():
        if "Norm" not in type(module).__name__:
            continue
        if any(name.startswith(layer_name + ".") for layer_name in decoder_layer_names):
            continue
        final_norm = module
    return final_norm


def apply_fsdp2(
    model,
    device_mesh,
    fsdp_plan: str | dict[str, str] | None,
):
    """
    Apply FSDP2 (fully_shard) to a model following TorchTitan's approach.
    fsdp_plan: Either ``"auto"`` for automatic block detection, a list/tuple
        starting with ``"auto"`` followed by optional policies, or a dict
        mapping module name patterns to sharding strategies.

        **Auto mode** (string or list):
          - ``"auto"`` — automatic block detection with default settings.
          - ``["auto", "cpu_offload"]`` — auto with CPU offloading.
          - ``["auto", "mixed_precision"]`` — auto with mixed precision
            (bf16 params/output, fp32 reduce).
          - ``["auto", "cpu_offload", "mixed_precision"]`` — auto with both.

        **Manual mode** (dict):
        A plan value can be:
          - ``"free_full_weight"`` (reshard after forward)
          - ``"keep_full_weight"`` (do not reshard after forward)
          - a list/tuple combining one of the above with optional policies:
            ``"cpu_offload"``, ``"mixed_precision"``

        fsdp_plan = {
            "model.embed_tokens": "free_full_weight",        # free full weight after forward
            "model.layers.0.self_attn": "free_full_weight",  # free full weight after forward
            "model.layers.0.mlp": "free_full_weight",        # free full weight after forward
            "model.layers.1.self_attn": "free_full_weight",  # free full weight after forward
            "model.layers.1.mlp": "free_full_weight",        # free full weight after forward
            "model.norm": "keep_full_weight",                # keep full weight until backward
            "lm_head": "keep_full_weight",                   # keep full weight until backward
            # Optional per-module policies:
            # "model.layers.0.self_attn": ["free_full_weight", "cpu_offload", "mixed_precision"],
        }
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    import torch

    if not is_torch_greater_or_equal("2.5"):
        raise OSError("FSDP2 requires torch>=2.5")

    # Import FSDP2 API
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

    if device_mesh is None:
        raise ValueError("device_mesh is required for FSDP2")

    # FSDP2 requires contiguous parameters - make them contiguous in-place
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            with torch.no_grad():
                param.data = param.data.contiguous()
            logger.debug(f"Made parameter {name} contiguous for FSDP2")

    input_embed = getattr(model, "get_input_embeddings", lambda: None)()
    output_embed = getattr(model, "get_output_embeddings", lambda: None)()
    weights_tied = (
        input_embed is not None
        and output_embed is not None
        and hasattr(input_embed, "weight")
        and hasattr(output_embed, "weight")
        and input_embed.weight is output_embed.weight
    )
    # Parse list/tuple auto mode: ["auto", "cpu_offload", "mixed_precision", ...]
    auto_mp_policy = None
    auto_offload_policy = None
    if isinstance(fsdp_plan, (list, tuple)) and fsdp_plan and fsdp_plan[0] == "auto":
        for item in fsdp_plan[1:]:
            if not isinstance(item, str):
                raise ValueError(
                    f"Auto-mode policy options must be strings, got {type(item)}. "
                    "Supported: 'cpu_offload', 'mixed_precision'."
                )
            token = item.lower()
            if token == "cpu_offload":
                auto_offload_policy = CPUOffloadPolicy()
            elif token == "mixed_precision":
                #TODO(3outeille): is the output_dtype what we want ?
                auto_mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(
                    f"Unknown auto-mode policy {item!r}. Supported: 'cpu_offload', 'mixed_precision'."
                )
        fsdp_plan = "auto"

    if fsdp_plan == "auto":
        # Build common kwargs for all fully_shard calls in auto mode
        auto_policy_kwargs = {}
        if auto_mp_policy is not None:
            auto_policy_kwargs["mp_policy"] = auto_mp_policy
        if auto_offload_policy is not None:
            auto_policy_kwargs["offload_policy"] = auto_offload_policy

        block_classes = get_transformer_block_classes(model)

        # Collect decoder layer names (needed for norm detection)
        decoder_layer_names = set()
        if block_classes:
            for name, module in model.named_modules():
                if type(module) in block_classes:
                    decoder_layer_names.add(name)

        if not block_classes:
            logger.warning(
                "Could not auto-detect transformer block classes for FSDP. "
                "Applying FSDP only to root module."
            )
        else:
            final_norm = _find_final_norm(model, decoder_layer_names)

            # Step 1: Shard input embeddings (only when not tied).
            # When tied, the shared weight is grouped with the final norm in step 3.
            if input_embed is not None and not weights_tied:
                fully_shard(input_embed, mesh=device_mesh, reshard_after_forward=True, **auto_policy_kwargs)
                logger.debug(f"Applied fully_shard to input embeddings ({type(input_embed).__name__})")

            # Step 2: Shard each decoder layer block
            for name, module in model.named_modules():
                if type(module) in block_classes:
                    fully_shard(module, mesh=device_mesh, reshard_after_forward=True, **auto_policy_kwargs)
                    logger.debug(f"Applied fully_shard to {name} ({type(module).__name__})")

            # Step 3: Group final norm + output head
            # NOTE(3outeille): Small optimization by forcing reshard_after_forward=False for the final norm and output head
            # Otherwise, that would mean reshard/freeing full params after the last forward and immediately re-all-gathering them in the backward pass.
            # which is wasteful. Better to keep them gathered for reuse
            # Untied: [final_norm, lm_head]
            # Tied:   [final_norm, embed_tokens] — embed_tokens.weight IS lm_head.weight,
            tail_modules = []
            if final_norm is not None:
                tail_modules.append(final_norm)
            if weights_tied:
                if input_embed is not None:
                    tail_modules.append(input_embed)
            else:
                if output_embed is not None:
                    tail_modules.append(output_embed)

            if len(tail_modules) > 1:
                fully_shard(tail_modules, mesh=device_mesh, reshard_after_forward=False, **auto_policy_kwargs)
                logger.debug(f"Applied fully_shard to {[type(m).__name__ for m in tail_modules]} grouped (reshard=False)")
            elif len(tail_modules) == 1:
                fully_shard(tail_modules[0], mesh=device_mesh, reshard_after_forward=False, **auto_policy_kwargs)
                logger.debug(f"Applied fully_shard to {type(tail_modules[0]).__name__} (reshard=False)")

        # Step 4: Shard root model
        fully_shard(model, mesh=device_mesh, reshard_after_forward=True, **auto_policy_kwargs)

        # Step 5: Re-tie weights.
        # fully_shard replaces nn.Parameter objects (swapping data for DTensor shards),
        # which breaks weight tying (e.g. lm_head.weight is no longer embed_tokens.weight).
        # Re-tying makes lm_head._parameters["weight"] point to the new DTensor parameter
        # so gradients accumulate correctly into a single buffer.
        if weights_tied and hasattr(model, "tie_weights"):
            model.tie_weights()

        logger.info(
            f"FSDP2 applied to model: {len(block_classes)} block type(s), "
            f"{len(decoder_layer_names)} decoder layers"
        )

    elif isinstance(fsdp_plan, dict):
        # Apply fully_shard based on explicit plan.
        #
        # Each key is a module name (or prefix), and the value controls whether
        # the all-gathered full weight is freed after forward:
        # (ModuleList, ModuleDict, Sequential) don't have a forward()
        # that gets called — the model loops over their children directly.
        # So when a pattern matches a container, we shard each child instead.

        name_to_module = dict(model.named_modules())
        sharded: set[str] = set()

        def _parse_plan_entry(entry):
            if isinstance(entry, str):
                items = [entry]
            elif isinstance(entry, (list, tuple)):
                items = list(entry)
            else:
                raise ValueError(
                    "fsdp_plan values must be a strategy string or a list/tuple combining strategy/policies, "
                    f"got {type(entry)}"
                )

            strategy = None
            offload_policy = None
            mp_policy = None

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

        def _shard(name: str, module, reshard: bool, mp_policy=None, offload_policy=None):
            if name not in sharded:
                shard_kwargs = {"mesh": device_mesh, "reshard_after_forward": reshard}
                if mp_policy is not None:
                    shard_kwargs["mp_policy"] = mp_policy
                if offload_policy is not None:
                    shard_kwargs["offload_policy"] = offload_policy
                fully_shard(module, **shard_kwargs)
                sharded.add(name)
                logger.debug(f"Applied fully_shard to {name}")

        for pattern, entry in fsdp_plan.items():
            reshard, mp_policy, offload_policy = _parse_plan_entry(entry)

            if pattern in name_to_module:
                target = name_to_module[pattern]
                if isinstance(target, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)):
                    for child_name, child in target.named_children():
                        _shard(f"{pattern}.{child_name}", child, reshard, mp_policy, offload_policy)
                else:
                    _shard(pattern, target, reshard, mp_policy, offload_policy)
            else:
                # Prefix match: "model.layers" matches "model.layers.0", etc.
                for name, module in model.named_modules():
                    if name in sharded or isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential)):
                        continue
                    if name != pattern and not name.startswith(pattern + "."):
                        continue
                    if any(name.startswith(s + ".") for s in sharded):
                        continue
                    _shard(name, module, reshard, mp_policy, offload_policy)

        fully_shard(model, mesh=device_mesh, reshard_after_forward=True)

        # Re-tie weights after fully_shard
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    else:
        raise ValueError(f"fsdp_plan must be 'auto', a list/tuple starting with 'auto', or a dict, got {type(fsdp_plan)}")

    # Used by generation code to detect FSDP and enable synced_gpus.
    model._is_fsdp_managed_module = True

    return model


def distribute_fsdp_model(model, fsdp_plan, device_mesh):
    """
    Distribute a model according to the FSDP plan.

    This function wraps apply_fsdp2 and sets model attributes for tracking.

    Args:
        model: The model to distribute.
        fsdp_plan: Either "auto" or a dict mapping module patterns to strategies.
        device_mesh: The DeviceMesh for FSDP communication.

    Returns:
        The FSDP-distributed model.
    """
    if fsdp_plan is None or device_mesh is None:
        return model

    model = apply_fsdp2(model, device_mesh, fsdp_plan)

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
