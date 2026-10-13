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

import os
import re
import warnings
from datetime import timedelta
from itertools import chain
from typing import TYPE_CHECKING, TypeGuard

from ..utils import is_torch_available, is_torch_distributed_available, is_torch_greater_or_equal, logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor

    from .configuration_utils import DistributedConfig


if is_torch_available():
    import torch


def _is_torch_distributed_initialized() -> bool:
    if not is_torch_distributed_available():
        return False
    return torch.distributed.is_initialized()


def is_dtensor(obj: object) -> TypeGuard[DTensor]:
    if not is_torch_distributed_available():
        return False
    from torch.distributed.tensor import DTensor

    return isinstance(obj, DTensor)


def _get_torch_distributed_rank() -> int:
    if not _is_torch_distributed_initialized():
        return 0
    return torch.distributed.get_rank()


def _get_torch_distributed_world_size() -> int:
    if not _is_torch_distributed_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_local_dist_rank_0() -> bool:
    return _is_torch_distributed_initialized() and int(os.environ.get("LOCAL_RANK", "-1")) == 0


DISTRIBUTED_BACKEND_MAP = {
    "cuda": "nccl",
    "cpu": "gloo",
    "xpu": "xccl",
    "hpu": "hccl",
    "neuron": "neuron",
    "tpu": "tpu_dist",
}


def get_distributed_backend(device_type: str) -> str | None:
    """Return the `torch.distributed` backend for `device_type`, or `None` if it has no dedicated one."""
    dist_backend = DISTRIBUTED_BACKEND_MAP.get(device_type)
    if dist_backend is None:
        raise ValueError(f"No distributed backend found for device type '{device_type}'")
    return dist_backend


def _ensure_torch_distributed(device_type: str | None = None):
    """Initialize torch.distributed if not already initialized.

    If `device_type` is not given, it is detected from the current accelerator.
    """
    if not torch.distributed.is_initialized():
        if device_type is None:
            device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            logger.warning_once(
                "PyTorch's built-in DeviceMesh/DTensor stack does not support an MPS mesh. Falling back to CPU."
            )
            device_type = "cpu"
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            backend = get_distributed_backend(device_type)

            # Bind the accelerator before init so the process group is created with a
            # device_id, otherwise collectives like barrier() warn (and may spin up an
            # extra NCCL comm) about the missing device binding.
            device_id = None
            if device_type != "cpu":
                torch.accelerator.set_device_index(local_rank)
                device_id = torch.device(device_type, local_rank)
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                device_id=device_id,
                # Sharded loading takes tens of minutes with high rank skew; the default 10-minute watchdog is too short
                timeout=timedelta(hours=2),
            )
        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use distributed training."
            ) from e


def _distributed_barrier():
    """Barrier bound to the current accelerator device.

    Passing `device_ids` is required when the process group was initialized without a
    `device_id`; with it, the call is a no-op compared to plain `barrier()`. Safe to call
    when torch.distributed has not been initialized — returns immediately.
    """
    if not _is_torch_distributed_initialized():
        return
    device_type = torch._C._get_accelerator().type
    if device_type != "cpu":
        torch.distributed.barrier(device_ids=[torch.accelerator.current_device_index()])
    else:
        torch.distributed.barrier()


class TransformersDeviceMesh:
    """
    Holds the device meshes used by a model.

    dense layers and experts are sharded differently, so they need different views of the same ranks.

        dense  : (pp, fsdp, tp)     attention, dense MLPs, embeddings, lm_heads
        expert : (pp, efsdp, ep)    experts

    Both views cover the same world, so pp * fsdp * tp == pp * efsdp * ep.
    efsdp is not something you pick, it is whatever is left once ep is fixed. The relationship is as follow:
    efsdp = fsdp * tp / ep. It is the FSDP axis for expert weights same role `fsdp` plays for the dense params.
    There is no etp (expert tensor parallel) axis: with EP on, experts are never tensor-sharded. (for now)
    For experts, EP plays the role TP plays for dense layers (weights stay sharded during compute),
    and efsdp plays the role of fsdp (weights all-gathered before compute).

    If one were ever added, the relationship would become pp * efsdp * ep * etp == pp * fsdp * tp and efsdp would shrink by etp
    (efsdp = fsdp * tp / (ep * etp))

    When ep_size == tp_size, efsdp == fsdp (given the relationship efsdp = fsdp * tp / ep), thus same axis, same size and same groups of ranks.
    In that case experts could reuse the dense mesh's fsdp axis.
    When ep_size != tp_size, efsdp != fsdp, so we can't reuse the same mesh as they don't have the same groups of ranks
    This explains why experts need their own efsdp axis.

    Regarding ep value, one can decide to default the value to node width (8 on most machines) so that all-to-all never leaves the node.
    That has several implications on efsdp value given your setup:
    - On a single node, ep == fsdp * tp thus efsdp = 1, the axis does nothing.
    - On several nodes, we still keep ep at node width, since all-to-all across nodes is expensive.
    However, each node then holds a full copy of the expert group and efsdp is the number of copies, which is where FSDP happens for the experts
    i.e: 2 nodes x 8 GPUs -> efsdp = 16 / 8 = 2, one EP group per node, two copies, sharded over efsdp.
    """

    def __init__(self, dense_mesh: DeviceMesh, expert_mesh: DeviceMesh):
        self._dense_mesh = dense_mesh
        self._expert_mesh = expert_mesh

    def get_mesh(self, dims: str | tuple[str, ...]) -> DeviceMesh:
        """Select expert axes for `ep`/`efsdp`, otherwise dense axes; DeviceMesh handles slicing."""
        dims = (dims,) if isinstance(dims, str) else dims
        mesh = self._expert_mesh if "ep" in dims or "efsdp" in dims else self._dense_mesh
        return mesh[dims]


# Retained for the legacy transformers.integrations.tensor_parallel API.
def initialize_tensor_parallelism(
    tp_plan: str | dict[str, str] | None, tp_size: int | None = None, device_mesh=None, device_map=None
):
    r"""
    Sets up the device mesh and initialized the backend for tensor parallelism.
    This function is called when the model is loaded and the TP plan is set to 'auto'.
    """
    warnings.warn(
        "`initialize_tensor_parallelism` is deprecated and will be removed in a future release. "
        "Use `initialize_distributed_mesh` with a `DistributedConfig` instead.",
        FutureWarning,
        stacklevel=2,
    )
    if tp_size is not None and tp_plan is None:
        raise ValueError("tp_plan has to be set when tp_size is passed.")
    if tp_plan is not None and device_map is not None:
        raise ValueError("`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization.")
    if device_mesh is None:
        # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            logger.warning_once(
                "PyTorch's built-in DeviceMesh/DTensor stack does not support an MPS mesh. Falling back to CPU."
            )
            device_type = "cpu"
        if device_type != "cpu":
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.accelerator.set_device_index(local_rank)
            tp_device = torch.device(device_type, local_rank)
            device_map = tp_device
        else:
            tp_device = torch.device(device_type)
            device_map = device_type or {}

        device_mesh = torch.distributed.init_device_mesh(tp_device.type, (tp_size,))
    else:
        if device_mesh.ndim > 1:
            if "tp" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `tp_plan` and n-d `device_mesh`, it must contain a 'tp' dimension. "
                    "Please provide a valid `device_mesh`."
                )
            device_mesh = device_mesh["tp"]
        device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

    return device_map, device_mesh


def initialize_fully_sharded_data_parallelism(distributed_config: DistributedConfig):
    warnings.warn(
        "`initialize_fully_sharded_data_parallelism` is deprecated and will be removed in a future release. "
        "Use `initialize_distributed_mesh` with a `DistributedConfig` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # `fully_shard` itself only needs torch>=2.6, but distributed checkpoint save/load
    # (DCP + HuggingFaceStorageWriter) needs 2.7, so that is the effective requirement.
    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.7"):
        raise OSError("FSDP2 requires `torch>=2.7` (distributed checkpoint save/load).")

    device_type = torch._C._get_accelerator().type

    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.accelerator.set_device_index(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    fsdp_size = distributed_config.fsdp_size

    dims, names = [], []
    if fsdp_size > 1:
        dims.append(fsdp_size)
        names.append("fsdp")

    # Build the N-dimensional device mesh
    mesh = torch.distributed.init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    # If N > 1, create a flattened sub-mesh so all-reduces across the world mesh ae done in one collective
    if len(dims) > 1:
        mesh._flatten("_".join(names))

    return device_map, mesh


def initialize_distributed_mesh(
    distributed_config: DistributedConfig,
) -> tuple[torch.device | None, TransformersDeviceMesh | None]:
    """Create a device mesh containing every configured parallel dimension."""
    mesh_shape = (distributed_config.pp_size, distributed_config.fsdp_size, distributed_config.tp_size)
    if mesh_shape == (1, 1, 1):
        return None, None

    device_type = torch._C._get_accelerator().type
    if distributed_config.tp_size > 1 and device_type == "mps":
        raise RuntimeError("Tensor parallelism is not supported on MPS devices.")

    _ensure_torch_distributed(device_type)
    world_size = torch.distributed.get_world_size()
    expected_world_size = distributed_config.pp_size * distributed_config.fsdp_size * distributed_config.tp_size
    if expected_world_size != world_size:
        raise RuntimeError(
            f"The parallel mesh requires {expected_world_size} processes, but world_size is {world_size}."
        )

    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.accelerator.set_device_index(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    dense_mesh = torch.distributed.init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=("pp", "fsdp", "tp"),
    )
    expert_mesh = torch.distributed.init_device_mesh(
        device_type,
        (distributed_config.pp_size, distributed_config.efsdp_size, distributed_config.ep_size),
        mesh_dim_names=("pp", "efsdp", "ep"),
    )
    return device_map, TransformersDeviceMesh(dense_mesh, expert_mesh)


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather FSDP-sharded params to full plain CPU tensors.

    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=options)
    if _get_torch_distributed_rank() == 0:
        return full_state_dict
    return {}


def save_model_checkpoint_distributed(model, checkpoint_dir: str) -> None:
    """Save model parameters as standard HF-format sharded safetensors using
    DCP + HuggingFaceStorageWriter with consolidation enabled.

    Every rank first writes its own shard in parallel under
    `<checkpoint_dir>/sharded/`, then a consolidation pass reads those shards
    and emits HF-compatible `model-*-of-N.safetensors` (+ index) at
    `<checkpoint_dir>/`. The result is a directory `from_pretrained` reads
    through its normal path — no special flag needed at load time.
    """
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state_dict = get_model_state_dict(model)
    dcp.save(
        state_dict,
        storage_writer=HuggingFaceStorageWriter(
            path=checkpoint_dir,
            save_distributed=True,
            enable_consolidation=True,
        ),
    )
    # Wait for rank 0 to finish writing the HF safetensors so other
    # ranks don't return (and hit `from_pretrained`) before the files exist.
    _distributed_barrier()


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Save optimizer state via DCP."""
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    dcp.save({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)


def load_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Load optimizer state via DCP."""
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    dcp.load({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace layer indices in a module/param FQN with ``.*`` (e.g. ``layers.3.mlp`` → ``layers.*.mlp``).
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_plan(parameter_name: str, plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the parallel style for a parameter or module from a TP or EP plan.

    The plan is a dictionary that maps parameter or module names to parallel styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in plan:
        return plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in plan:
        return plan[module_name]
    return None


def _validate_parallel_plan_styles(plan: dict[str, str] | None) -> None:
    from .tensor_parallel import ALL_PARALLEL_STYLES

    unsupported_styles = {style for style in (plan or {}).values() if style not in ALL_PARALLEL_STYLES}
    if unsupported_styles:
        raise ValueError(
            f"Unsupported parallel styles: {unsupported_styles}. Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
        )


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
        elif key not in name_lookup and not any(replace_layer_number_by_wildcard(name) == key for name in name_lookup):
            unused_rules[key] = strategy

    if invalid_strategies:
        logger.warning(f"The following FSDP entries have unknown strategies: {invalid_strategies}")
    if unused_rules:
        logger.warning(f"The following FSDP rules were not applied to any module: {unused_rules}")


def expand_parallel_plan(model: nn.Module, plan: dict[str, str]) -> dict[str, str]:
    """Expand plan patterns to concrete FQNs, placing each module's direct parameters before its hook target."""
    from .tensor_parallel import ALL_PARALLEL_STYLES

    expanded_plan = {}

    for name, module in model.named_modules():
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_plan(parameter_name=full, plan=plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                expanded_plan[full] = style_name

        style_name = _get_parameter_plan(parameter_name=name, plan=plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            expanded_plan[name] = style_name

    return expanded_plan


def _get_input_output_embeddings(model: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    input_embed = None
    output_head = None
    if hasattr(model, "get_input_embeddings"):
        input_embed = model.get_input_embeddings()
    if hasattr(model, "get_output_embeddings"):
        output_head = model.get_output_embeddings()
    return input_embed, output_head


def _resolve_tied_embed_lm_head_plan(
    fsdp_plan: dict[str, str],
    model: nn.Module,
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
    tied_keys = getattr(model, "all_tied_weights_keys", None) or {}
    if not tied_keys:
        return fsdp_plan

    input_embed, output_head = _get_input_output_embeddings(model)
    name_by_module = {module: name for name, module in model.named_modules()}
    embed_module = name_by_module.get(input_embed)
    head_module = name_by_module.get(output_head)

    if embed_module is None or head_module is None:
        return fsdp_plan

    adapted_plan = fsdp_plan.copy()
    adapted_plan.pop(embed_module, None)

    if fsdp_plan.get(head_module) == "keep_full_weight":
        adapted_plan.pop(head_module, None)
        adapted_plan[embed_module] = "keep_full_weight"

    return adapted_plan


def expand_fsdp_plan(
    model: nn.Module,
    fsdp_plan: dict[str, str],
) -> dict[str, str]:
    """Expand plan patterns to concrete module FQNs and their reshard policies."""
    fsdp_plan = _resolve_tied_embed_lm_head_plan(fsdp_plan, model)

    expanded_plan = {}

    for module_name, _ in model.named_modules():
        plan_key = module_name if module_name in fsdp_plan else replace_layer_number_by_wildcard(module_name)
        if plan_key in fsdp_plan:
            expanded_plan[module_name] = fsdp_plan[plan_key]

    return expanded_plan


def resolve_parallel_plans(
    model: nn.Module, distributed_config: DistributedConfig
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Merge config overrides and expand TP, EP, and FSDP plans to concrete apply targets.

    TP/EP dictionaries include both parameter and module FQNs. The FSDP dictionary maps module FQNs to reshard policies.
    EP owns every module it names, so matching TP rules are dropped before expansion.
    """
    names = {replace_layer_number_by_wildcard(n) for n, _ in chain(model.named_modules(), model.named_parameters())}
    for plan_name in ("tp_plan", "ep_plan", "fsdp_plan"):
        override = getattr(distributed_config, plan_name)
        if isinstance(override, dict):
            plan = getattr(model, plan_name)
            if unknown := override.keys() - names - plan.keys():
                raise ValueError(f"`{plan_name}` keys {sorted(unknown)} match nothing in {type(model).__name__}.")
            setattr(model, f"_{plan_name}", plan | override)

    tp_plan = dict(model.tp_plan) if distributed_config.tp_size > 1 else {}
    ep_plan = dict(model.ep_plan) if distributed_config.ep_size > 1 else {}
    fsdp_plan = dict(model.fsdp_plan) if distributed_config.fsdp_size > 1 else {}

    if distributed_config.tp_size > 1 and not tp_plan:
        raise ValueError(
            f"Tensor parallelism was requested (`tp_size={distributed_config.tp_size}`), but `{type(model).__name__}` "
            "does not define a tensor-parallel plan. Pass `tp_plan` in `DistributedConfig`, add a "
            "`base_model_tp_plan` to the model's config, or disable tensor parallelism."
        )

    if distributed_config.ep_size > 1 and not ep_plan:
        raise ValueError(
            f"Expert parallelism was requested (`ep_size={distributed_config.ep_size}`), but `{type(model).__name__}` "
            "does not define an expert-parallel plan. Pass `ep_plan` in `DistributedConfig`, add a "
            "`base_model_ep_plan` to the model's config, or disable expert parallelism."
        )

    if "ep_dispatch_experts" in ep_plan.values() and "ep_router" in ep_plan.values():
        raise ValueError("`ep_dispatch_experts` routes tokens itself; remove the `ep_router` rules from `ep_plan`.")

    if distributed_config.fsdp_size > 1 and not fsdp_plan:
        raise ValueError(
            f"FSDP was requested (`fsdp_size={distributed_config.fsdp_size}`), but `{type(model).__name__}` "
            "does not define an FSDP plan. Pass `fsdp_plan` in `DistributedConfig`, add a "
            "`base_model_fsdp_plan` to the model's config, or disable FSDP."
        )

    is_expert = re.compile(rf"(?:{'|'.join(map(re.escape, ep_plan))})(?:\..+)?").fullmatch
    tp_plan = {name: style for name, style in tp_plan.items() if not is_expert(name)}

    _validate_parallel_plan_styles(tp_plan)
    _validate_parallel_plan_styles(ep_plan)
    distributed_config._validate_resolved_ep_plan(ep_plan)
    verify_fsdp_plan([name for name, _ in model.named_modules()], fsdp_plan)

    tp_plan = expand_parallel_plan(model, tp_plan) if tp_plan else {}
    ep_plan = expand_parallel_plan(model, ep_plan) if ep_plan else {}
    fsdp_plan = expand_fsdp_plan(model, fsdp_plan) if fsdp_plan else {}

    return tp_plan, ep_plan, fsdp_plan
