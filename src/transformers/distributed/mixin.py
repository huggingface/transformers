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
import shutil
from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING

import torch

from ..dynamic_module_utils import custom_object_save
from ..utils import is_torch_greater_or_equal, logging
from ..utils.hub import create_and_tag_model_card
from ..utils.loading_report import LoadStateDictInfo
from .checkpoint import (
    _load_model_checkpoint_distributed,
    _save_model_checkpoint_distributed,
    consolidate_distributed_checkpoint,
)
from .configuration_utils import DistributedConfig
from .expert_parallel import apply_expert_parallelism
from .fsdp import ALL_FSDP_STYLES, _resolve_tied_embed_lm_head_plan, apply_fully_sharded_data_parallelism
from .parallel_utils import ALL_PARALLEL_STYLES, _get_parameter_plan, replace_layer_number_by_wildcard
from .pipeline_parallel import apply_pipeline_parallelism
from .tensor_parallel import apply_tensor_parallelism, gather_state_dict_for_save
from .utils import (
    TransformersDeviceMesh,
    _check_distributed_checkpointing_available,
    _distributed_barrier,
    _get_torch_distributed_rank,
    _is_torch_distributed_initialized,
    gather_full_state_dict,
    initialize_distributed_mesh,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    import torch.nn as nn


def _apply_dtype_plan(model: nn.Module, dtype_plan: dict[str, torch.dtype] | None) -> None:
    """Give the parameters the dtypes they will have once the weights are loaded.

    The model is built uniformly in the requested dtype, but the loader applies `dtype_plan`
    (`_keep_in_fp32_modules{_strict}`) when the checkpoint lands -- which is *after* the model is sharded. The parallel
    plans are built from the parameter dtypes, so they have to see the post-load ones rather than the init ones;
    retyping is free while the parameters are still on meta.
    """
    if not dtype_plan:
        return

    from ..core_model_loading import build_glob_alternation

    pattern, group_to_glob, _ = build_glob_alternation(list(dtype_plan))
    for name, param in model.named_parameters():
        match = pattern.search(name)
        if match is not None and param.dtype != (dtype := dtype_plan[group_to_glob[match.lastgroup]]):
            param.data = param.data.to(dtype)


class DistributedMixin:
    """Distributed orchestration and save/load hooks for [`PreTrainedModel`]."""

    _device_mesh = None
    _is_distributed_loading_by_transformers = False
    _mesh_manager: TransformersDeviceMesh | None = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
    _fsdp_size = None
    _pp_plan: dict[str, tuple[str, str]] | None = None
    _fsdp_plan: dict[str, str] | None = None

    @property
    def is_distributed_loading_by_transformers(self) -> bool:
        """
        Whether `from_pretrained(distributed_config=...)` sharded the model at load time (each rank reads only its
        shard through `DtensorShardOperation`), rather than a wrapper such as Accelerate sharding it afterwards.
        """
        return self._is_distributed_loading_by_transformers

    def init_parallel_plans(self) -> None:
        """Copy class-level plans onto the instance and merge config/children contributions."""
        model_cls = type(self)
        self._tp_plan = dict(getattr(model_cls, "_tp_plan", None) or {})
        self._ep_plan = dict(getattr(model_cls, "_ep_plan", None) or {})
        self._pp_plan = dict(getattr(model_cls, "_pp_plan", None) or {})
        self._fsdp_plan = dict(getattr(model_cls, "_fsdp_plan", None) or {})

        if self.base_model is self:
            self._pp_plan.update(self.config.base_model_pp_plan or {})
            self._tp_plan.update(self.config.base_model_tp_plan or {})
            self._ep_plan.update(self.config.base_model_ep_plan or {})
            self._fsdp_plan.update(self.config.base_model_fsdp_plan or {})
            # Shard the input embeddings on the vocabulary unless the plan already names them.
            if self._tp_plan and isinstance(getattr(self, "embed_tokens", None), torch.nn.Embedding):
                self._tp_plan = {"embed_tokens": "embedding_rowwise", **self._tp_plan}

        for name, module in self.named_children():
            if plan := getattr(module, "_ep_plan", None):
                self._ep_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_tp_plan", None):
                self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_pp_plan", None):
                self._pp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_fsdp_plan", None):
                self._fsdp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})

    @property
    def tp_plan(self) -> dict[str, str]:
        """The full tensor parallel plan for the model's modules."""
        return self._tp_plan if self._tp_plan is not None else {}

    @property
    def ep_plan(self) -> dict[str, str]:
        """The full expert parallel plan for the model's modules, kept separate from `tp_plan`."""
        return self._ep_plan if self._ep_plan is not None else {}

    @property
    def fsdp_plan(self) -> dict[str, str]:
        return self._fsdp_plan if self._fsdp_plan is not None else {}

    @property
    def pp_plan(self) -> dict[str, tuple[str, str]]:
        return self._pp_plan

    @tp_plan.setter
    def tp_plan(self, plan: dict[str, str] | None):
        if plan is None:
            self._tp_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `tp_plan`")

        self._validate_parallel_plan(plan, ALL_PARALLEL_STYLES, "tp_plan")
        self._tp_plan = plan

    @ep_plan.setter
    def ep_plan(self, plan: dict[str, str] | None):
        if plan is None:
            self._ep_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `ep_plan`")

        self._validate_parallel_plan(plan, ALL_PARALLEL_STYLES, "ep_plan")
        self._ep_plan = plan

    @fsdp_plan.setter
    def fsdp_plan(self, plan: dict[str, str] | None):
        if plan is None:
            self._fsdp_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `fsdp_plan`")

        self._validate_parallel_plan(plan, ALL_FSDP_STYLES, "fsdp_plan")
        self._fsdp_plan = plan

    @pp_plan.setter
    def pp_plan(self, plan: dict[str, tuple[str, str]] | None):
        if plan is None:
            self._pp_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `pp_plan`")

        self._pp_plan = plan

    @classmethod
    def prepare_distribute_model(
        cls,
        distributed_config: DistributedConfig | dict | None,
        device_map=None,
    ) -> tuple[DistributedConfig | None, object, TransformersDeviceMesh | None]:
        if distributed_config is None:
            return None, device_map, None

        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)

        if distributed_config.tp_size == 1 and distributed_config.fsdp_size == 1 and distributed_config.pp_size == 1:
            return distributed_config, device_map, None

        if distributed_config.tp_size > 1 and device_map is not None:
            raise ValueError("Tensor parallelism and `device_map` are mutually exclusive.")
        if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.8"):
            raise OSError("FSDP2 requires `torch>=2.8` (distributed checkpoint save/load).")

        device_map, mesh_manager = initialize_distributed_mesh(distributed_config)

        return distributed_config, device_map, mesh_manager

    def _validate_parallel_plan(self, plan: dict[str, str], supported_styles, plan_name: str) -> None:
        """Raise on a style that `supported_styles` does not know, warn on a rule that names nothing in the model."""
        unsupported_styles = {style for style in plan.values() if style not in supported_styles}
        if unsupported_styles:
            raise ValueError(
                f"Unsupported `{plan_name}` styles: {unsupported_styles}. Supported styles are {list(supported_styles)}"
            )

        names = {replace_layer_number_by_wildcard(n) for n, _ in chain(self.named_modules(), self.named_parameters())}
        if unused := plan.keys() - names:
            if _get_torch_distributed_rank() == 0:
                logger.warning(
                    f"`{plan_name}` rules {sorted(unused)} match nothing in {type(self).__name__} and will not be applied."
                )

    def _expand_parallel_plan(self, plan: dict[str, str]) -> dict[str, str]:
        """Replace the `*` of the plan's rules with the layer numbers of the model, one entry per module or parameter."""
        expanded_plan = {}

        for module_name, module in self.named_modules():
            module_style = _get_parameter_plan(parameter_name=module_name, plan=plan, is_weight=False)

            if module_style in ALL_FSDP_STYLES:
                # FSDP: list the module to wrap.
                expanded_plan[module_name] = module_style
            else:
                # TP/EP: list the module's own parameters (to shard) before the module (to hook).
                for param_name, _ in module.named_parameters(recurse=False):
                    name = f"{module_name}.{param_name}" if module_name else param_name
                    param_style = _get_parameter_plan(parameter_name=name, plan=plan, is_weight=True)
                    if param_style is not None:
                        expanded_plan[name] = param_style
                if module_style is not None:
                    expanded_plan[module_name] = module_style

        return expanded_plan

    def _resolve_parallel_plans(
        self, distributed_config: DistributedConfig
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Build the final TP, EP and FSDP plans: apply the config overrides, then replace `*` with the layer numbers.

        Each returned plan maps one concrete module or parameter name to a style, e.g. `model.layers.3.mlp.up_proj`
        instead of `model.layers.*.mlp.up_proj`. A module that has an EP rule keeps no TP rule.
        """
        names = {replace_layer_number_by_wildcard(n) for n, _ in chain(self.named_modules(), self.named_parameters())}
        for plan_name in ("tp_plan", "ep_plan", "fsdp_plan"):
            override = getattr(distributed_config, plan_name)
            if isinstance(override, dict):
                plan = getattr(self, plan_name)
                if unknown := override.keys() - names - plan.keys():
                    raise ValueError(f"`{plan_name}` keys {sorted(unknown)} match nothing in {type(self).__name__}.")
                setattr(self, f"_{plan_name}", {} if plan_name == "tp_plan" and not override else plan | override)

        tp_plan = dict(self.tp_plan) if distributed_config.tp_size > 1 else {}
        ep_plan = dict(self.ep_plan) if distributed_config.ep_size > 1 else {}
        fsdp_plan = dict(self.fsdp_plan) if distributed_config.fsdp_size > 1 else {}

        if distributed_config.tp_size > 1 and not tp_plan:
            raise ValueError(
                f"Tensor parallelism was requested (`tp_size={distributed_config.tp_size}`), but `{type(self).__name__}` "
                "does not define a tensor-parallel plan. Pass `tp_plan` in `DistributedConfig`, add a "
                "`base_model_tp_plan` to the model's config, or disable tensor parallelism."
            )

        if distributed_config.ep_size > 1 and not ep_plan:
            raise ValueError(
                f"Expert parallelism was requested (`ep_size={distributed_config.ep_size}`), but `{type(self).__name__}` "
                "does not define an expert-parallel plan. Pass `ep_plan` in `DistributedConfig`, add a "
                "`base_model_ep_plan` to the model's config, or disable expert parallelism."
            )

        if "ep_dispatch_experts" in ep_plan.values() and "ep_router" in ep_plan.values():
            raise ValueError(
                "`ep_dispatch_experts` routes tokens itself; remove the `ep_router` rules from `ep_plan`."
            )

        if distributed_config.fsdp_size > 1 and not fsdp_plan:
            raise ValueError(
                f"FSDP was requested (`fsdp_size={distributed_config.fsdp_size}`), but `{type(self).__name__}` "
                "does not define an FSDP plan. Pass `fsdp_plan` in `DistributedConfig`, add a "
                "`base_model_fsdp_plan` to the model's config, or disable FSDP."
            )

        # EP rules take precedence: drop TP rules on EP modules and their children.
        is_expert = re.compile(rf"(?:{'|'.join(map(re.escape, ep_plan))})(?:\..+)?").fullmatch
        tp_plan = {name: style for name, style in tp_plan.items() if not is_expert(name)}

        self._validate_parallel_plan(tp_plan, ALL_PARALLEL_STYLES, "tp_plan")
        self._validate_parallel_plan(ep_plan, ALL_PARALLEL_STYLES, "ep_plan")
        self._validate_parallel_plan(fsdp_plan, ALL_FSDP_STYLES, "fsdp_plan")
        distributed_config._validate_resolved_ep_plan(ep_plan)

        # Tied embed/lm_head weights are wrapped once: fold the head's FSDP rule into the embedding's.
        fsdp_plan = _resolve_tied_embed_lm_head_plan(fsdp_plan, self)

        tp_plan = self._expand_parallel_plan(tp_plan)
        ep_plan = self._expand_parallel_plan(ep_plan)
        fsdp_plan = self._expand_parallel_plan(fsdp_plan)

        # FSDP2 all-gathers a group into a single flat buffer, so every group must hold exactly one dtype.
        fsdp_plan = self._split_fsdp_groups_by_dtype(fsdp_plan)

        return tp_plan, ep_plan, fsdp_plan

    def _split_fsdp_groups_by_dtype(self, fsdp_plan: dict[str, str]) -> dict[str, str]:
        """Carve the modules that break a planned FSDP group's dtype uniformity into groups of their own.

        FSDP2 flattens the parameters of a group into one all-gather buffer, so it requires a single original dtype per
        group. A model that keeps some parameters in another dtype (`_keep_in_fp32_modules_strict`, e.g. a router) would
        otherwise trip that check at the first forward. A group holding more than one dtype therefore keeps its dominant
        dtype, and every submodule owning a parameter of another dtype becomes a group of its own, listed ahead of its
        parent (FSDP2 requires children to be sharded first).

        `maybe_distribute_model` has already applied the loader's dtype plan, so the dtypes read here are the ones the
        parameters will have once the weights land.
        """
        carved: dict[str, str] = {}
        for group_name, style in fsdp_plan.items():
            group = self.get_submodule(group_name)
            dtypes = Counter(param.dtype for param in group.parameters())
            if len(dtypes) < 2:
                continue
            dominant = dtypes.most_common(1)[0][0]
            # `named_modules` is a pre-order walk, so reversing it visits children before their parents.
            for sub_name, submodule in reversed(list(group.named_modules())):
                owned = [param for _, param in submodule.named_parameters(recurse=False)]
                # A group that owns mixed dtypes directly cannot be split any further; FSDP2 will report it.
                if sub_name and owned and any(param.dtype != dominant for param in owned):
                    carved[f"{group_name}.{sub_name}"] = style

        # A carved module nested in another carved module of the same dtype is already in a uniform group: drop it,
        # so a float32 router and the statics it holds share one group instead of two.
        for name in list(carved):
            ancestor = next((other for other in carved if name.startswith(f"{other}.")), None)
            if ancestor is not None and {p.dtype for p in self.get_submodule(ancestor).parameters()} == {
                p.dtype for p in self.get_submodule(name).parameters()
            }:
                del carved[name]

        if carved:
            logger.info(f"Split {len(carved)} module(s) into their own FSDP group to keep each group dtype-uniform.")
        return carved | {name: style for name, style in fsdp_plan.items() if name not in carved}

    @classmethod
    def maybe_distribute_model(
        cls,
        model: nn.Module,
        distributed_config: DistributedConfig | None,
        mesh_manager: TransformersDeviceMesh | None,
        dtype_plan: dict[str, torch.dtype] | None = None,
    ):
        """Apply pipeline, tensor and expert parallelism, then FSDP2, after model init and before weight loading."""
        if mesh_manager is None:
            return model

        _apply_dtype_plan(model, dtype_plan)

        model.config.distributed_config = distributed_config
        model._mesh_manager = mesh_manager
        model._device_mesh = mesh_manager.get_mesh(("pp", "fsdp", "tp"))
        model._is_distributed_loading_by_transformers = True
        model._tp_size = distributed_config.tp_size
        model._fsdp_size = distributed_config.fsdp_size

        tp_plan, ep_plan, fsdp_plan = model._resolve_parallel_plans(distributed_config)

        if distributed_config.pp_size > 1:
            model = apply_pipeline_parallelism(model, mesh_manager.get_mesh("pp"))

        if distributed_config.tp_size > 1 and tp_plan:
            model = apply_tensor_parallelism(model, mesh_manager.get_mesh("tp"), tp_plan)

        if distributed_config.ep_size > 1 and ep_plan:
            tp_mesh = mesh_manager.get_mesh("tp")
            ep_mesh = mesh_manager.get_mesh("ep")

            if "megamoe_experts" in ep_plan.values() or {"ep_router", "moe_tp_experts"}.issubset(ep_plan.values()):
                # Legacy masked EP, or Mega MoE dispatching inside its kernel: the EP group is the TP group, every
                # rank keeps every token.
                model = apply_tensor_parallelism(model, tp_mesh, ep_plan)
            elif "ep_dispatch_experts" in ep_plan.values():
                # EP + DP with tp_size >= 1: the ranks of a TP group share the same batch. If we want a specific token,
                # we will have to slice the batch here in order to avoid computing tp_size times the same batch.
                model = apply_expert_parallelism(model, ep_mesh, tp_mesh, ep_plan)

        if distributed_config.fsdp_size > 1 and fsdp_plan:
            model = apply_fully_sharded_data_parallelism(model, mesh_manager, fsdp_plan, ep_plan)

        return model

    def should_save_on_this_rank(self, is_main_process: bool) -> bool:
        """Return whether this rank should write checkpoint files."""
        save_on_this_rank = is_main_process
        if _is_torch_distributed_initialized():
            save_on_this_rank = save_on_this_rank and _get_torch_distributed_rank() == 0
        return save_on_this_rank

    def save_distributed_checkpoint(
        self,
        model_to_save,
        save_directory: str | os.PathLike,
        *,
        push_to_hub: bool = False,
        save_on_this_rank: bool = True,
        repo_id: str | None = None,
        commit_message: str | None = None,
        token: str | bool | None = None,
        create_pr: bool = False,
    ) -> None:
        """Save rank-local DCP files, optionally consolidating into consolidated/ on rank 0 for Hub upload."""
        if not _check_distributed_checkpointing_available():
            raise OSError("save_pretrained(..., distributed_checkpoint=True) requires torch>=2.8.")

        distributed_config = getattr(model_to_save.config, "distributed_config", None)
        if distributed_config is None or (distributed_config.tp_size <= 1 and distributed_config.fsdp_size <= 1):
            raise ValueError(
                "save_pretrained(..., distributed_checkpoint=True) requires an FSDP- or TP-sharded model."
            )
        if getattr(model_to_save, "_device_mesh", None) is None:
            raise ValueError(
                "save_pretrained(..., distributed_checkpoint=True) requires the model to have been "
                "initialized with a distributed_config (_device_mesh is None)."
            )
        _save_model_checkpoint_distributed(model_to_save, save_directory)

        if push_to_hub:
            try:
                if self.should_save_on_this_rank(save_on_this_rank):
                    output_dir = os.path.join(save_directory, "consolidated")
                    # Everything in output_dir is uploaded, so drop leftovers from a previous save.
                    shutil.rmtree(output_dir, ignore_errors=True)
                    consolidate_distributed_checkpoint(save_directory, output_dir)
                    if self.is_remote_code():
                        custom_object_save(model_to_save, output_dir, config=model_to_save.config)
                    model_to_save.config.save_pretrained(output_dir)
                    if self.can_generate():
                        model_to_save.generation_config.save_pretrained(output_dir)
                    model_card = create_and_tag_model_card(repo_id, self.model_tags, token=token)
                    model_card.save(os.path.join(output_dir, "README.md"))
                    self._upload_modified_files(
                        output_dir,
                        repo_id,
                        files_timestamps={},
                        commit_message=commit_message,
                        token=token,
                        create_pr=create_pr,
                    )
            finally:
                _distributed_barrier()

    def gather_sharded_state_dict_for_save(
        self,
        model_to_save,
        state_dict: dict,
        distributed_config: DistributedConfig | None,
        *,
        save_on_this_rank: bool = True,
    ) -> dict:
        """Gather TP- or FSDP-sharded weights to full CPU tensors for checkpoint writing."""
        if distributed_config is None:
            return state_dict

        if distributed_config.fsdp_size > 1:
            # Also covers the 2-D (fsdp, tp) mesh and token dispatch with expert replicas: every parameter is
            # FSDP-managed, and the full state dict is only materialized on rank 0.
            if not _is_torch_distributed_initialized():
                raise ValueError(
                    "Saving an FSDP-wrapped model requires torch.distributed to be initialized. "
                    "Call save_pretrained from every rank after init_process_group."
                )
            return gather_full_state_dict(model_to_save)

        if distributed_config.tp_size > 1:
            state_dict = gather_state_dict_for_save(
                state_dict, self._tp_plan, self._device_mesh, distributed_config.tp_size
            )
            if not save_on_this_rank:
                state_dict = {}
            return state_dict

        return state_dict

    def barrier_after_gathered_checkpoint_save(self, distributed_config: DistributedConfig | None) -> None:
        """Barrier so non-writer ranks wait for rank 0 to finish gathered checkpoint writes."""
        if distributed_config is None:
            return
        if distributed_config.tp_size > 1 or distributed_config.fsdp_size > 1 or distributed_config.ep_size > 1:
            _distributed_barrier()

    def _load_and_finalize_distributed_pretrained_model(self, distributed_checkpoint_dir, device_map, device_mesh):
        """
        Loads a distributed checkpoint into an uninitialized model:

            1. DCP expects the parameters to be materialized before loading, so we move them from meta to device.
            2. Restore parameter sharing broken by materialization by tying the weights again.
            3. Load the distributed checkpoint into the model.
            4. Initialize any non-persistent buffers that DCP cannot provide.
        """
        # First, materialize the parameters.
        self._move_missing_keys_from_meta_to_device(set(self.state_dict()), device_map, device_mesh, None)
        self.tie_weights(recompute_mapping=False)

        # Load the distributed checkpoint into the model.
        _load_model_checkpoint_distributed(self, distributed_checkpoint_dir)

        # Everything in the state dict was loaded: only initialize what DCP cannot provide (non-persistent buffers).
        for tensor in self.state_dict(keep_vars=True).values():
            tensor._is_hf_initialized = True
        self.initialize_weights()

        # DCP loads strictly: a missing key raises, so there is nothing to report.
        loading_info = LoadStateDictInfo(
            missing_keys=set(),
            unexpected_keys=set(),
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
            skipped_pp_keys=set(),
        )
        return loading_info
