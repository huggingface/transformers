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
from typing import TYPE_CHECKING

from ..utils import is_torch_greater_or_equal, logging
from ..utils.hub import create_and_tag_model_card
from .configuration_utils import DistributedConfig
from .fsdp import apply_fully_sharded_data_parallelism, is_fsdp_managed_module
from .pipeline_parallel import apply_pipeline_parallelism
from .tensor_parallel import (
    _validate_parallel_plan_styles,
    apply_expert_parallelism,
    apply_tensor_parallelism,
    gather_state_dict_for_save,
    resolve_parallel_plans,
)
from .utils import (
    MeshManager,
    _distributed_barrier,
    _get_torch_distributed_rank,
    _is_torch_distributed_initialized,
    gather_full_state_dict,
    initialize_distributed_mesh,
    save_model_checkpoint_distributed,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    import torch.nn as nn


class DistributedMixin:
    """Distributed orchestration and save/load hooks for [`PreTrainedModel`]."""

    _device_mesh = None
    _mesh_manager: MeshManager | None = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
    _fsdp_size = None
    _pp_plan: dict[str, tuple[str, str]] | None = None
    _fsdp_plan: dict[str, str] | None = None

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
        return self._fsdp_plan

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

        _validate_parallel_plan_styles(plan)

        model_param_names = [name for name, _ in self.named_parameters()]
        for layer_pattern in plan.keys():
            regex_pattern = layer_pattern.replace("*", r"\d+")
            pattern_matched = False
            for param_name in model_param_names:
                if re.match(regex_pattern, param_name):
                    pattern_matched = True
                    break
            if not pattern_matched:
                warnings.warn(
                    f"Layer pattern '{layer_pattern}' does not match any parameters in the model. This rule may not "
                    "be applied during tensor parallelization, or may lead to dimension mismatches"
                )

        self._tp_plan = plan

    @ep_plan.setter
    def ep_plan(self, plan: dict[str, str] | None):
        if plan is None:
            self._ep_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `ep_plan`")

        _validate_parallel_plan_styles(plan)
        self._ep_plan = plan

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
    ) -> tuple[DistributedConfig | None, object, MeshManager | None]:
        if distributed_config is None:
            return None, device_map, None

        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)

        if distributed_config.tp_size == 1 and distributed_config.fsdp_size == 1 and distributed_config.pp_size == 1:
            return distributed_config, device_map, None

        if distributed_config.tp_size > 1 and device_map is not None:
            raise ValueError("Tensor parallelism and `device_map` are mutually exclusive.")
        if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.7"):
            raise OSError("FSDP2 requires `torch>=2.7` (distributed checkpoint save/load).")

        device_map, mesh_manager = initialize_distributed_mesh(distributed_config)

        return distributed_config, device_map, mesh_manager

    @classmethod
    def maybe_distribute_model(
        cls,
        model: nn.Module,
        distributed_config: DistributedConfig | None,
        mesh_manager: MeshManager | None,
    ):
        """Apply pipeline, tensor and expert parallelism, then FSDP2, after model init and before weight loading."""
        if mesh_manager is None:
            return model

        model.config.distributed_config = distributed_config
        model._mesh_manager = mesh_manager
        model._device_mesh = mesh_manager.get_mesh(("pp", "fsdp", "tp"))
        model._tp_size = distributed_config.tp_size
        model._fsdp_size = distributed_config.fsdp_size

        # Resolve both plans before sharding anything: overrides are merged into `model.tp_plan` / `model.ep_plan`,
        # and the experts named by the EP plan are removed from the TP plan so they are sharded once.
        tp_plan, ep_plan = resolve_parallel_plans(model, distributed_config)
        distributed_config._validate_resolved_ep_plan(ep_plan)

        if distributed_config.pp_size > 1:
            model = apply_pipeline_parallelism(model, mesh_manager.get_mesh("pp"))

        if tp_plan:
            model = apply_tensor_parallelism(model, mesh_manager.get_mesh("tp"), tp_plan)

        if ep_plan:
            tp_mesh = mesh_manager.get_mesh("tp")
            ep_mesh = mesh_manager.get_mesh("ep")

            if {"ep_router", "moe_tp_experts"}.issubset(ep_plan.values()):
                # Legacy masked EP: the EP group is the TP group, every rank keeps every token.
                model = apply_tensor_parallelism(model, tp_mesh, ep_plan)
            elif "ep_dispatch_experts" in ep_plan.values():
                # EP + DP with tp_size >= 1: the ranks of a TP group share the same batch. If we want a specific token,
                # we will have to slice the batch here in order to avoid computing tp_size times the same batch.
                model = apply_expert_parallelism(model, ep_mesh, tp_mesh, ep_plan)

        if distributed_config.fsdp_size > 1 or "ep_dispatch_experts" in ep_plan.values():
            model = apply_fully_sharded_data_parallelism(model, mesh_manager)

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
        files_timestamps: dict | None = None,
        commit_message: str | None = None,
        token: str | bool | None = None,
        create_pr: bool = False,
    ) -> None:
        """Save an FSDP-wrapped model via DCP and optionally push to the Hub."""
        if not is_torch_greater_or_equal("2.7"):
            raise OSError("save_pretrained(..., distributed_checkpoint=True) requires torch>=2.7.")
        if not is_fsdp_managed_module(model_to_save):
            raise ValueError(
                "save_pretrained(..., distributed_checkpoint=True) is only supported for FSDP-wrapped models."
            )
        if getattr(model_to_save, "_device_mesh", None) is None:
            raise ValueError(
                "save_pretrained(..., distributed_checkpoint=True) requires the model to have been "
                "initialized with a distributed_config (_device_mesh is None)."
            )
        save_model_checkpoint_distributed(model_to_save, save_directory)

        if push_to_hub and save_on_this_rank:
            model_card = create_and_tag_model_card(repo_id, self.model_tags, token=token)
            model_card.save(os.path.join(save_directory, "README.md"))
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

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

        if distributed_config.fsdp_size > 1 or (
            distributed_config.ep_size > 1 and "ep_dispatch_experts" in self.ep_plan.values()
        ):
            # Also covers the 2-D (fsdp, tp) mesh and token dispatch: every parameter is FSDP-managed, and the
            # full state dict is only materialized on rank 0.
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
