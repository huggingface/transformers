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

from ..integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    apply_tensor_parallelism,
    gather_state_dict_for_save,
    initialize_tensor_parallelism,
)
from ..utils import is_torch_available, is_torch_greater_or_equal, logging
from ..utils.hub import create_and_tag_model_card
from .configuration_utils import DistributedConfig
from .fsdp import apply_fully_sharded_data_parallelism, is_fsdp_managed_module
from .utils import (
    _distributed_barrier,
    _get_torch_distributed_rank,
    _is_torch_distributed_initialized,
    gather_full_state_dict,
    initialize_fully_sharded_data_parallelism,
    save_model_checkpoint_distributed,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch

    _torch_distributed_available = torch.distributed.is_available()
else:
    _torch_distributed_available = False


class DistributedMixin:
    """Distributed orchestration and save/load hooks for [`PreTrainedModel`].

    Stateless heavy lifting stays in `transformers.distributed.*` and
    `integrations.tensor_parallel`. This mixin owns orchestration and instance state.
    """

    _device_mesh = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
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
        """The full tp plan for the model's modules."""
        if hasattr(self.config, "distributed_config") and self.config.distributed_config.enable_expert_parallel:
            if not self._ep_plan:
                raise ValueError(
                    f"Expert parallelism was requested (`enable_expert_parallel=True`), but "
                    f"`{self.__class__.__name__}` does not define an expert-parallel plan. Add a "
                    f"`base_model_ep_plan` to its config, or disable expert parallelism."
                )
            return self._ep_plan
        return self._tp_plan

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

        for layer_pattern, parallel_style in plan.items():
            if parallel_style not in ALL_PARALLEL_STYLES:
                raise ValueError(
                    f"Unsupported tensor parallel style '{parallel_style}' for layer '{layer_pattern}'. "
                    f"Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
                )

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
        *,
        device_mesh=None,
        device_map=None,
    ) -> tuple[DistributedConfig | None, object, object]:
        """Parse ``distributed_config``, init TP/FSDP mesh, and validate."""
        if distributed_config is None:
            return None, device_map, device_mesh

        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)

        if distributed_config.tp_size > 1:
            if distributed_config.tp_plan is None:
                distributed_config.tp_plan = "auto"
            device_map, device_mesh = initialize_tensor_parallelism(
                distributed_config.tp_plan,
                tp_size=distributed_config.tp_size,
                device_mesh=device_mesh,
                device_map=device_map,
            )
        elif distributed_config.fsdp_size > 1:
            device_map, device_mesh = initialize_fully_sharded_data_parallelism(distributed_config)

        distributed_config.validate()
        return distributed_config, device_map, device_mesh

    @classmethod
    def maybe_distribute_model(
        cls,
        model: nn.Module,
        distributed_config: DistributedConfig | None,
        device_mesh,
    ):
        """Apply TP or FSDP2 after model init, before weight loading."""
        if _torch_distributed_available and device_mesh is not None:
            model.config.distributed_config = distributed_config
            model._device_mesh = device_mesh

            if distributed_config.tp_size > 1:
                model = apply_tensor_parallelism(
                    model,
                    distributed_config.tp_plan,
                    distributed_config,
                    device_mesh,
                )
            elif distributed_config.fsdp_size > 1:
                fsdp_mesh = device_mesh["fsdp"] if device_mesh.ndim > 1 else device_mesh
                model = apply_fully_sharded_data_parallelism(model, fsdp_mesh)
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

        if distributed_config.tp_size > 1:
            state_dict = gather_state_dict_for_save(state_dict, self._tp_plan, self._device_mesh, self._tp_size)
            if not save_on_this_rank:
                state_dict = {}
            return state_dict

        if distributed_config.fsdp_size > 1:
            if not _is_torch_distributed_initialized():
                raise ValueError(
                    "Saving an FSDP-wrapped model requires torch.distributed to be initialized. "
                    "Call save_pretrained from every rank after init_process_group."
                )
            return gather_full_state_dict(model_to_save)

        return state_dict

    def barrier_after_gathered_checkpoint_save(self, distributed_config: DistributedConfig | None) -> None:
        """Barrier so non-writer ranks wait for rank 0 to finish gathered checkpoint writes."""
        if distributed_config is None:
            return
        if distributed_config.tp_size > 1 or distributed_config.fsdp_size > 1:
            _distributed_barrier()
