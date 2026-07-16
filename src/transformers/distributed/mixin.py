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

import re
import warnings
from typing import TYPE_CHECKING

from ..integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    apply_tensor_parallelism,
    initialize_tensor_parallelism,
)
from ..utils import is_torch_available
from .configuration_utils import DistributedConfig
from .fsdp import apply_fully_sharded_data_parallelism
from .utils import initialize_fully_sharded_data_parallelism


if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch

    _torch_distributed_available = torch.distributed.is_available()
else:
    _torch_distributed_available = False


class DistributedMixin:
    """Distributed orchestration hooks for [`PreTrainedModel`].

    Stateless heavy lifting stays in `transformers.distributed.*` and
    `integrations.tensor_parallel`. This mixin owns orchestration and instance state.
    """

    _device_mesh = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
    _pp_plan: dict[str, tuple[str, str]] = None
    _fsdp_plan: dict[str, str] | None = None

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
