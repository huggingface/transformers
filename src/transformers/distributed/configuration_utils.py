# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal


# How the expert outputs get back to the tokens that need them, mapped to the parallel style the experts take.
# `all-reduce` runs the whole batch on every rank and needs no style of its own. A new backend such as DeepEP is a
# new entry here plus the matching style in `ParallelInterface`, not a new flag.
EXPERTS_DISPATCH_STRATEGIES = {"all-reduce": None, "all-to-all": "ep_dispatch_experts"}


@dataclass
class DistributedConfig:
    """
    Configuration for native distributed inference and training with tensor, pipeline, or FSDP2 parallelism.

    Args:
        tp_size (`int`, *optional*):
            Number of devices for tensor parallelism. If `None` and `tp_plan` is set, defaults to
            `WORLD_SIZE // (other_parallel_size)`. If `None` and no `tp_plan` is set, defaults to 1.
        tp_plan (`dict[str, str]` or `"auto"`, *optional*):
            Tensor parallel sharding plan. Pass `"auto"`, or leave as `None` when `tp_size` is set, to use the
            model's predefined `base_model_tp_plan`. Pass a dictionary to override the predefined plan.
        enable_sequence_parallel (`bool`, *optional*, defaults to `False`):
            Reserved for sequence parallelism. Not wired up yet.
        enable_expert_parallel (`bool`, *optional*, defaults to `False`):
            Route MoE models through the expert-parallel path (``base_model_ep_plan``).
        experts_dispatch (`str`, *optional*, defaults to `"all-reduce"`):
            How the expert outputs get back to the tokens that need them. `"all-reduce"` runs the whole batch on
            every rank and all-reduces the expert outputs. `"all-to-all"` sends each token to the rank that owns its
            experts instead, so each rank trains on its own part of the batch and the parameters that are not
            expert-parallel are sharded with FSDP2 across every rank. Anything but `"all-reduce"` requires
            `enable_expert_parallel`.
        fsdp_size (`int`, *optional*):
            Number of devices for FSDP (data parallelism). If `None` and `tp_size` is set, defaults to 1.
        fsdp_cpu_offload (`bool`, *optional*, defaults to `False`):
            Whether to enable CPU offloading for FSDP2.
        fsdp_mixed_precision (`bool`, *optional*, defaults to `False`):
            Whether to enable mixed precision for FSDP2.
        pp_size (`int`, *optional*):
            Number of devices for pipeline parallelism. If `None` and another parallel mode is set, defaults to 1.
    """

    tp_size: int | None = None
    tp_plan: dict[str, str] | Literal["auto"] | None = None
    enable_sequence_parallel: bool = False
    enable_expert_parallel: bool = False
    experts_dispatch: str = "all-reduce"
    fsdp_size: int | None = None
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = False
    pp_size: int | None = None

    @property
    def dispatches_tokens(self) -> bool:
        """Whether each rank routes and trains on its own part of the batch, which is every strategy but the
        `all-reduce` default."""
        return self.experts_dispatch != "all-reduce"

    def __post_init__(self):
        if self.tp_plan is None and self.tp_size is None and self.fsdp_size is None and self.pp_size is None:
            return

        if self.fsdp_size is None:
            self.fsdp_size = 1
        if self.pp_size is None:
            self.pp_size = 1
        if self.tp_size is None and self.tp_plan is not None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            other_parallel_size = self.fsdp_size * self.pp_size
            if world_size % other_parallel_size != 0:
                raise ValueError(
                    f"WORLD_SIZE ({world_size}) must be divisible by fsdp_size * pp_size "
                    f"({other_parallel_size}) to derive tp_size."
                )
            self.tp_size = world_size // other_parallel_size
        elif self.tp_size is None:
            self.tp_size = 1

        if self.experts_dispatch not in EXPERTS_DISPATCH_STRATEGIES:
            raise ValueError(
                f"Unknown `experts_dispatch={self.experts_dispatch!r}`, expected one of "
                f"{sorted(EXPERTS_DISPATCH_STRATEGIES)}."
            )
        if self.dispatches_tokens and not self.enable_expert_parallel:
            raise ValueError(f"`experts_dispatch={self.experts_dispatch!r}` requires `enable_expert_parallel=True`.")

        if self.pp_size > 1 and (self.tp_size > 1 or self.fsdp_size > 1):
            raise ValueError(
                "Pipeline parallelism cannot be combined with tensor or FSDP parallelism yet. "
                "Use DistributedConfig(pp_size=N) on its own, or DistributedConfig(tp_size=N, fsdp_size=M)."
            )

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs) -> "DistributedConfig":
        merged = {**config_dict, **kwargs}
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in merged.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
