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
    fsdp_size: int | None = None
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = False
    pp_size: int | None = None

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

        if self.tp_size > 1 and self.fsdp_size > 1 and self.pp_size > 1:
            raise ValueError(
                "FSDP+TP+PP is not supported yet. "
                "Use DistributedConfig(fsdp_size=N) or DistributedConfig(tp_size=N) or DistributedConfig(pp_size=N), not all three. "
                "Only 1D support is available for now."
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
