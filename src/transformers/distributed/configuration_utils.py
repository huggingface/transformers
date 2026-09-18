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
import warnings
from dataclasses import asdict, dataclass
from typing import Literal

from .utils import _get_torch_distributed_rank


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
            Deprecated alias for `ep_size=tp_size` when `ep_size` is omitted, removed in v5.20. An explicit
            `ep_size` takes precedence. This flag does not change `tp_size` or `fsdp_size`.
        fsdp_size (`int`, *optional*):
            Number of devices for FSDP (data parallelism). If `None` and `tp_size` is set, defaults to 1.
        fsdp_cpu_offload (`bool`, *optional*, defaults to `False`):
            Whether to enable CPU offloading for FSDP2.
        fsdp_mixed_precision (`bool`, *optional*, defaults to `False`):
            Whether to enable mixed precision for FSDP2.
        pp_size (`int`, *optional*):
            Number of devices for pipeline parallelism. If `None` and another parallel mode is set, defaults to 1.
        ep_size (`int`, *optional*):
            Number of devices owning distinct expert shards. Defaults to 1. Set it explicitly to enable EP.
            Model execution currently requires `ep_size=tp_size` when EP is enabled.
    """

    tp_size: int | None = None
    tp_plan: dict[str, str] | Literal["auto"] | None = None
    enable_sequence_parallel: bool = False
    enable_expert_parallel: bool = False
    fsdp_size: int | None = None
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = False
    pp_size: int | None = None
    ep_size: int | None = None

    @property
    def efsdp_size(self) -> int:
        """Size of the expert FSDP axis in the expert mesh view."""
        return self.fsdp_size * self.tp_size // self.ep_size

    def __post_init__(self):
        self._resolve_parallelism()
        self._validate_mesh_config()

    def _resolve_parallelism(self):
        """Resolve parallel sizes and legacy EP settings."""
        for value in (self.tp_size, self.fsdp_size, self.pp_size, self.ep_size):
            if value is not None and value < 1:
                raise ValueError(f"Parallelism sizes must be >= 1, got {value}.")

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

        if self.enable_expert_parallel and self.ep_size is None:
            self.ep_size = self.tp_size
            if _get_torch_distributed_rank() == 0:
                warnings.warn(
                    f"`enable_expert_parallel` without `ep_size` is deprecated and will be removed in v5.20. "
                    f"Use ep_size={self.ep_size} instead.",
                    FutureWarning,
                    stacklevel=4,
                )

        if self.ep_size is None:
            self.ep_size = 1
        # Retain the legacy attribute for callers; internal EP decisions use ep_size.
        self.enable_expert_parallel = self.ep_size > 1

    def _validate_mesh_config(self):
        """Validate mesh sizes before the model's expert plan is available."""
        if self.ep_size > 1:
            if self.ep_size % self.tp_size:
                raise ValueError("`ep_size` must be a multiple of `tp_size`.")
            if (self.fsdp_size * self.tp_size) % self.ep_size:
                raise ValueError("`ep_size` must divide `fsdp_size * tp_size`.")

        if self.fsdp_size > 1 and self.pp_size > 1:
            raise ValueError(
                "Combining FSDP with pipeline parallelism is not supported yet. "
                "Use DistributedConfig(tp_size=N, fsdp_size=M), or combine TP and PP."
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
