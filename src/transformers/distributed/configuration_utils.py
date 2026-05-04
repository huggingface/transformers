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

import torch


@dataclass
class DistributedConfig:
    """
    Configuration for native distributed training (FSDP2 + TP).

    Args:
        tp_size (`int`, *optional*):
            Number of devices for tensor parallelism. If `None` and `fsdp_size` is set, defaults to 1.
        tp_plan (`str` or `dict`, *optional*):
            Tensor parallel sharding plan. Use `"auto"` for the model's default plan.
        fsdp_size (`int`, *optional*):
            Number of devices for FSDP (data parallelism). If `None` and `tp_size` is set, defaults to 1.
        fsdp_plan (`str` or `dict`, *optional*):
            FSDP wrapping plan. Use `"auto"` to wrap each transformer layer + root.
    """

    tp_size: int | None = None
    tp_plan: str | dict[str, str] | None = None
    enable_sequence_parallel: bool = False
    fsdp_size: int | None = None
    fsdp_plan: str | dict | None = None

    def __post_init__(self):
        if self.tp_size is None and self.fsdp_size is None:
            return

        if self.tp_size is None:
            self.tp_size = 1
        if self.fsdp_size is None:
            self.fsdp_size = 1
        if self.tp_plan is None:
            self.tp_plan = "auto"
        if self.fsdp_plan is None:
            self.fsdp_plan = "auto"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            assert self.tp_size * self.fsdp_size == world_size, (
                f"tp_size ({self.tp_size}) * fsdp_size ({self.fsdp_size}) must be equal to world_size ({world_size})"
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
