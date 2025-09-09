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

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Union

import torch
from ..utils import is_torch_greater_or_equal

@dataclass
class DistributedConfig:
    """
    Base class for distributed configs
    """

    enable_expert_parallel: bool = False
    # TODO: add tp_plan, pp_plan, device_mesh etc..

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a DistributedConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.
        Returns:
            DistributedConfig: Instance of DistributedConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.
        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.
        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


def initialize_parallelism(tp_plan, pp_plan, tp_size=None, pp_size=None):
    """
    Initializes the parallelism and returns all the necessary variables.
    """
    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Tensor parallel is only supported for `torch>=2.5`.")

    # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
    device_type = torch._C._get_accelerator().type
    current_device = getattr(torch, device_type)
    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            assert tp_size * pp_size == world_size, f"tp_size ({tp_size}) * pp_size ({pp_size}) must be equal to world_size ({world_size})"
    
            backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
            backend = backend_map.get(device_type)
            if device_type == "cpu" and int(os.environ.get("CCL_WORKER_COUNT", "0")):
                backend = "ccl"
            if device_type == "xpu":

                if not is_torch_greater_or_equal("2.8", accept_dev=True):
                    backend = "ccl"

            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
            current_device = getattr(torch, device_type)
            if device_type != "cpu":
                current_device.set_device(local_rank)

        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use `tp_plan='auto'`."
            ) from e

    if device_type != "cpu":
        current_device.set_device(int(os.environ["LOCAL_RANK"]))
    index = current_device.current_device() if device_type != "cpu" else None

    # Silence output for non-primary ranks
    if index is not None and index > 0:
        import sys

        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    device_mesh = torch.distributed.init_device_mesh(device_type, (tp_size, pp_size), mesh_dim_names=("tp", "pp"))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = torch.device(f"{device_type}:{local_rank}")

    return tp_plan, pp_plan, device_map, device_mesh