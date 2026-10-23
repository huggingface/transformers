# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import ast
import re
from contextlib import contextmanager
from typing import Any

from ..utils.quantization_config import NVFP4Config


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def should_convert_module(full_name, patterns: list[str] | None = None):
    if patterns is None:
        return True

    # We should avoid converting in the following situations:
    # 1. The pattern appears as a prefix followed by a dot in `full_name`
    #    (e.g., "model.decoder.layer.11." matches "model.decoder.layer.11.attn.weight").
    # 2. The pattern matches `full_name` exactly or via regex
    #    (e.g., "lm_head" matches "lm_head"; "model.decoder.layer.*" matches "model.decoder.layer.11.attn.weight").
    # 3. `full_name` ends with the pattern
    #    (e.g., "fc1" matches "model.decoder.layers.23.fc1").

    should_not_convert = any(
        re.match(f"{key}\\.", full_name) or re.match(f"{key}", full_name) or full_name.endswith(key)
        for key in patterns
    )
    return not should_not_convert


@contextmanager
def on_device(device):
    """Align the current accelerator device with a tensor or device-like object."""
    from ..utils import is_torch_available

    if is_torch_available():
        import torch

        if isinstance(device, torch.Tensor):
            device = device.device
        elif isinstance(device, str):
            device = torch.device(device)

        device_type = getattr(device, "type", None)
        if device_type == "cuda":
            with torch.cuda.device(device):
                yield
                return
        if device_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(device):
                yield
                return

    yield


def read_modelopt(config_dict: dict) -> NVFP4Config:
    """Read a ModelOpt export with NVFP4 layers. Its FP8, block FP8 or MXFP8 layers load as they ship."""
    config_dict = {**config_dict, **config_dict.get("quantization", {})}
    if config_dict.get("group_size", 16) != 16:
        raise ValueError(f"NVFP4 uses a group size of 16; got {config_dict['group_size']}.")
    quant_algo = config_dict.get("quant_algo")
    if quant_algo == "MIXED_PRECISION":
        algos = {name: spec.get("quant_algo") for name, spec in (config_dict.get("quantized_layers") or {}).items()}
    else:
        algos = None
    used = set(algos.values()) if algos is not None else {quant_algo}
    # ModelOpt's algorithms, by how they load
    loads = {
        "NVFP4": "nvfp4",
        "W4A16_NVFP4": "nvfp4_weight_only",
        "FP8": "fp8",
        "FP8_PB_WO": "fp8_per_block",
        "MXFP8": "mxfp8",
    }
    if not used & {"NVFP4", "W4A16_NVFP4"}:
        raise ValueError(f"This ModelOpt export has no NVFP4 layer; its formats are {sorted(used)}.")
    if unsupported := used - loads.keys():
        raise ValueError(f"ModelOpt formats {sorted(unsupported)} are not supported next to NVFP4.")
    skip = config_dict.get("modules_to_not_convert") or []
    # some exports write the list as its string
    if isinstance(skip, str):
        skip = ast.literal_eval(skip)
    # ModelOpt matches its exclusions anywhere in a name, with `*` as a wildcard
    globs = [*(config_dict.get("ignore") or []), *(config_dict.get("exclude_modules") or [])]
    skip = [*skip, *(".*" + re.escape(glob).replace(r"\*", ".*") for glob in globs)]
    # a `MIXED_PRECISION` export lists every quantized module, so the default skips (e.g. `lm_head`) do not apply
    if not skip and algos is None:
        skip = None
    quantized_layers = {name: loads[algo] for name, algo in algos.items()} if algos is not None else None
    activation_format = "bf16" if quant_algo == "W4A16_NVFP4" else None
    return NVFP4Config(
        modules_to_not_convert=skip, quantized_layers=quantized_layers, activation_format=activation_format
    )


def read_mxfp8(config_dict: dict) -> NVFP4Config | None:
    """Read an MXFP8 checkpoint whose MoE experts are NVFP4. Return `None` for a plain MXFP8 one.

    The experts load NVFP4 and the other layers MXFP8. The ignored layers stay as they ship.
    """
    if config_dict.get("moe_quant_algo") != "NVFP4":
        return None
    if config_dict.get("group_size", 16) != 16:
        raise ValueError(f"NVFP4 uses a group size of 16; got {config_dict['group_size']}.")
    if (checkpoint_format := config_dict.get("moe_quant_format")) != "nvfp4-pack-quantized":
        raise ValueError(f"NVFP4 experts load from the 'nvfp4-pack-quantized' format; got {checkpoint_format!r}.")
    # an ignored module stays as it ships, with its children
    quantized_layers = {re.escape(name) + r"(\..*)?": None for name in config_dict.get("ignored_layers") or []}
    quantized_layers.update({r".*\.experts": "nvfp4", r".*": "mxfp8"})
    return NVFP4Config(
        modules_to_not_convert=[], quantized_layers=quantized_layers, checkpoint_format=checkpoint_format
    )
