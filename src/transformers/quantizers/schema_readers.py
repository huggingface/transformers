# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Readers of a checkpoint's own quantization scheme (ModelOpt's per-layer `quantized_layers`, DeepSeek-V4's
`expert_dtype`), converting it into our format: a quantization plan `{module pattern: config}`.

A new scheme gets its own function here, added to `SCHEMA_READERS`.
"""

import json
import os
import re
from collections.abc import Callable

from ..utils import logging
from ..utils.quantization_config import FineGrainedFP8Config, Mxfp4Config, NVFP4Config


logger = logging.get_logger(__name__)


# ModelOpt `quant_algo` -> its config
MODELOPT_ALGOS = {
    "NVFP4": NVFP4Config,
    "W4A16_NVFP4": lambda: NVFP4Config(weight_only=True),
    "FP8": lambda: FineGrainedFP8Config(activation_scheme="static", weight_block_size=None),
    "FP8_PB_WO": lambda: FineGrainedFP8Config(activation_scheme="none", weight_block_size=(128, 128)),
    # E4M3 weights, a ue8m0 scale per 32: on the FP8 kernels for now (not their best path on Blackwell)
    "MXFP8": lambda: FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0"),
}


# TODO: MiniMax M3's `moe_quant_algo` (its MoE layers' own algo), once it is supported


# (weight dtype, block scale dtype) in safetensors' names -> the ModelOpt `quant_algo` storing them so
STORED_ALGOS = {("U8", "F8_E4M3"): "NVFP4", ("F8_E4M3", "U8"): "MXFP8", ("F8_E4M3", "F32"): "FP8"}


def stored_layers(dtypes: dict[str, str]) -> dict[str, dict]:
    """
    The quantized modules of a checkpoint, from its tensors' dtypes: a weight (`*.weight` or `*_weight`) with a block
    scale `{weight}_scale` is quantized with the `quant_algo` storing that pair, e.g. Inkling2's
    `mlp.experts.w13_weight` (U8) and `mlp.experts.w13_weight_scale` (F8_E4M3): NVFP4 experts. Per-expert names are
    collapsed (`experts.3.gate_proj` -> `experts.*.gate_proj`).
    """
    layers = {}
    for key, dtype in dtypes.items():
        algo = STORED_ALGOS.get((dtype, dtypes.get(f"{key}_scale")))
        if algo is None or not key.endswith(("weight", "_weight")):
            continue
        module = key.removesuffix(".weight") if key.endswith(".weight") else key.rpartition(".")[0]
        layers[re.sub(r"\.experts\.\d+\.", ".experts.*.", module)] = {"quant_algo": algo}
    return layers


def modelopt_plan(config: dict, checkpoint_dtypes: Callable[[], dict[str, str]] | None = None) -> dict | None:
    """
    ModelOpt's per-layer `quantized_layers`, or its single `quant_algo` for every layer. The layers it leaves out keep
    the base `quant_method` if it is one (e.g. `fp8`), else stay unquantized like its `ignore`d ones.

    A single `quant_algo` may undersell a checkpoint mixing formats, e.g. Inkling2's NVFP4 and MXFP8 experts next to
    bf16 layers under `quant_algo: NVFP4`: when its tensors (`checkpoint_dtypes`) store more than one format, they give
    the per-layer plan instead.
    """
    from .auto import AUTO_QUANTIZATION_CONFIG_MAPPING  # `auto` imports this file

    if "quantized_layers" not in config and config.get("quant_algo") not in MODELOPT_ALGOS:
        return None
    if "quantized_layers" not in config and checkpoint_dtypes is not None:
        layers = stored_layers(checkpoint_dtypes())
        if len({layer["quant_algo"] for layer in layers.values()}) > 1:
            config = {**config, "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}
    layers = config.get("quantized_layers", {"*": config})
    base = {key: value for key, value in config.items() if key != "quantized_layers"}
    base = base if base.get("quant_method") in AUTO_QUANTIZATION_CONFIG_MAPPING else None
    plan = {} if base else dict.fromkeys(config.get("ignore") or config.get("exclude_modules") or [])
    plan |= {name: MODELOPT_ALGOS[layer["quant_algo"]]() for name, layer in layers.items()}
    if config.get("kv_cache_quant_algo"):
        logger.warning(
            f"The checkpoint's `kv_cache_quant_algo={config['kv_cache_quant_algo']}` is not supported: the KV cache "
            "stays in the model's dtype."
        )
    return plan | {"*": base} if base else plan


def fp4_experts_plan(config: dict, checkpoint_dtypes=None) -> dict | None:
    """DeepSeek-V4's FP8 checkpoints, whose routed experts are MXFP4 (`expert_dtype: fp4`)"""
    if config.get("quant_method") != "fp8" or config.get("expert_dtype") != "fp4":
        return None
    return {
        "*.mlp.experts": Mxfp4Config(),
        "*": {key: value for key, value in config.items() if key != "expert_dtype"},
    }


# Tried in order, the first plan wins. ModelOpt first: its per-layer configs (e.g. NVFP4 experts) override DeepSeek's
# `expert_dtype`
SCHEMA_READERS = [modelopt_plan, fp4_experts_plan]


def read_quantization_schema(config: dict, checkpoint_dtypes: Callable[[], dict] | None = None) -> dict | None:
    """
    The quantization plan of the first reader knowing the checkpoint's `quantization_config`, if any. A reader may
    call `checkpoint_dtypes()` for the checkpoint's tensor dtypes (safetensors names, read from the headers only).
    """
    for reader in SCHEMA_READERS:
        plan = reader(config, checkpoint_dtypes)
        if plan is not None:
            return plan
    return None


def safetensors_dtypes(pretrained_model_name_or_path, subfolder: str = "", **hub_kwargs) -> dict[str, str]:
    """`{tensor name: dtype}` of a checkpoint's safetensors files, from their headers (a local directory, else the
    Hub's header ranges). Empty when they can't be read, e.g. offline."""
    if pretrained_model_name_or_path is None:
        return {}
    folder = os.path.join(str(pretrained_model_name_or_path), subfolder)
    if os.path.isdir(folder):
        dtypes = {}
        for file in sorted(os.listdir(folder)):
            if file.endswith(".safetensors"):
                with open(os.path.join(folder, file), "rb") as f:
                    header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
                dtypes |= {name: info["dtype"] for name, info in header.items() if name != "__metadata__"}
        return dtypes
    try:
        from huggingface_hub import get_safetensors_metadata

        metadata = get_safetensors_metadata(
            str(pretrained_model_name_or_path), revision=hub_kwargs.get("revision"), token=hub_kwargs.get("token")
        )
    except Exception:  # noqa: BLE001  offline, gated, no safetensors: the config alone decides
        return {}
    files = metadata.files_metadata.values()
    return {name: info.dtype for file in files for name, info in file.tensors.items()}
