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

from ..utils import logging
from ..utils.quantization_config import FineGrainedFP8Config, Mxfp4Config, NVFP4Config


logger = logging.get_logger(__name__)


# ModelOpt `quant_algo` -> its config. W4A16 NVFP4 loads with the W4A4 config, which quantizes activations
MODELOPT_ALGOS = {
    "NVFP4": NVFP4Config,
    "W4A16_NVFP4": NVFP4Config,
    "FP8": lambda: FineGrainedFP8Config(activation_scheme="static", weight_block_size=None),
    "FP8_PB_WO": lambda: FineGrainedFP8Config(activation_scheme="none", weight_block_size=(128, 128)),
    # E4M3 weights, a ue8m0 scale per 32: on the FP8 kernels for now (not their best path on Blackwell)
    "MXFP8": lambda: FineGrainedFP8Config(weight_block_size=(1, 32), scale_fmt="ue8m0"),
}


# TODO: MiniMax M3's `moe_quant_algo` (its MoE layers' own algo), once it is supported


def modelopt_plan(config: dict) -> dict | None:
    """
    ModelOpt's per-layer `quantized_layers`, or its single `quant_algo` for every layer. The layers it leaves out keep
    the base `quant_method` if it is one (e.g. `fp8`), else stay unquantized like its `ignore`d ones.
    """
    from .auto import AUTO_QUANTIZATION_CONFIG_MAPPING  # `auto` imports this file

    if "quantized_layers" not in config and config.get("quant_algo") not in MODELOPT_ALGOS:
        return None
    layers = config.get("quantized_layers", {"*": config})
    base = {key: value for key, value in config.items() if key != "quantized_layers"}
    base = base if base.get("quant_method") in AUTO_QUANTIZATION_CONFIG_MAPPING else None
    plan = {} if base else dict.fromkeys(config.get("ignore") or config.get("exclude_modules") or [])
    plan |= {name: MODELOPT_ALGOS[layer["quant_algo"]]() for name, layer in layers.items()}
    algos = {layer["quant_algo"] for layer in layers.values()}
    if "W4A16_NVFP4" in algos:
        logger.warning(
            "This checkpoint is NVFP4 weight-only (W4A16), but NVFP4 layers quantize their activations to FP4 beyond "
            "single-token decoding, which changes the numerics. Pass `quantization_config=NVFP4Config(dequantize=True)` "
            "to run them in full precision instead."
        )
    if config.get("kv_cache_quant_algo"):
        logger.warning(
            f"The checkpoint's `kv_cache_quant_algo={config['kv_cache_quant_algo']}` is not supported: the KV cache "
            "stays in the model's dtype."
        )
    return plan | {"*": base} if base else plan


def fp4_experts_plan(config: dict) -> dict | None:
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


def read_quantization_schema(config: dict) -> dict | None:
    """The quantization plan of the first reader knowing the checkpoint's `quantization_config`, if any"""
    for reader in SCHEMA_READERS:
        plan = reader(config)
        if plan is not None:
            return plan
    return None
