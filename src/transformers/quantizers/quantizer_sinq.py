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
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Dict, Any, List

import torch
import torch.nn as nn

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name, should_convert_module
from ..utils import logging
from ..utils.quantization_config import SinqConfig

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

def _normalize_cuda_device(dev: Optional[Union[str, int]]) -> str:
    if dev is None or dev == "auto":
        if torch.cuda.is_available():
            idx = torch.cuda.current_device() if torch.cuda.device_count() else 0
            return f"cuda:{idx}"
        return "cpu"

    if dev == "cuda":
        return "cuda:0"

    if dev == "cpu":
        return "cpu"

    if isinstance(dev, int):
        if torch.cuda.is_available():
            return f"cuda:{dev}"
        return "cpu"

    if isinstance(dev, str) and dev.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        return dev

    raise ValueError(f"Unsupported device spec: {dev!r}")

def _flatten_device_map(dmap: Optional[dict]) -> set[str]:
    if not isinstance(dmap, dict):
        return set()
    out: set[str] = set()

    def _walk(v):
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, dict):
            for vv in v.values():
                _walk(vv)

    _walk(dmap)
    return out

class SinqHfQuantizer(HfQuantizer):
    """
    HF v5 quantizer for SINQ.

    Modes:
      - method="sinq" (default):
          * weight-only SINQ
          * param-level ConversionOps (`SinqQuantize`) during load for pure language models
            (each Linear.weight is turned into a SINQLinear module)
          * module-level quantization after load for multimodal models
      - method="asinq":
          * A-SINQ (activation-aware) SINQ quantization
    """

    requires_calibration: bool = False
    requires_parameters_quantization: bool = True

    def __init__(self, quantization_config: SinqConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self._normalized_device_str: str | None = None
        self._do_param_level_sinq: bool = False

    def is_serializable(self, safe_serialization: Optional[bool] = None) -> bool:
        if safe_serialization:
            return True
        return False

    @property
    def is_trainable(self) -> bool:
        return True

    def validate_environment(self, *args, **kwargs) -> None:
        from ..utils import is_sinq_available
        if not is_sinq_available():
            raise ImportError(
                "The 'sinq' package is not installed. Please install it with: pip install sinq"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("SINQ currently expects a CUDA device (for GemLite backend).")
        
        # Validate and set dtype
        passed_dtype = kwargs.get("torch_dtype", None) or kwargs.get("dtype", None)
        if passed_dtype is not None:
            if isinstance(passed_dtype, str):
                # Convert string to torch dtype
                if not hasattr(torch, passed_dtype):
                    raise ValueError(f"Unsupported torch_dtype string: {passed_dtype!r}")
                passed_dtype = getattr(torch, passed_dtype)
            
            if not isinstance(passed_dtype, torch.dtype):
                raise TypeError(f"Expected torch.dtype, got {type(passed_dtype)}")
            
            # Warn if using unsupported dtype for SINQ
            if passed_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                logger.warning(
                    f"SINQ quantization with dtype={passed_dtype} may not be supported. "
                    f"Recommended dtypes: torch.float16, torch.bfloat16, torch.float32"
                )
            
            self.dtype = passed_dtype
        else:
            # Set default dtype if not provided
            self.dtype = torch.bfloat16

        device_map = kwargs.get("device_map", None)

        devs = _flatten_device_map(device_map)
        if devs:
            if len(devs) > 1:
                raise RuntimeError(
                    "SinqHfQuantizer: multi-GPU device_map detected, but SINQ currently supports only a single CUDA "
                    f"device. Got {sorted(devs)}. Please use device_map=None"# and set SinqConfig(device='{device_str}')."
                )

    def _build_sinq_quant_dict(self, cfg: SinqConfig) -> dict:
        """
        Build the dict that SINQLinear expects as quant_config.
        """
        from sinq.sinqlinear_hf import sinq_base_quant_config as sinq_base_quant_config_fn

        method = cfg.method
        return sinq_base_quant_config_fn(
            nbits=int(cfg.nbits),
            group_size=int(cfg.group_size) if cfg.group_size is not None else None,
            quant_zero=False,
            quant_scale=False,
            view_as_float=False,
            axis=1,
            tiling_mode=str(cfg.tiling_mode),
            method=method,
        )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        """
        Called per-parameter to decide whether to run `SinqQuantize` on it.

        - If `self.pre_quantized`, we do *not* quantize again (handled by SinqDeserialize instead).
        - For method="asinq": return False (ASINQ is not supported in Hugging Face).
        - For method="sinq": True only for SINQLinear.weight not in modules_to_not_convert.

        Note: After _process_model_before_weight_loading(), the modules are already SINQLinear,
        not nn.Linear. We check for SINQLinear modules that are not yet quantized (ready=False).
        """
        from sinq.sinqlinear_hf import SINQLinear

        if self.pre_quantized:
            return False

        if self.quantization_config.method == "asinq":
            return False

        # SINQ param-level only if deemed safe
        if not self._do_param_level_sinq:
            return False

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name != "weight":
            return False
        module_name = param_name.rsplit(".", 1)[0] if param_name.endswith((".weight", ".bias")) else param_name
        if not should_convert_module(module_name, self.modules_to_not_convert):
            return False

        # Check if it's an unquantized SINQLinear
        is_sinq = isinstance(module, SINQLinear)
        is_ready = getattr(module, 'ready', True)
        result = is_sinq and not is_ready
        return result

    def get_quantize_ops(self):
        """
        Return the ConversionOps used for param-level quantization (Sinq).
        The actual SINQLinear construction is in integrations/sinq.py.
        """
        from ..integrations.sinq import SinqQuantize

        return SinqQuantize(self)

    def get_weight_conversions(self):
        """
        If `pre_quantized=True`, interpret a checkpoint produced by SINQLinear.state_dict:

            <prefix>.W_q
            <prefix>.bias
            <prefix>.meta

        via a WeightConverter + SinqDeserialize so that we reconstruct a SINQLinear
        module instead of a plain nn.Linear.
        """
        from ..core_model_loading import WeightConverter

        if self.pre_quantized:
            from ..integrations.sinq import SinqDeserialize

            return [
                WeightConverter(
                    source_patterns=[
                        ".W_q",
                        ".meta",
                        ".bias",
                    ],
                    target_patterns=[".weight"],
                    operations=[SinqDeserialize(self)],
                )
            ]
        return []
    
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs,
    ):
        """
        Called on meta-initialized model, before loading any weights.

        For SINQ, we replace nn.Linear modules with empty SINQLinear modules here.
        The actual quantization happens later in SinqQuantize.convert() when weights are loaded.
        """
        from sinq.sinqlinear_hf import SINQLinear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, (self.quantization_config.modules_to_not_convert or []), keep_in_fp32_modules
        )

        # Enable param-level quantization for SINQ method
        self._do_param_level_sinq = (
            self.quantization_config.method == "sinq" and not self.pre_quantized
        )

        if not self.pre_quantized and self.quantization_config.method == "asinq":
            raise ValueError("A-SINQ is not supported in HuggingFace integration")

        sinq_quant_dict = None if self.pre_quantized else self._build_sinq_quant_dict(self.quantization_config)
        device_str = _normalize_cuda_device(getattr(self.quantization_config, "device", "auto"))

        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not should_convert_module(full_name, self.modules_to_not_convert):
                continue

            parent_path, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model

            # Create empty SINQLinear (no weights yet)
            sinq_layer = SINQLinear(
                in_features=module.in_features if not self.pre_quantized else None,
                out_features=module.out_features if not self.pre_quantized else None,
                bias=(module.bias is not None) if not self.pre_quantized else False,
                quant_config=sinq_quant_dict,
                compute_dtype=self.dtype,
                device=device_str,
                use_unpack_kernel=True,
            )

            setattr(parent, child_name, sinq_layer)

    def _process_model_after_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        """
        Called after *all* weights have been loaded.

        - For method="sinq": nothing to do, param-level quantization already
          replaced Linear modules with SINQLinear during load.

        - For method="asinq": please use the official SINQ repository.
        """

        if self.quantization_config.method == "asinq" and not self.pre_quantized:
                raise ValueError(
                "You are using `method='asinq'` in the quantization config. Right now the calibrated version of SINQ"
                " is not supported in Hugging Face, please refer and use the official SINQ repository "
                "`to quantized a model with this method. "
            )

        return model

    def _resolve_tokenizer_and_model_id(self, model, kwargs):
        tok = kwargs.get("tokenizer", None)
        model_id = None
        cache_dir = kwargs.get("cache_dir", None)

        try:
            if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                model_id = model.config._name_or_path
            if model_id is None:
                model_id = kwargs.get("pretrained_model_name_or_path", None)
            if model_id is None and "config" in kwargs and hasattr(kwargs["config"], "_name_or_path"):
                model_id = getattr(kwargs["config"], "_name_or_path", None)

            logger.info(f"[SinqHfQuantizer] Detected model_id = {model_id}")

            if tok is None and model_id is not None:
                try:
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                    logger.info("[SinqHfQuantizer] Tokenizer loaded from model_id.")
                except Exception as e:
                    logger.warning(f"[SinqHfQuantizer] AutoTokenizer load failed: {e}")
        except Exception as outer_e:
            logger.warning(f"[SinqHfQuantizer] Tokenizer resolution failed: {outer_e}")

        if tok is not None and getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token = tok.eos_token

        return tok, model_id
