# coding=utf-8
# Copyright 2025 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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
from ..utils import is_torch_bf16_gpu_available, logging
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


def _validate_cuda_device_str(device_str: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SinqHfQuantizer requires CUDA, but no CUDA device is available.")
    if ":" in device_str:
        try:
            idx = int(device_str.split(":")[1])
        except Exception:
            raise ValueError(f"Invalid CUDA device string: {device_str!r}")
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device_str!r}, but visible CUDA devices are [0 .. {torch.cuda.device_count() - 1}]."
            )


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

def _is_param_level_safe_for_model(model) -> bool:
    """
    Heuristic: param-level replacement is safe only for text-only LMs that don't
    have custom init that reaches into q_proj.weight, etc.

    Conservative default: treat multimodal/vision models as unsafe.
    """
    cfg = getattr(model, "config", None)

    # Strong signals of multimodal / vision towers
    if cfg is not None:
        if hasattr(cfg, "vision_config") or hasattr(cfg, "image_size") or hasattr(cfg, "vision_hidden_size"):
            return False
        mt = getattr(cfg, "model_type", None)
        if isinstance(mt, str):
            mt = mt.lower()
            # Denylist: models known to have init touching projection weights
            if mt in {"gemma3", "siglip", "paligemma", "clip", "vit", "llava", "idefics", "idefics3"}:
                return False

    # Class-name heuristic (catches Gemma3ForConditionalGeneration, Siglip*, etc.)
    name = model.__class__.__name__.lower()
    if any(tok in name for tok in ["siglip", "gemma3", "paligemma", "clip", "vit", "vision", "conditionalgeneration"]):
        return False

    return True

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

    requires_calibration: bool = False  # A-SINQ does its own internal calibration
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
            
            # Validate that it's actually a torch dtype
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
        from sinq.sinqlinear import sinq_base_quant_config as sinq_base_quant_config_fn

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
        - For method="asinq": return False (ASINQ is module-level post-load).
        - For method="sinq": True only for nn.Linear.weight not in modules_to_not_convert.
        """

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
        return isinstance(module, nn.Linear)

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
        """
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, (self.quantization_config.modules_to_not_convert or []), keep_in_fp32_modules
        )
        
        # Allow user override via extra kwargs if you want (optional)
        force = getattr(self.quantization_config, "force_param_level", None)  # if you add it
        if force is True:
            self._do_param_level_sinq = True
        elif force is False:
            self._do_param_level_sinq = False
        else:
            # Hybrid default
            self._do_param_level_sinq = (
                (self.quantization_config.method == "sinq")
                and (not self.pre_quantized)
                and _is_param_level_safe_for_model(model)
            )

    def _process_model_after_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        """
        Called after *all* weights have been loaded.

        - For method="sinq": nothing to do, param-level quantization already
          replaced Linear modules with SINQLinear during load.

        - For method="asinq": run internal AWQ-like calibration and then replace
          Linear modules with SINQLinear, using layer activations.
        """

        if self.quantization_config.method == "asinq" and not self.pre_quantized:
            self._run_asinq_calibration_and_replace(model, **kwargs)

        if self.quantization_config.method == "sinq" and not self.pre_quantized and not self._do_param_level_sinq:
            self._replace_linear_with_sinqlinear(model)

        return model

    def _replace_linear_with_sinqlinear(self, model: "PreTrainedModel"):

        from sinq.sinqlinear import SINQLinear

        dtype = self.dtype
        device_str = self._normalized_device_str or _normalize_cuda_device(getattr(self.quantization_config, "device", "auto"))
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        compute_dtype = dtype
        sinq_quant_dict = self._build_sinq_quant_dict(self.quantization_config)

        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not should_convert_module(full_name, self.modules_to_not_convert):
                continue

            parent_path, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model

            module = module.to(device=device, dtype=compute_dtype)

            sinq_layer = SINQLinear(
                linear_layer=module,
                quant_config=sinq_quant_dict,
                del_orig=True,
                compute_dtype=compute_dtype,
                device=device_str,
                use_unpack_kernel=True,
                layer_activations=None,
            )
            setattr(parent, child_name, sinq_layer)
            sinq_layer._is_hf_initialized = True

    def _run_asinq_calibration_and_replace(self, model: "PreTrainedModel", **kwargs):
        """
        A-SINQ (activation-aware) pipeline:
        """
        from sinq.sinqlinear import SINQLinear
        from sinq.awq import (
            get_simple_calibration_data as awq_get_simple_calibration_data,
            get_calib_dataset as awq_get_calib_dataset,
            collect_activations_blockwise as awq_collect_activations_blockwise
        )

        device_str = self._normalized_device_str or _normalize_cuda_device(getattr(self.quantization_config, "device", "auto"))
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        tokenizer, model_id = self._resolve_tokenizer_and_model_id(model, kwargs)

        calib = None
        if tokenizer is not None:
            try:
                calib = awq_get_simple_calibration_data(tokenizer, block_size=128)
            except Exception as e:
                logger.warning(f"SINQ A-SINQ: get_simple_calibration_data failed: {e}")
                calib = None

        if (calib is None or len(calib) == 0) and tokenizer is not None:
            try:
                calib = awq_get_calib_dataset(tokenizer, n_samples=16, max_seq_len=512, col="text")
            except Exception as e:
                logger.warning(f"SINQ A-SINQ: get_calib_dataset failed: {e}")
                calib = None

        layer_acts: dict[str, torch.Tensor] = {}
        if calib is not None and len(calib) > 0:
            try:
                layer_acts = awq_collect_activations_blockwise(
                    model, calib, num_samples=len(calib), device=device_str
                )
            except Exception as e:
                logger.warning(f"SINQ A-SINQ: collect_activations failed; falling back to plain SINQ. {e}")
                layer_acts = {}
        else:
            logger.warning("SINQ A-SINQ: No calibration data produced; falling back to plain SINQ.")
            layer_acts = {}

        def _lookup_acts(name: str) -> Optional[torch.Tensor]:
            if not layer_acts:
                return None
            if name in layer_acts:
                return layer_acts[name]
            for k in layer_acts.keys():
                if k.endswith(name):
                    return layer_acts[k]
            return None

        sinq_quant_dict_base = self._build_sinq_quant_dict(self.quantization_config)
        compute_dtype = self.dtype

        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not should_convert_module(full_name, self.modules_to_not_convert):
                continue

            acts = _lookup_acts(full_name)
            quant_dict = sinq_quant_dict_base
            if acts is None:
                # fallback: treat as plain SINQ
                quant_dict = {
                    k: (v.copy() if isinstance(v, dict) else v)
                    for k, v in sinq_quant_dict_base.items()
                }
                quant_dict["weight_quant_params"]["method"] = "sinq"

            parent_path, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model

            module = module.to(device=device, dtype=compute_dtype)

            sinq_layer = SINQLinear(
                linear_layer=module,
                quant_config=quant_dict,
                del_orig=True,
                compute_dtype=compute_dtype,
                device=device_str,
                use_unpack_kernel=True,
                layer_activations=acts,
            )
            setattr(parent, child_name, sinq_layer)
            sinq_layer._is_hf_initialized = True

        logger.info("SinqHfQuantizer: A-SINQ calibration and replacement completed.")

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
