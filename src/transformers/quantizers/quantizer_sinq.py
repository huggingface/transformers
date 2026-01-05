# src/transformers/quantizers/quantizer_sinq.py
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, List

import torch
import torch.nn as nn

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_torch_bf16_gpu_available, logging
from ..utils.quantization_config import SinqConfig

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

_SINQ_IMPORTED = False


def _import_sinq():
    """
    Lazy import of SINQ components.
    We only need SINQLinear for A-SINQ post-processing and for building
    the quantization config dict.
    """
    global _SINQ_IMPORTED, SINQLinear, sinq_base_quant_config_fn
    global awq_get_simple_calibration_data, awq_get_calib_dataset, awq_collect_activations, awq_collect_activations_blockwise

    if _SINQ_IMPORTED:
        return

    if importlib.util.find_spec("sinq") is None:
        raise ImportError("The 'sinq' package is not installed. Please `pip install sinq` to use SinqConfig.")

    from sinq.sinqlinear import SINQLinear, sinq_base_quant_config as sinq_base_quant_config_fn
    from sinq.awq import (
        get_simple_calibration_data as awq_get_simple_calibration_data,
        get_calib_dataset as awq_get_calib_dataset,
        collect_activations as awq_collect_activations,
        collect_activations_blockwise as awq_collect_activations_blockwise
    )

    _SINQ_IMPORTED = True

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
    required_packages: List[str] = ["sinq"]
    requires_parameters_quantization: bool = True

    def __init__(self, quantization_config: SinqConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self._normalized_device_str: str | None = None
        self.modules_to_not_convert: list[str] = []
        self._do_param_level_sinq: bool = False

    def is_serializable(self, safe_serialization: Optional[bool] = None) -> bool:
        """
        SINQLinear.meta is a nested dict with non-tensor entries, which is not directly
        compatible with safetensors flattening without extra work.

            - safe_serialization=True  -> return False and warn
            - safe_serialization=False -> ok
        """
        if safe_serialization:
            logger.warning(
                "SINQ quantized models do not currently support `safe_serialization=True` "
                "(safetensors) because SINQLinear.meta is a nested dict with non-tensor "
                "entries. Please save with `safe_serialization=False` for now."
            )
            return False
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def validate_environment(self, *args, **kwargs) -> None:
        _import_sinq()

        cfg: SinqConfig = self.quantization_config

        if getattr(cfg, "dtype", "auto") not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported dtype in SinqConfig: {cfg.dtype}")

        if not torch.cuda.is_available():
            raise RuntimeError("SINQ currently expects a CUDA device (for GemLite backend).")

        device_map = kwargs.get("device_map", None)
        device_str = _normalize_cuda_device(getattr(cfg, "device", "auto"))
        _validate_cuda_device_str(device_str)
        self._normalized_device_str = device_str

        if device_str.startswith("cuda"):
            torch.cuda.set_device(torch.device(device_str))

        devs = _flatten_device_map(device_map)
        if devs:
            if len(devs) > 1:
                raise RuntimeError(
                    "SinqHfQuantizer: multi-GPU device_map detected, but SINQ currently supports only a single CUDA "
                    f"device. Got {sorted(devs)}. Please use device_map=None and set SinqConfig(device='{device_str}')."
                )
            only = next(iter(devs))
            if only != device_str and not (only == "cuda" and device_str == "cuda:0"):
                raise RuntimeError(
                    "SinqHfQuantizer: device_map device and SinqConfig.device disagree. "
                    f"device_map={only!r}, SinqConfig.device={device_str!r}. "
                    "Please pick one device and keep them consistent."
                )

    def update_torch_dtype(self, torch_dtype: Optional[torch.dtype]) -> torch.dtype:
        """
        Decide compute dtype for quantized layers.
        """
        cfg: SinqConfig = self.quantization_config
        if cfg.dtype == "auto":
            return torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
        return getattr(torch, cfg.dtype)

    # keep old name for BC
    def update_dtype(self, dtype: Optional[torch.dtype]) -> torch.dtype:
        return self.update_torch_dtype(dtype)

    def update_device_map(self, device_map):
        # We already enforced single CUDA device in validate_environment;
        # nothing else to do, let HF / accelerate handle placement.
        return device_map

    def _build_sinq_quant_dict(self, cfg: SinqConfig) -> dict:
        """
        Build the dict that SINQLinear expects as quant_config.
        """
        _import_sinq()
        method = str(getattr(cfg, "method", "sinq")).lower()
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

    def _should_skip(self, name: str) -> bool:
        """
        `name` can be a module path ("model.layers.0.self_attn.q_proj")
        or a param path ("model.layers.0.self_attn.q_proj.weight").
        We always compare against the *module* part.
        """
        if name.endswith(".weight") or name.endswith(".bias"):
            name = name.rsplit(".", 1)[0]

        return any(name == m or name.startswith(m + ".") for m in self.modules_to_not_convert)

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        """
        Called per-parameter to decide whether to run `SinqQuantize` on it.

        - If `self.pre_quantized`, we do *not* quantize again (handled by SinqDeserialize instead).
        - For method="asinq": return False (ASINQ is module-level post-load).
        - For method="sinq": True only for nn.Linear.weight not in modules_to_not_convert.
        """
        cfg: SinqConfig = self.quantization_config
        method = str(getattr(cfg, "method", "sinq")).lower()

        if self.pre_quantized:
            return False

        if method == "asinq":
            return False
        
        # SINQ param-level only if deemed safe
        if not self._do_param_level_sinq:
            return False

        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name != "weight":
            return False
        if self._should_skip(param_name):
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
                    target_patterns=[".weight", ".weight", ".weight"],
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
        cfg: SinqConfig = self.quantization_config
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, cfg.modules_to_not_convert, keep_in_fp32_modules
        )

        method = str(getattr(cfg, "method", "sinq")).lower()

        # Allow user override via extra kwargs if you want (optional)
        force = getattr(cfg, "force_param_level", None)  # if you add it
        if force is True:
            self._do_param_level_sinq = True
        elif force is False:
            self._do_param_level_sinq = False
        else:
            # Hybrid default
            self._do_param_level_sinq = (
                (method == "sinq")
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
        cfg: SinqConfig = self.quantization_config
        method = str(getattr(cfg, "method", "sinq")).lower()

        if method == "asinq" and not self.pre_quantized:
            self._run_asinq_calibration_and_replace(model, **kwargs)

        if method == "sinq" and not self.pre_quantized and not self._do_param_level_sinq:
            self._replace_linear_with_sinqlinear(model)
        
        try:
            setattr(model, "_is_sinq_quantized", True)
        except Exception:
            pass
        return model

    def _replace_linear_with_sinqlinear(self, model: "PreTrainedModel"):
        _import_sinq()
        cfg: SinqConfig = self.quantization_config

        device_str = self._normalized_device_str or _normalize_cuda_device(getattr(cfg, "device", "auto"))
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        compute_dtype = self.update_torch_dtype(None)
        sinq_quant_dict = self._build_sinq_quant_dict(cfg)

        from sinq.sinqlinear import SINQLinear

        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if self._should_skip(full_name):
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
        _import_sinq()
        cfg: SinqConfig = self.quantization_config
        device_str = self._normalized_device_str or _normalize_cuda_device(getattr(cfg, "device", "auto"))
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

        # # 3) Wrap model to define HF-like forward & attention mask
        # if tokenizer is not None:
        #     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
        # else:
        #     pad_id = 0

        # class _WrapHF(nn.Module):
        #     def __init__(self, base, pad_id: int):
        #         super().__init__()
        #         self._base = base
        #         self._pad = pad_id
        #         # Expose a .device attribute for SINQ AWQ helpers
        #         try:
        #             self.device = next(base.parameters()).device
        #         except StopIteration:
        #             self.device = torch.device("cpu")

        #     def forward(self, x):
        #         attn = (x != self._pad).to(x.device)
        #         return self._base(input_ids=x, attention_mask=attn)

        #     def named_modules(self, *args, **kwargs):
        #         return self._base.named_modules(*args, **kwargs)

        #     def eval(self):
        #         self._base.eval()
        #         return self

        #     def to(self, *args, **kwargs):
        #         super().to(*args, **kwargs)
        #         try:
        #             self.device = next(self._base.parameters()).device
        #         except StopIteration:
        #             self.device = torch.device("cpu")
        #         return self

        # #wrapped = _WrapHF(model.to(device=device), pad_id)
        # # keep model on CPU for this step
        # wrapped = _WrapHF(model, pad_id).eval()

        layer_acts: dict[str, torch.Tensor] = {}
        if calib is not None and len(calib) > 0:
            try:
                #layer_acts = awq_collect_activations(wrapped, calib, num_samples=len(calib))
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

        sinq_quant_dict_base = self._build_sinq_quant_dict(cfg)
        compute_dtype = self.update_torch_dtype(None)

        for full_name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if self._should_skip(full_name):
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

            from sinq.sinqlinear import SINQLinear

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