# src/transformers/quantizers/quantizer_sinq.py
from __future__ import annotations

import importlib
from typing import Optional, Union

import torch
import torch.nn as nn

from ..utils import is_torch_bf16_gpu_available
from ..utils.quantization_config import SinqConfig
from .base import HfQuantizer


try:
    from ..utils import logging

    logger = logging.get_logger(__name__)
except Exception:  # pragma: no cover
    logger = None


# top-level helpers in quantizer_sinq.py
def _normalize_cuda_device(dev: Optional[Union[str, int]]) -> str:
    """
    Returns a canonical device string like 'cuda', 'cuda:1', or 'cpu'.
    Behavior:
      - 'auto' or None -> 'cuda' if available, else 'cpu'
      - 'cpu' -> 'cpu'
      - 'cuda', 'cuda:0', 0, 1 -> canonicalized CUDA forms
    """
    if dev is None or dev == "auto":
        if torch.cuda.is_available():
            # pick the current device if it's set, otherwise 0
            idx = torch.cuda.current_device() if torch.cuda.device_count() else 0
            return f"cuda:{idx}"
        return "cpu"

    if dev == "cuda":
        return "cuda:0"  # explicit index
    
    # Explicit CPU
    if dev == "cpu":
        return "cpu"
    
    # Numeric device index
    if isinstance(dev, int):
        if torch.cuda.is_available():
            return f"cuda:{dev}"
        else:
            return "cpu"
    
    # String CUDA device
    if isinstance(dev, str):
        if dev.startswith("cuda"):
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
    # HF device_map can be nested; collect all device strings
    out = set()

    def _walk(v):
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, dict):
            for vv in v.values():
                _walk(vv)

    _walk(dmap)
    return out

# ------------------------------------------------------------------------------------
# SINQ dynamic import
# ------------------------------------------------------------------------------------
_SINQ_IMPORTED = False


def _import_sinq():
    """
    Import from the SINQ package (lazy).
    """
    global _SINQ_IMPORTED, SINQLinear, sinq_base_quant_config_fn
    global awq_get_simple_calibration_data, awq_get_calib_dataset, awq_collect_activations
    if _SINQ_IMPORTED:
        return
    if importlib.util.find_spec("sinq") is None:
        raise ImportError("The 'sinq' package is not installed. Please `pip install sinq` to use SinqConfig.")

    from sinq.awq import collect_activations as awq_collect_activations
    from sinq.awq import get_calib_dataset as awq_get_calib_dataset
    from sinq.awq import get_simple_calibration_data as awq_get_simple_calibration_data
    from sinq.sinqlinear import SINQLinear
    from sinq.sinqlinear import sinq_base_quant_config as sinq_base_quant_config_fn

    _SINQ_IMPORTED = True


# ------------------------------------------------------------------------------------
# Quantizer implementation
# ------------------------------------------------------------------------------------
class _SinqLoadTimeLinear(nn.Module):
    """
    Placeholder exposing weight/bias for state_dict loading and storing SINQ settings.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
        *,
        sinq_quant_dict: dict,
        compute_dtype: torch.dtype,
        device_str: str,  # = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device="meta"))
        self.bias = nn.Parameter(torch.empty(out_features, device="meta")) if use_bias else None

        self._sinq_quant_dict = sinq_quant_dict
        self._compute_dtype = compute_dtype
        self._device_str = device_str


class SinqHfQuantizer(HfQuantizer):
    """
    Hugging Face Quantizer integration for SINQ
    """

    requires_calibration: bool = False
    required_packages: list[str] = ["sinq"]
    requires_parameters_quantization: bool = False

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    # Check that the environment is ok
    def validate_environment(self, dtype=None, device_map=None, weights_only=None, **kwargs) -> None:
        _import_sinq()

        cfg: SinqConfig = self.quantization_config

        if getattr(cfg, "dtype", "auto") not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported dtype: {cfg.dtype}")
        if not torch.cuda.is_available():
            raise RuntimeError("SINQ currently expects a CUDA device.")

        device_str = _normalize_cuda_device(getattr(cfg, "device", "auto"))
        _validate_cuda_device_str(device_str)

        print(f'Device string is: {device_str}')
        
        self._normalized_device_str = device_str

        if device_str.startswith("cuda"):
            torch.cuda.set_device(torch.device(device_str))

        # Not supported: multi-GPU sharding via device_map
        devs = _flatten_device_map(device_map)
        if devs:
            # e.g., {'cuda:0'} is okay; more than one is not
            if len(devs) > 1:
                raise RuntimeError(
                    "SinqHfQuantizer: multi-GPU device_map detected, but SINQ currently supports only a single CUDA device. "
                    f"Got {sorted(devs)}. Please use device_map=None and set SinqConfig(device='{device_str}')."
                )
            # If they provided a single device_map device that doesn't match our config, error out to avoid surprise.
            only = next(iter(devs))
            if only != device_str and not (only == "cuda" and device_str == "cuda:0"):
                raise RuntimeError(
                    "SinqHfQuantizer: device_map device and SinqConfig.device disagree. "
                    f"device_map={only!r}, SinqConfig.device={device_str!r}. "
                    "Please pick one device and keep them consistent."
                )

    def update_dtype(self, dtype):
        cfg: SinqConfig = self.quantization_config
        if cfg.dtype == "auto":
            return torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
        return getattr(torch, cfg.dtype)

    def update_device_map(self, device_map):
        return device_map

    @staticmethod
    def _should_skip(name: str, modules_to_skip: set[str]) -> bool:
        """
        Check which layers from the original model should not be quantized
        """
        return any(name == m or name.endswith("." + m) for m in modules_to_skip)

    def _build_sinq_quant_dict(self, cfg: SinqConfig) -> dict:
        """
        Map HF config -> SINQ config dict.
          - method="sinq"  -> "sinq_quantAux"
          - method="asinq" -> "sinq_awq_l1_quantAux"
        """
        return sinq_base_quant_config_fn(
            nbits=int(cfg.nbits),
            group_size=int(cfg.group_size) if cfg.group_size is not None else None,
            quant_zero=False,
            quant_scale=False,
            view_as_float=False,
            axis=1,
            tiling_mode=str(cfg.tiling_mode),
            method=str(getattr(cfg, "method", "sinq")).lower(),
        )

    # ---------------- pre-load: install placeholders ----------------

    def _process_model_before_weight_loading(self, model: nn.Module, **kwargs) -> tuple[nn.Module, dict]:
        _import_sinq()

        cfg: SinqConfig = self.quantization_config

        sinq_quant_dict = self._build_sinq_quant_dict(cfg)
        compute_dtype = self.update_dtype(None)
        to_skip = set(cfg.modules_to_not_convert or [])
        device_str = getattr(self, "_normalized_device_str", _normalize_cuda_device(cfg.device))
        print(f'Device string in process model before_weights: {device_str}')

        def _convert(m: nn.Module, prefix: str = ""):
            for child_name, child in list(m.named_children()):
                full = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, nn.Linear) and not self._should_skip(full, to_skip):
                    ph = _SinqLoadTimeLinear(
                        child.in_features,
                        child.out_features,
                        use_bias=(child.bias is not None),
                        sinq_quant_dict=sinq_quant_dict,
                        compute_dtype=compute_dtype,
                        device_str=device_str,
                    )
                    setattr(m, child_name, ph)
                else:
                    _convert(child, full)

        _convert(model)
        return model, {}

    # --------------- post-load: swap to SINQ / A-SINQ ---------------

    def _process_model_after_weight_loading(
        self,
        model: nn.Module,
        loaded_in_4bit_or_8bit: bool = True,
        **kwargs,
    ) -> nn.Module:
        _import_sinq()

        cfg: SinqConfig = self.quantization_config
        method = str(getattr(cfg, "method", "sinq")).lower()
        device_str = getattr(self, "_normalized_device_str", _normalize_cuda_device(cfg.device))
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = model 

        print(f'Device string in process model after weights: {device_str}')

        placeholders: list[tuple[str, _SinqLoadTimeLinear, nn.Module]] = []

        def _gather(m: nn.Module, prefix: str = "", parent: Optional[nn.Module] = None):
            for child_name, child in list(m.named_children()):
                full = f"{prefix}.{child_name}" if prefix else child_name
                if isinstance(child, _SinqLoadTimeLinear):
                    placeholders.append((full, child, m))
                else:
                    _gather(child, full, m)

        _gather(model)

        def _resolve_tokenizer_and_model_id(model, kwargs, logger):
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

                print(f"[SinqHfQuantizer] Detected model_id = {model_id}", flush=True)
                if logger:
                    logger.info(f"Detected model_id = {model_id}")

                if tok is None and model_id is not None:
                    try:
                        from transformers import AutoTokenizer

                        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                        print("[SinqHfQuantizer] Tokenizer loaded from model_id.", flush=True)
                        if logger:
                            logger.info("Tokenizer loaded from model_id.")
                    except Exception as e:
                        print(f"[SinqHfQuantizer] AutoTokenizer load failed: {e}", flush=True)
                        if logger:
                            logger.warning(f"AutoTokenizer load failed: {e}")
            except Exception as outer_e:
                print(f"[SinqHfQuantizer] Tokenizer resolution failed: {outer_e}", flush=True)
                if logger:
                    logger.warning(f"Tokenizer resolution failed: {outer_e}")

            return tok, model_id

        tokenizer, _resolved_model_id = _resolve_tokenizer_and_model_id(model, kwargs, logger)

        # === A-SINQ branch: run internal calibration ===
        layer_acts: dict[str, torch.Tensor] = {}
        if method == "asinq":
            for full, ph, parent in placeholders:
                with torch.inference_mode():
                    dense = nn.Linear(
                        ph.in_features,
                        ph.out_features,
                        bias=(ph.bias is not None),
                        device=torch.device("meta"),
                        dtype=ph._compute_dtype,
                    )
                    w = ph.weight.detach().to(device, dtype=ph._compute_dtype, non_blocking=True)
                    dense.weight = nn.Parameter(w, requires_grad=False)
                    if ph.bias is not None:
                        b = ph.bias.detach().to(device, dtype=ph._compute_dtype, non_blocking=True)
                        dense.bias = nn.Parameter(b, requires_grad=False)
                setattr(parent, full.split(".")[-1], dense)

            model.to(dtype=torch.bfloat16, device=device)

            if (
                tokenizer is not None
                and getattr(tokenizer, "pad_token_id", None) is None
                and getattr(tokenizer, "eos_token_id", None) is not None
            ):
                tokenizer.pad_token = tokenizer.eos_token

            calib = None
            if tokenizer is not None:
                try:
                    calib = awq_get_simple_calibration_data(tokenizer, block_size=128)
                except Exception as e:
                    if logger:
                        logger.warning("get_simple_calibration_data failed: %s", e)
                    calib = None
            if (calib is None or len(calib) == 0) and tokenizer is not None:
                try:
                    calib = awq_get_calib_dataset(tokenizer, n_samples=16, max_seq_len=512, col="text")
                except Exception as e:
                    if logger:
                        logger.warning("get_calib_dataset failed: %s", e)
                    calib = None

            wrapped = model
            if tokenizer is not None:
                pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0

                class _WrapHF(torch.nn.Module):
                    def __init__(self, base, pad_id: int):
                        super().__init__()
                        self._base = base
                        self._pad = pad_id

                    def forward(self, x):
                        attn = (x != self._pad).to(x.device)
                        return self._base(input_ids=x, attention_mask=attn)

                    @property
                    def device(self):
                        try:
                            return next(self._base.parameters()).device
                        except StopIteration:
                            return device

                    def named_modules(self, *args, **kwargs):
                        return self._base.named_modules(*args, **kwargs)

                    def eval(self):
                        self._base.eval()
                        return self

                    def train(self, mode: bool = True):
                        self._base.train(mode)
                        return self

                wrapped = _WrapHF(model, pad_id)

                if calib is not None and len(calib) > 0:
                    try:
                        layer_acts = awq_collect_activations(wrapped, calib, num_samples=len(calib))
                    except Exception as e:
                        if logger:
                            logger.warning("collect_activations failed; falling back to SINQ. %s", e)
                        layer_acts = {}
                else:
                    if logger:
                        logger.warning("No calibration data produced; falling back to SINQ.")
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

        num_asinq = 0
        for full, ph, parent in placeholders:
            cur = getattr(parent, full.split(".")[-1])
            if isinstance(cur, nn.Linear):
                dense = cur
            else:
                with torch.no_grad():
                    dense = nn.Linear(ph.in_features, ph.out_features, bias=(ph.bias is not None), device=device)
                    try:
                        if ph.weight.device.type == "cpu":
                            ph.weight.data = ph.weight.data.pin_memory()
                        if ph.bias is not None and ph.bias.device.type == "cpu":
                            ph.bias.data = ph.bias.data.pin_memory()
                    except Exception:
                        pass
                    dense.weight.copy_(ph.weight.detach().to(device, non_blocking=True))
                    if ph.bias is not None:
                        dense.bias.copy_(ph.bias.detach().to(device, non_blocking=True))

            acts = _lookup_acts(full) if method == "asinq" else None

            quant_dict = ph._sinq_quant_dict
            if method == "asinq" and acts is None:
                quant_dict = {k: (v.copy() if isinstance(v, dict) else v) for k, v in quant_dict.items()}
                quant_dict["weight_quant_params"]["method"] = "sinq"
            else:
                num_asinq += 1

            sinq = SINQLinear(
                linear_layer=dense,
                quant_config=quant_dict,
                del_orig=True,
                compute_dtype=ph._compute_dtype,
                device=device_str,
                use_unpack_kernel=True,
                layer_activations=acts,
            )
            setattr(parent, full.split(".")[-1], sinq)

        final_dtype = self.update_dtype(None)

        if kwargs.get("device_map") is not None:
            raise RuntimeError(
                "SinqHfQuantizer: device_map was provided, but SINQ currently supports a single CUDA device only. "
                "Please remove device_map and set SinqConfig(device='cuda:x')."
            )

        final_dtype = self.update_dtype(None)
        try:
            model = model.to(dtype=final_dtype, device=device)
        except Exception as e:
            if logger:
                logger.warning(f"SINQ: final move to {device} ({final_dtype}) failed; continuing. {e}")

        if logger and method == "asinq":
            total = len(placeholders)
            if total > 0:
                logger.info(f"A-SINQ applied to {num_asinq}/{total} Linear layers ({100.0 * num_asinq / total:.1f}%).")

        return model