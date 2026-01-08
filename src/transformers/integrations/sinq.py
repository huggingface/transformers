# src/transformers/integrations/sinq.py
from __future__ import annotations

from typing import Optional, Dict, Any

from transformers.utils import is_torch_available, logging

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn as nn


class SinqQuantize(ConversionOps):
    """
    Param-level ConversionOp for SINQ (from FP weights).

    At load time, for each `Linear.weight` that should be quantized:
      - we take the loaded weight tensor,
      - we build a temporary dense nn.Linear (weight-only),
      - we wrap it into SINQLinear,
      - we replace the original module inside the model.

    This is structurally similar to TorchAoQuantize / Fp8Quantize, but
    we delegate SINQ-specific details here instead of in the HfQuantizer.
    """

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def _get_runtime_device_str(self) -> str:
        cfg = self.hf_quantizer.quantization_config
        device_str = getattr(self.hf_quantizer, "_normalized_device_str", None)
        if device_str is not None:
            return device_str
        if getattr(cfg, "device", None) in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return str(cfg.device)

    def convert(
        self,
        input_dict: Dict[str, Any],
        model: Optional["torch.nn.Module"] = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        if model is None or full_layer_name is None:
            logger.warning_once(
                "SinqQuantize.convert called without `model` or `full_layer_name`; "
                "skipping SINQ quantization for this parameter."
            )
            key, values = next(iter(input_dict.items()))
            val = values[0] if isinstance(values, list) else values
            return {full_layer_name or key: val}

        _, values = next(iter(input_dict.items()))
        weight_tensor = values[0] if isinstance(values, list) else values

        module, tensor_name = get_module_from_name(model, full_layer_name)
        if tensor_name != "weight":
            logger.warning_once(
                f"SinqQuantize.convert called for non-weight parameter: {full_layer_name}; "
                "treating as normal param."
            )
            return {full_layer_name: weight_tensor}

        module_path, _, _ = full_layer_name.rpartition(".") 
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        from sinq.sinqlinear import SINQLinear
        from sinq.sinqlinear import sinq_base_quant_config as sinq_base_quant_config_fn

        device_str = self._get_runtime_device_str()
        device = torch.device(device_str)
        compute_dtype = self.hf_quantizer.update_torch_dtype(None)

        in_features = weight_tensor.shape[1]
        out_features = weight_tensor.shape[0]

        dense = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=weight_tensor.dtype,
        )
        with torch.no_grad():
            dense.weight.copy_(weight_tensor.to(device=device, dtype=weight_tensor.dtype))

        cfg = self.hf_quantizer.quantization_config
        sinq_quant_dict = self.hf_quantizer._build_sinq_quant_dict(cfg)

        sinq_layer = SINQLinear(
            linear_layer=dense,
            quant_config=sinq_quant_dict,
            del_orig=True,
            compute_dtype=compute_dtype,
            device=device_str,
            use_unpack_kernel=True,
            layer_activations=None,
        )

        setattr(parent, child_name, sinq_layer)

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        sinq_layer._is_hf_initialized = True

        return {}


class SinqDeserialize(ConversionOps):
    """
    ConversionOp for loading *pre-quantized* SINQ checkpoints.

    Checkpoint layout (what `SINQLinear.state_dict` produces) is, per module:
        <prefix>.W_q
        <prefix>.bias
        <prefix>.meta

    WeightConverter in the quantizer is configured so that:
      - we group ".W_q", ".meta", ".bias" as input_dict
      - conceptually treat them as belonging to "<prefix>.weight"
      - and call this SinqDeserialize.convert to reconstruct a SINQLinear.

    The returned dict is {} because we perform module replacement directly.
    """

    def __init__(self, hf_quantizer: "SinqHfQuantizer"):
        self.hf_quantizer = hf_quantizer

    def _get_runtime_device_str(self) -> str:
        cfg = self.hf_quantizer.quantization_config
        device_str = getattr(self.hf_quantizer, "_normalized_device_str", None)
        if device_str is not None:
            return device_str
        if getattr(cfg, "device", None) in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return str(cfg.device)

    def convert(
        self,
        input_dict: Dict[str, Any],
        model: Optional["torch.nn.Module"] = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        from sinq.sinqlinear import SINQLinear 

        if model is None or full_layer_name is None:
            logger.warning_once(
                "SinqDeserialize.convert called without `model` or `full_layer_name`; "
                "returning raw tensors (no SINQ module reconstruction)."
            )
            out: Dict[str, "torch.Tensor"] = {}
            for k, v in input_dict.items():
                out[k] = v[0] if isinstance(v, list) else v
            return out

        for k, v in list(input_dict.items()):
            if isinstance(v, list):
                input_dict[k] = v[0]

        W_q = input_dict.get(".W_q", None)
        meta = input_dict.get(".meta", None)
        bias = input_dict.get(".bias", None)

        if W_q is None or meta is None:
            logger.warning_once(
                "SinqDeserialize.convert did not find '.W_q' and '.meta' entries; "
                "treating as normal (non-SINQ) parameter."
            )
            v = next(iter(input_dict.values()))
            if isinstance(v, list):
                v = v[0]
            return {full_layer_name: v}

        device_str = self._get_runtime_device_str()
        device = torch.device(device_str)
        compute_dtype = self.hf_quantizer.update_torch_dtype(None)

        module, tensor_name = get_module_from_name(model, full_layer_name)
        if tensor_name != "weight":
            logger.warning_once(
                f"SinqDeserialize called for non-weight parameter: {full_layer_name}. "
                "Skipping SINQ reconstruction."
            )
            return {full_layer_name: W_q}

        module_path, _, _ = full_layer_name.rpartition(".") 
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        sinq_layer = SINQLinear(
            linear_layer=None,
            quant_config=None,
            del_orig=True,
            compute_dtype=compute_dtype,
            device=device_str,
            use_unpack_kernel=True,
            layer_activations=None,
        )

        state = {
            "W_q": W_q.to(device),
            "meta": meta,
        }
        if bias is not None:
            state["bias"] = bias.to(device)

        sinq_layer.load_state_dict(state)

        setattr(parent, child_name, sinq_layer)
        sinq_layer._is_hf_initialized = True

        return {}