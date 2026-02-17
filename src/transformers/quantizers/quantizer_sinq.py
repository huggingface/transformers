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
from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import is_torch_available, logging
from ..utils.quantization_config import SinqConfig
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


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

    requires_parameters_quantization: bool = True

    def __init__(self, quantization_config: SinqConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self._normalized_device_str: str | None = None
        self._do_param_level_sinq: bool = False

    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
            else:
                device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                f"Setting device_map to {device_map}. "
                "If you want to use the model for inference, please set device_map='auto'"
            )
        return device_map

    def update_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if dtype is None:
            dtype = torch.bfloat16
        self.dtype = dtype
        return dtype

    def validate_environment(self, *args, **kwargs) -> None:
        from ..utils import is_sinq_available

        if not is_sinq_available():
            raise ImportError("The 'sinq' package is not installed. Please install it with: pip install sinq")

        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA device is available. Quantization and inference will run on the CPU. Please note that this will significantly slow down inference speed and increase quantization time."
            )

        device_map = kwargs.get("device_map")

        if isinstance(device_map, dict):
            device_map_values = set(device_map.values())
            if len(device_map_values) > 1:
                raise RuntimeError(
                    "SinqHfQuantizer: multi-GPU device_map detected, but SINQ currently supports only a single CUDA "
                    f"device. Got {sorted(device_map_values)}. Please use device_map=None."
                )

        if self.quantization_config.method == "asinq" and not self.pre_quantized:
            raise ValueError(
                "You are using `method='asinq'` in the quantization config. Right now the calibrated version of SINQ"
                " is not supported in Hugging Face, please refer and use the official SINQ repository "
                "`to quantize a model with this method. "
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

    def param_needs_quantization(self, model: PreTrainedModel, param_name: str, **kwargs) -> bool:
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

        # Check if it's an unquantized SINQLinear
        is_sinq = isinstance(module, SINQLinear)
        is_ready = getattr(module, "ready", True)
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
        model: PreTrainedModel,
        device_map,
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs,
    ):
        """
        Called on meta-initialized model, before loading any weights.

        For SINQ, we replace nn.Linear modules with empty SINQLinear modules here.
        The actual quantization happens later in SinqQuantize.convert() when weights are loaded.
        """
        from ..integrations.sinq import replace_with_sinq_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, (self.quantization_config.modules_to_not_convert or []), keep_in_fp32_modules
        )

        # Enable param-level quantization for SINQ method
        self._do_param_level_sinq = self.quantization_config.method == "sinq" and not self.pre_quantized

        sinq_quant_dict = None if self.pre_quantized else self._build_sinq_quant_dict(self.quantization_config)

        # Extract device from device_map (guaranteed to be set by update_device_map)
        if isinstance(device_map, dict):
            first_device = next(iter(device_map.values()), 0)
            if isinstance(first_device, int):
                device_str = f"cuda:{first_device}"
            else:
                device_str = str(first_device)
        else:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = replace_with_sinq_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quant_config=sinq_quant_dict,
            compute_dtype=self.dtype,
            device=device_str,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(
        self,
        model: PreTrainedModel,
        **kwargs,
    ):
        """
        Called after *all* weights have been loaded.

        For SINQ:
        1. Move non-SINQLinear modules to GPU (embeddings, norms, lm_head, etc.)
           - SINQLinear modules already have GemLite buffers on GPU
           - We skip moving SINQLinear's W_q/meta to avoid memory duplication
        2. Patch HF save/load methods for SINQ serialization
        """
        from sinq.hf_io import patch_hf_pretrained_io

        # Patch HF save/load methods for SINQ serialization
        patch_hf_pretrained_io()

        return model
