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
import re
from typing import Any

from ..utils import logging


logger = logging.get_logger(__name__)


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


def maybe_quantize_on_init(model):
    """Quantize a freshly-constructed model in-place when `config.quantization_config` is set.

    Mirrors the on-the-fly quantization that `from_pretrained` performs (build the `HfQuantizer`, swap the
    modules via `preprocess_model`, route the weights through the quantization-aware loader, then
    `postprocess_model`), but for the construct-from-scratch path (`cls(config)` / `AutoModel.from_config`)
    where there is no checkpoint. The randomly-initialized weights are routed through the exact same loader
    used by `from_pretrained`, so the result is equivalent to loading those weights quantized.

    No-op unless `model` is the construction root (the outermost model, see `PreTrainedModel.__init__`) and
    holds real (non-meta) weights - `from_pretrained` builds on meta and quantizes itself, so that path is
    left untouched. Quantizers that need calibration data / a pre-quantized checkpoint (AWQ, GPTQ, ...) cannot
    quantize random weights, so the model is left in full precision with a warning.
    """
    import torch

    from ..conversion_mapping import get_model_conversion_mapping
    from ..core_model_loading import convert_and_load_state_dict_in_model
    from ..modeling_utils import LoadStateDictConfig
    from .auto import AutoHfQuantizer

    config = model.config
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None or getattr(config, "_quantization_init_root", None) != id(model):
        return
    try:
        # `from_pretrained` builds the skeleton on meta and quantizes during weight loading; bail out here so
        # we only handle the construct-from-scratch path.
        if any(param.device.type == "meta" for param in model.parameters()):
            return

        # Resolve the quantizer class without instantiating it, so we can check `requires_calibration` before
        # triggering any backend import or `__init__` side effect.
        quantization_config, quantizer_cls = AutoHfQuantizer.resolve(quantization_config)
        if quantizer_cls.requires_calibration:
            # Calibration-only methods (AWQ, GPTQ, ...) cannot quantize freshly-initialized weights. Warn instead
            # of silently returning a full-precision model - the user set a `quantization_config` and expects a
            # quantized model.
            logger.warning(
                f"A `quantization_config` ({quantization_config.quant_method}) was set on a model built from "
                "scratch (`from_config` / `cls(config)`), but this method requires a pre-quantized checkpoint or "
                "calibration data and cannot quantize randomly-initialized weights. The model was left in full "
                "precision and was NOT quantized; load a pre-quantized checkpoint with `from_pretrained` instead."
            )
            return

        hf_quantizer = quantizer_cls(quantization_config, pre_quantized=False)
        hf_quantizer.validate_environment(device_map=None, weights_only=True)

        # Read the dtype off the freshly-initialized parameters so it matches the captured `state_dict`, falling
        # back to the default float dtype for a (degenerate) param-less model. `cls(config)` builds in the global
        # default (fp32) unless `from_config` opened a dtype context, so this tracks the real weights either way.
        dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.get_default_dtype())
        # Capture the full-precision random weights *before* swapping modules - `preprocess_model` replaces the
        # `nn.Linear`/experts with fresh (empty) quantized modules, which would otherwise discard them.
        state_dict = model.state_dict()
        with torch.device("meta"):
            hf_quantizer.preprocess_model(
                model=model, dtype=dtype, device_map=None, checkpoint_files=None, use_kernels=False
            )

        load_config = LoadStateDictConfig(
            dtype=dtype,
            dtype_plan=model._get_dtype_plan(dtype),
            hf_quantizer=hf_quantizer,
            weight_mapping=get_model_conversion_mapping(model, None, hf_quantizer),
        )
        convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=None)

        model.hf_quantizer = hf_quantizer
        hf_quantizer.postprocess_model(model)
    finally:
        # Always release the marker so the next independent construction can re-claim the root, even if the
        # quantizer bailed out (no GPU, calibration-only, ...) or raised.
        if hasattr(config, "_quantization_init_root"):
            del config._quantization_init_root
