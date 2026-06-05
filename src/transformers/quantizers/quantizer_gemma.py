# Copyright 2025 Google LLC
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

"""HfQuantizer implementation for pre-quantized Gemma checkpoints.

Handles loading of checkpoints that contain:
  - Packed integer weights (INT2/4/8) with per-channel scales
  - Static Range Quantization (SRQ) activation scales
  - Audio residual quantization (rqv2_muls)
  - Quantized embeddings
  - KV cache quantization scales
"""

from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..utils.quantization_config import GemmaQuantizationConfig


class GemmaQuantizer(HfQuantizer):
    """HfQuantizer for pre-quantized Gemma checkpoints.

    Replaces `nn.Linear` / `nn.Embedding` modules with their quantized
    counterparts during model loading, and loads quantized weights + SRQ
    scales directly from safetensors. Wrappers and unquantized layers are
    skipped via `quantization_config.modules_to_not_convert`.
    """

    quantization_config: "GemmaQuantizationConfig"
    requires_calibration = True

    def _process_model_before_weight_loading(self, model, **kwargs):
        from ..integrations.gemma_quant import replace_with_quant_layers

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        model = replace_with_quant_layers(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

        # KV-cache quantization scales live in the safetensors but the modeling
        # code doesn't load them (consumed by external tooling, e.g. a quantized
        # cache). Silence the unexpected-key warning.
        ignored = set(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
        ignored.update([r".*\.k_cache_scale$", r".*\.v_cache_scale$"])
        model._keys_to_ignore_on_load_unexpected = ignored  # type: ignore[unresolved-attribute]

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False

    @property
    def is_compileable(self) -> bool:
        return True
