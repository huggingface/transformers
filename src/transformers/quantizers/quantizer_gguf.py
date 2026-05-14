# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""GGUF quantizer.

The state-dict produced by ``load_gguf_checkpoint`` contains
:class:`GGUFQuantizedTensor` instances — ``torch.Tensor`` subclasses that
carry raw uint8 bytes plus the per-tensor ``quant_type`` metadata. They flow
through the standard loader unchanged (``.to(device)`` preserves the
subclass via ``__torch_function__``), and the :class:`GGUFDequantize` op
injected at the head of every ``WeightConverter`` produces the final
floating-point tensor — same pattern as :class:`Fp8Dequantize` for FP8.
"""

from __future__ import annotations

from .base import HfQuantizer


class GGUFQuantizer(HfQuantizer):
    """Quantizer for GGUF checkpoints — carries the rename table and injects ``GGUFDequantize``."""

    requires_calibration = False

    def __init__(self, weight_mapping=None, **kwargs):
        # ``pre_quantized=True`` so the loader keeps `_dtype=None` (no uint8→float
        # cast in ``spawn_materialize`` — the GGUFDequantize op handles dtype).
        kwargs.setdefault("pre_quantized", True)
        super().__init__(quantization_config=None, **kwargs)
        self.weight_mapping = list(weight_mapping or [])

    @property
    def renaming_dequantize_op(self):
        """Op the loader attaches to freshly-constructed ``WeightRenaming``s so the
        rename path produces dequantized fp32 (not raw GGUF bytes cast to fp32)."""
        from ..gguf_conversion_ops import GGUFDequantize

        return GGUFDequantize()

    def update_weight_conversions(self, weight_conversions):
        """Prepend the GGUF→HF rename table and inject ``GGUFDequantize`` at the head of every
        ``WeightConverter`` — same pattern as ``Fp8Quantizer.update_weight_conversions``.

        ``WeightRenaming`` entries also need dequant on the rename path: ``WeightRenaming.convert``
        returns the source ``GGUFQuantizedTensor`` directly under the renamed key, and
        ``set_param_for_module`` skips the shape check when a quantizer is active, so otherwise
        ``v_proj`` / ``o_proj`` / FFN projs / RMSNorms / embeddings would land in the model as
        uint8-bytes-cast-to-float32 with the wrong (byte-) shape. Attaching ``GGUFDequantize``
        as the ``quantization_operation`` on each renaming routes them through the dequant op
        (``WeightRenaming.convert`` runs ``self.quantization_operation`` when ``hf_quantizer`` is
        active), producing the correctly-expanded fp32 tensor.
        """
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..gguf_conversion_ops import GGUFDequantize

        injected = []
        for conv in self.weight_mapping:
            if isinstance(conv, WeightConverter):
                conv = WeightConverter(
                    source_patterns=conv._original_source_patterns,
                    target_patterns=conv._original_target_patterns,
                    operations=[GGUFDequantize(), *conv.operations],
                )
            elif isinstance(conv, WeightRenaming):
                conv.quantization_operation = GGUFDequantize()
            injected.append(conv)
        return injected + list(weight_conversions)

    def param_needs_quantization(self, model, param_name, **kwargs):
        # TODO: for on the fly quantization :)
        return False

    def preprocess_model(self, model, **kwargs):
        # No module swapping needed; skip the base class logic that tries to
        # set is_quantized / quantization_method on the model.
        pass

    def _process_model_before_weight_loading(self, model, **kwargs):
        pass

    def postprocess_model(self, model, **kwargs):
        # GGUF loading does not set any quantization config on the model.
        pass

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass

    @property
    def is_trainable(self):
        return True

    def is_serializable(self):
        return False
