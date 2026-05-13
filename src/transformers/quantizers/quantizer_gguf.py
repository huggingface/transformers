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
"""Transparent 'quantizer' for GGUF checkpoints.

The only purpose of this class is to override ``spawn_materialize`` so that
``GGUFQuantizedTensor`` objects are dequantized during the spawn step, keeping
``isinstance(GGUFQuantizedTensor)`` checks out of the hot loading path in
``convert_and_load_state_dict_in_model``.
"""

from __future__ import annotations

from .base import HfQuantizer


class GGUFQuantizer(HfQuantizer):
    """Transparent quantizer for GGUF checkpoints.

    ``pre_quantized=False`` means no on-the-fly re-quantization is attempted.
    The only real work this class does is override ``spawn_materialize`` so
    that ``GGUFQuantizedTensor`` wrappers are dequantized transparently before
    entering the ``WeightConverter`` op chain.

    Note: ``isinstance(tensor, GGUFQuantizedTensor)`` is permitted inside
    ``spawn_materialize`` because that is the one semantically correct place
    for the check – the hot loading path remains clean.
    """

    requires_calibration = False

    def __init__(self, weight_mapping=None, **kwargs):
        # GGUFQuantizer has no QuantizationConfig; pass None to base class.
        # We also set pre_quantized=False to skip dtype branches meant for
        # already-quantized checkpoints.
        kwargs.setdefault("pre_quantized", False)
        super().__init__(quantization_config=None, **kwargs)
        self.weight_mapping = list(weight_mapping or [])

    def update_weight_conversions(self, weight_conversions):
        """Prepend the GGUF→HF rename table to the model's own conversions, and
        inject :class:`GGUFDequantize` as the first op of every
        :class:`WeightConverter` — mirrors how :class:`Fp8Quantizer` injects
        :class:`Fp8Dequantize`. ``WeightRenaming`` entries are passed through
        untouched (pure key renames need no op chain).
        """
        from ..core_model_loading import WeightConverter
        from ..gguf_conversion_ops import GGUFDequantize

        injected = []
        for conv in self.weight_mapping:
            if isinstance(conv, WeightConverter):
                conv = WeightConverter(
                    source_patterns=conv._original_source_patterns,
                    target_patterns=conv._original_target_patterns,
                    operations=[GGUFDequantize(), *conv.operations],
                )
            injected.append(conv)
        return injected + list(weight_conversions)

    def spawn_materialize(self, thread_pool, tensor, device=None, dtype=None):
        import torch

        from ..core_model_loading import GGUFQuantizedTensor, spawn_materialize

        if not isinstance(tensor, GGUFQuantizedTensor):
            return spawn_materialize(thread_pool, tensor, device, dtype)

        def _job():
            from ..integrations.gguf_dequant import dequantize_gguf_tensor

            compute_dtype = dtype if (dtype is not None and dtype.is_floating_point) else torch.float32
            # Move the small uint8 input to the target device first, then dequant on-device —
            # avoids transferring the much larger fp32 output across the CPU↔accelerator bus.
            w = dequantize_gguf_tensor(tensor.data, tensor.tensor_type, dtype=compute_dtype, device=device)
            if dtype is not None and dtype != compute_dtype:
                w = w.to(dtype=dtype)
            return w

        return thread_pool.submit(_job) if thread_pool is not None else _job

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
