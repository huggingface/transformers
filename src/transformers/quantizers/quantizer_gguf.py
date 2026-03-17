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

    def __init__(self, **kwargs):
        # GGUFQuantizer has no QuantizationConfig; pass None to base class.
        # We also set pre_quantized=False to skip dtype branches meant for
        # already-quantized checkpoints.
        kwargs.setdefault("pre_quantized", False)
        super().__init__(quantization_config=None, **kwargs)

    def spawn_materialize(self, thread_pool, tensor, device=None, dtype=None):
        from ..core_model_loading import GGUFQuantizedTensor, spawn_gguf_materialize, spawn_materialize

        if isinstance(tensor, GGUFQuantizedTensor):
            return spawn_gguf_materialize(thread_pool, tensor, device, dtype)
        return spawn_materialize(thread_pool, tensor, device, dtype)

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
