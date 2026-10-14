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

"""Cache storage selection, independent of execution order and model parameters."""

import torch

from ..cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    STATIC_LAYER_TYPE_MAPPING,
    HQQQuantizedLayer,
    QuantizedLayer,
    QuantoQuantizedLayer,
)


class _QuantizedLifecycle(QuantizedLayer):
    # Native quantizers own encoded buffers as well as their floating-point residuals.
    def reset(self):
        super().reset()
        self.is_initialized = False
        self.cumulative_length = 0

    def batch_select_indices(self, indices):
        if not self.is_initialized:
            return
        self.reorder_cache(indices)
        self.batch_size = len(indices)

    def batch_repeat_interleave(self, repeats):
        if self.is_initialized:
            self.batch_select_indices(torch.arange(self.batch_size, device=self.device).repeat_interleave(repeats))

    def crop(self, tokens_to_remove):
        if not self.is_initialized or tokens_to_remove == 0:
            return
        length = (
            self.cumulative_length + tokens_to_remove
            if tokens_to_remove < 0
            else min(tokens_to_remove, self.cumulative_length)
        )
        keys, values = self._apply_pending_reorder(
            self._dequantize(self._quantized_keys), self._dequantize(self._quantized_values)
        )
        if self.keys is not None and self.keys.numel():
            keys, values = torch.cat((keys, self.keys), dim=-2), torch.cat((values, self.values), dim=-2)
        if length <= 0:
            self.reset()
            return
        self._quantized_keys = self._quantize(keys[..., :length, :], self.axis_key)
        self._quantized_values = self._quantize(values[..., :length, :], self.axis_value)
        self.keys, self.values = keys[..., :0, :], values[..., :0, :]
        self._pending_beam_idx = None
        self.cumulative_length = length


class _QuantoLayer(_QuantizedLifecycle, QuantoQuantizedLayer):
    pass


class _HQQLayer(_QuantizedLifecycle, HQQQuantizedLayer):
    pass


class _TorchQuantizedLayer(_QuantizedLifecycle):
    """Portable packed 2/4/8-bit per-channel group quantization without an optional native extension."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.nbits not in (2, 4, 8) or self.q_group_size < 1 or self.q_group_size % (8 // self.nbits):
            raise ValueError("torch quantization needs 2/4/8 bits and a positive group size divisible by 8/nbits.")

    def _quantize(self, tensor, axis):
        values = tensor.movedim(axis, 0).contiguous()
        shape = values.shape
        matrix = values.flatten(start_dim=1).float()
        padded = torch.nn.functional.pad(matrix, (0, (-matrix.shape[-1]) % self.q_group_size))
        groups = padded.view(shape[0], -1, self.q_group_size)
        zero = groups.amin(dim=-1, keepdim=True)
        scale = ((groups.amax(dim=-1, keepdim=True) - zero) / ((1 << self.nbits) - 1)).clamp_min(1e-8)
        codes = ((groups - zero) / scale).round().clamp(0, (1 << self.nbits) - 1).to(torch.uint8)
        factor = 8 // self.nbits
        packed = codes[..., ::factor].clone()
        for index in range(1, factor):
            packed |= codes[..., index::factor] << (self.nbits * index)
        return packed, scale, zero, tuple(shape), tensor.dtype, axis

    def _dequantize(self, encoded):
        packed, scale, zero, shape, dtype, axis = encoded
        factor = 8 // self.nbits
        codes = torch.stack(
            [(packed >> (self.nbits * index)) & ((1 << self.nbits) - 1) for index in range(factor)], dim=-1
        )
        values = (codes.flatten(start_dim=-2).float() * scale + zero).flatten(start_dim=1)
        elements = 1
        for dimension in shape[1:]:
            elements *= dimension
        return values[:, :elements].reshape(shape).to(dtype).movedim(0, axis)


def create_cache_layer(layer_type, layer_kwargs, implementation, max_cache_len, cache_config):
    """Reuse native cache layers; quantized hybrid decoders keep their non-KV states at native precision."""
    if implementation in ("static", "offloaded_static"):
        if type(max_cache_len) is not int or max_cache_len < 1:
            raise ValueError("Static layer execution caches require a positive max_cache_len.")
        return STATIC_LAYER_TYPE_MAPPING[layer_type](max_cache_len=max_cache_len, **layer_kwargs)
    if implementation == "quantized" and layer_type == "full_attention":
        options = dict(cache_config)
        backend = options.pop("backend", "quanto")
        quantizer = {"quanto": _QuantoLayer, "hqq": _HQQLayer, "torch": _TorchQuantizedLayer}.get(backend)
        if quantizer is None:
            raise ValueError(f"Unknown quantization backend {backend!r}.")
        return quantizer(**options)
    # Sliding and recurrent states retain their native storage, rather than accepting unsupported quantizers.
    return DYNAMIC_LAYER_TYPE_MAPPING[layer_type](**layer_kwargs)
