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
"""GGUF-specific :class:`ConversionOps` for use with :class:`WeightConverter`.

The ``GGUFDequantize`` op runs first in every GGUF ``WeightConverter`` chain
and turns each :class:`GGUFQuantizedTensor` input (raw uint8 bytes carrying
``quant_type`` metadata) into a regular ``torch.Tensor`` — same role as
``Fp8Dequantize`` in the FP8 quantizer's chain.

The remaining ops here (``Unsqueeze``/``SubtractOne``/``LogNegate``/permute/
reshape) operate on already-dequantized ``torch.Tensor`` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core_model_loading import ConversionOps


if TYPE_CHECKING:
    import torch


def _single_input_target(input_dict, source_patterns, target_patterns):
    """Return the output key for a single-input conversion op."""
    if len(input_dict) != 1:
        raise ValueError("Undefined Operation encountered!")
    if len(target_patterns) > 1:
        if len(source_patterns) == 1:
            return source_patterns[0]
        raise ValueError("Undefined Operation encountered!")
    target = target_patterns[0]
    if r"\1" in target:
        return next(iter(input_dict))
    return target


class GGUFDequantize(ConversionOps):
    """First op in every GGUF ``WeightConverter`` chain.

    Reads ``quant_type`` from each input :class:`GGUFQuantizedTensor` and
    dequantizes the raw uint8 bytes to a floating-point ``torch.Tensor`` using
    the pure-torch kernels in ``integrations/gguf_dequant.py`` (city96-style,
    same kernels diffusers uses). The dequant runs on whatever device the
    input tensor is already on, so the loader's ``.to(device)`` upstream means
    MPS / CUDA dequant happens on-device.
    """

    def convert(
        self,
        input_dict,
        source_patterns,
        target_patterns,
        **kwargs,
    ):
        from .integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor

        out = {}
        for key, tensors in input_dict.items():
            tensors_list = tensors if isinstance(tensors, list) else [tensors]
            dequantized = [
                dequantize_gguf_tensor(t, t.quant_type, device=t.device) if isinstance(t, GGUFQuantizedTensor) else t
                for t in tensors_list
            ]
            out[key] = dequantized if isinstance(tensors, list) else dequantized[0]
        return out

    @property
    def reverse_op(self):
        raise NotImplementedError("GGUFDequantize is one-way")


class Unsqueeze(ConversionOps):
    """Unsqueeze a tensor along ``dim``."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.unsqueeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Squeeze(self.dim)


class Squeeze(ConversionOps):
    """Squeeze a tensor along ``dim``. Inverse of :class:`Unsqueeze`."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.squeeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Unsqueeze(self.dim)


class SubtractOne(ConversionOps):
    """Subtract 1 from a tensor (used for GGUF norm weight de-offset)."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor - 1}

    @property
    def reverse_op(self) -> ConversionOps:
        return AddOne()


class AddOne(ConversionOps):
    """Add 1 to a tensor. Inverse of :class:`SubtractOne`."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor + 1}

    @property
    def reverse_op(self) -> ConversionOps:
        return SubtractOne()


class LogNegate(ConversionOps):
    """Apply ``log(-tensor)`` (used for GGUF Mamba SSM-A de-transform)."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: torch.log(-tensor)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("LogNegate is not easily reversible")


class ReversePermuteAttnQ(ConversionOps):
    """Reverse Q-projection GGUF permutation. Reads ``config.num_attention_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_heads = config.num_attention_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_heads // 2
        return {
            target_pattern: tensor.reshape(num_heads, dim, 2, *tensor.shape[1:]).swapaxes(2, 1).reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteAttnQ()


class PermuteAttnQ(ConversionOps):
    """Forward Q-projection GGUF permutation — inverse of :class:`ReversePermuteAttnQ`.
    Used by :func:`save_pretrained_gguf` when re-emitting Q weights into llama.cpp's
    expected layout. Reads ``config.num_attention_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_heads = config.num_attention_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_heads // 2
        return {
            target_pattern: tensor.reshape(num_heads, 2, dim, *tensor.shape[1:]).swapaxes(2, 1).reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return ReversePermuteAttnQ()


class ReversePermuteAttnK(ConversionOps):
    """Reverse K-projection GGUF permutation. Reads ``config.num_attention_heads`` and
    ``config.num_key_value_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_kv_heads = config.num_key_value_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_kv_heads // 2
        return {
            target_pattern: tensor.reshape(num_kv_heads, dim, 2, *tensor.shape[1:])
            .swapaxes(2, 1)
            .reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteAttnK()


class PermuteAttnK(ConversionOps):
    """Forward K-projection GGUF permutation — inverse of :class:`ReversePermuteAttnK`."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_kv_heads = config.num_key_value_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_kv_heads // 2
        return {
            target_pattern: tensor.reshape(num_kv_heads, 2, dim, *tensor.shape[1:])
            .swapaxes(2, 1)
            .reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return ReversePermuteAttnK()


class BloomReshapeQKVWeight(ConversionOps):
    """Reverse Bloom QKV weight reshape. Reads ``config.n_head`` and ``config.hidden_size`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        q, k, v = torch.chunk(w, 3, dim=0)
        q = q.reshape(n_head, n_embed // n_head, n_embed)
        k = k.reshape(n_head, n_embed // n_head, n_embed)
        v = v.reshape(n_head, n_embed // n_head, n_embed)
        qkv = torch.stack([q, k, v], dim=1)
        return {target_pattern: qkv.reshape(n_head * 3 * (n_embed // n_head), n_embed)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("BloomReshapeQKVWeight is one-way")


class BloomReshapeQKVBias(ConversionOps):
    """Reverse Bloom QKV bias reshape. Reads ``config.n_head`` and ``config.hidden_size`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        q, k, v = torch.chunk(w, 3)
        q = q.reshape(n_head, n_embed // n_head)
        k = k.reshape(n_head, n_embed // n_head)
        v = v.reshape(n_head, n_embed // n_head)
        return {target_pattern: torch.stack([q, k, v], dim=1).flatten()}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("BloomReshapeQKVBias is one-way")
