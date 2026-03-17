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
"""GGUF-specific ConversionOps for use with WeightConverter.

All ops in this file operate on already-dequantized ``torch.Tensor`` objects.
The actual dequantization from raw GGUF bytes is handled by
``spawn_gguf_materialize`` in ``core_model_loading``, which is invoked by
``GGUFQuantizer.spawn_materialize`` before any op chain runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core_model_loading import ConversionOps


if TYPE_CHECKING:
    import torch


class GGUFDequantize(ConversionOps):
    """
    First op in every GGUF WeightConverter chain.

    Since dequantization already happened inside ``spawn_gguf_materialize``
    (called by ``GGUFQuantizer.spawn_materialize``), this op is a pure
    key-renaming pass-through:

    - **1:1 case** (``len(input_dict)==1`` and ``len(target_patterns)==1``):
      renames the key to ``target_patterns[0]`` so that the WeightConverter
      prefix/suffix step can find it in the HF name.
    - **many:1 case** (e.g. Qwen2MoE gate+up): keeps source keys intact so a
      subsequent ``Concatenate`` op can iterate over them by name.

    No ``device``/``dtype`` kwargs – those are handled at spawn time.
    """

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        rename_to_target = len(input_dict) == 1 and len(target_patterns) == 1
        result = {}
        for key, tensors in input_dict.items():
            output_key = target_patterns[0] if rename_to_target else key
            result[output_key] = tensors
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("GGUFDequantize is one-way")


class Unsqueeze(ConversionOps):
    """Unsqueeze a tensor along ``dim``."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = self._get_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.unsqueeze(self.dim)}

    def _get_target(self, input_dict, source_patterns, target_patterns):
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Squeeze(self.dim)


class Squeeze(ConversionOps):
    """Squeeze a tensor along ``dim``. Inverse of :class:`Unsqueeze`."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = self._get_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.squeeze(self.dim)}

    def _get_target(self, input_dict, source_patterns, target_patterns):
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Unsqueeze(self.dim)


class SubtractOne(ConversionOps):
    """Subtract 1 from a tensor (used for GGUF norm weight de-offset)."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = self._get_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor - 1}

    def _get_target(self, input_dict, source_patterns, target_patterns):
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return AddOne()


class AddOne(ConversionOps):
    """Add 1 to a tensor. Inverse of :class:`SubtractOne`."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = self._get_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor + 1}

    def _get_target(self, input_dict, source_patterns, target_patterns):
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return SubtractOne()


class LogNegate(ConversionOps):
    """Apply ``log(-tensor)`` (used for GGUF Mamba SSM-A de-transform)."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        target_pattern = self._get_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: torch.log(-tensor)}

    def _get_target(self, input_dict, source_patterns, target_patterns):
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("LogNegate is not easily reversible")


class ReversePermuteAttnQ(ConversionOps):
    """Reverse Q-projection GGUF permutation. Needs ``config.num_attention_heads``."""

    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: self._reverse_permute(tensor, self.num_heads, self.num_heads)}

    def _reverse_permute(self, w: "torch.Tensor", n_head: int, num_kv_heads: int) -> "torch.Tensor":
        if n_head != num_kv_heads:
            n_head = num_kv_heads
        dim = w.shape[0] // n_head // 2
        return w.reshape(n_head, dim, 2, *w.shape[1:]).swapaxes(2, 1).reshape(w.shape)

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnQ is one-way")


class ReversePermuteAttnK(ConversionOps):
    """Reverse K-projection GGUF permutation. Needs ``num_attention_heads`` and ``num_key_value_heads``."""

    def __init__(self, num_heads: int, num_kv_heads: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: self._reverse_permute(tensor, self.num_heads, self.num_kv_heads)}

    def _reverse_permute(self, w: "torch.Tensor", n_head: int, num_kv_heads: int) -> "torch.Tensor":
        if n_head != num_kv_heads:
            n_head = num_kv_heads
        dim = w.shape[0] // n_head // 2
        return w.reshape(n_head, dim, 2, *w.shape[1:]).swapaxes(2, 1).reshape(w.shape)

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnK is one-way")


class BloomReshapeQKVWeight(ConversionOps):
    """Reverse Bloom QKV weight reshape."""

    def __init__(self, n_head: int, n_embed: int):
        self.n_head = n_head
        self.n_embed = n_embed

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        n_head, n_embed = self.n_head, self.n_embed
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
    """Reverse Bloom QKV bias reshape."""

    def __init__(self, n_head: int, n_embed: int):
        self.n_head = n_head
        self.n_embed = n_embed

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        n_head, n_embed = self.n_head, self.n_embed
        q, k, v = torch.chunk(w, 3)
        q = q.reshape(n_head, n_embed // n_head)
        k = k.reshape(n_head, n_embed // n_head)
        v = v.reshape(n_head, n_embed // n_head)
        return {target_pattern: torch.stack([q, k, v], dim=1).flatten()}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("BloomReshapeQKVBias is one-way")
