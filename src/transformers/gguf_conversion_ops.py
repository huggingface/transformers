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

from .core_model_loading import ConversionOps, WeightConverter


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
    """Reverse Q-projection GGUF permutation. Reads ``config.num_attention_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        num_heads = config.num_attention_heads
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_heads // 2
        return {target_pattern: tensor.reshape(num_heads, dim, 2, *tensor.shape[1:]).swapaxes(2, 1).reshape(tensor.shape)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnQ is one-way")


class ReversePermuteAttnK(ConversionOps):
    """Reverse K-projection GGUF permutation. Reads ``config.num_attention_heads`` and
    ``config.num_key_value_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        num_kv_heads = config.num_key_value_heads
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_kv_heads // 2
        return {target_pattern: tensor.reshape(num_kv_heads, dim, 2, *tensor.shape[1:]).swapaxes(2, 1).reshape(tensor.shape)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnK is one-way")


class BloomReshapeQKVWeight(ConversionOps):
    """Reverse Bloom QKV weight reshape. Reads ``config.n_head`` and ``config.hidden_size`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
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
        input_dict: dict[str, "torch.Tensor"],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = target_patterns[0] if len(target_patterns) == 1 else source_patterns[0]
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


class GGUFDequantizer(WeightConverter):
    """``WeightConverter`` that always prepends ``GGUFDequantize`` as the first operation.

    Dequantization (converting raw GGUF bytes to ``torch.Tensor``) is handled
    by ``GGUFQuantizer.spawn_materialize`` before the op chain runs, so
    ``GGUFDequantize`` acts as a key-rename pass-through at that point.

    Compact form using ``*`` as a layer-index wildcard::

        GGUFDequantizer(
            "blk.*.attn_q.weight",
            "model.layers.*.self_attn.q_proj.weight",
            [ReversePermuteAttnQ()],
        )

    ``*`` in source patterns is converted to ``(\\d+)`` (capturing group).
    ``*`` in target patterns is converted to ``\\1`` (backreference).
    Non-layer tensors simply omit ``*``::

        GGUFDequantizer("token_embd.weight", "model.embed_tokens.weight")

    The explicit list form is also supported::

        GGUFDequantizer(
            source_patterns=["blk.0.attn_q.weight"],
            target_patterns=["model.layers.0.self_attn.q_proj.weight"],
            operations=[ReversePermuteAttnQ()],
        )
    """

    def __init__(self, source_patterns, target_patterns, operations=None):
        # Normalise to lists
        if isinstance(source_patterns, str):
            source_patterns = [source_patterns]
        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]

        # * in source → capturing group (\d+) for the layer index
        # * in target → backreference \1 substituted by rename_source_key
        source_patterns = [p.replace("*", r"(\d+)") for p in source_patterns]
        target_patterns = [p.replace("*", r"\1") for p in target_patterns]

        super().__init__(
            source_patterns=source_patterns,
            target_patterns=target_patterns,
            operations=[GGUFDequantize(), *(operations or [])],
        )
