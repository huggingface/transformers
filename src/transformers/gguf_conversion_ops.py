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
    """
    Reads `quant_type` from each input :class:`GGUFQuantizedTensor` and
    dequantizes the raw uint8 bytes to a floating-point `torch.Tensor` using
    the pure-torch kernels in `integrations/gguf_dequant.py`
    """

    def __init__(self, hf_quantizer=None):
        # Quantizer reference is only needed for the reverse op (re-quantize on save).
        self.hf_quantizer = hf_quantizer

    def __deepcopy__(self, memo):
        # `WeightConverter` deep-copies its op list during loading. The default
        # deepcopy would walk `self.hf_quantizer` → `gguf_tensors` (which
        # contains :class:`GGUFQuantizedTensor` subclasses that lack `__deepcopy__`).
        # We don't need a deep copy of the quantizer anyway — share the reference.
        return GGUFDequantize(self.hf_quantizer)

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        from .integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor

        out = {}
        for key, tensors in input_dict.items():
            tensors_list = tensors if isinstance(tensors, list) else [tensors]
            processed = [
                dequantize_gguf_tensor(t, t.quant_type, device=t.device) if isinstance(t, GGUFQuantizedTensor) else t
                for t in tensors_list
            ]
            out[key] = processed if isinstance(tensors, list) else processed[0]
        return out

    @property
    def reverse_op(self):
        return GGUFQuantize(self.hf_quantizer)


class GGUFQuantize(ConversionOps):
    """Reverse op of :class:`GGUFDequantize`. Re-packs a floating-point tensor
    into GGUF block bytes via `gguf.quants.quantize`. Used on the save path
    to round-trip a dequantized / on-the-fly-quantized model back to a `.gguf`
    file. Mirrors :class:`~transformers.integrations.finegrained_fp8.Fp8Quantize`.

    Today gguf-py only ships a Python quantizer for `Q4_0` and `Q8_0`; other
    quant types (K-quants, IQ4) are read-only upstream, so attempting to use
    them here raises with a clear message.
    """

    def __init__(self, hf_quantizer=None):
        self.hf_quantizer = hf_quantizer

    def __deepcopy__(self, memo):
        return GGUFQuantize(self.hf_quantizer)

    # TODO support all quantization types.
    _GGUF_PY_QUANTIZE_SUPPORTED = ("Q4_0", "Q8_0")

    def _quantize_one(self, key: str, value):
        import gguf
        import torch

        quant_type = getattr(getattr(self.hf_quantizer, "quantization_config", None), "quant_type", None) or "Q4_0"
        if quant_type not in self._GGUF_PY_QUANTIZE_SUPPORTED:
            raise ValueError(
                f"On-the-fly GGUF quantize only supports {self._GGUF_PY_QUANTIZE_SUPPORTED} (gguf-py limit); "
                f"got quant_type={quant_type!r}. Pick Q4_0 or Q8_0, or load an existing .gguf via `gguf_file=`."
            )
        ggml_type = getattr(gguf.GGMLQuantizationType, quant_type)
        # gguf-py expects a numpy array in fp32.
        fp = value.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        packed = gguf.quants.quantize(fp, ggml_type)
        return {key: torch.from_numpy(packed.view("uint8").reshape(-1))}

    def convert(self, input_dict, **kwargs):
        result: dict = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            result.update(self._quantize_one(key, tensor))
        return result

    @property
    def reverse_op(self):
        return GGUFDequantize(self.hf_quantizer)


class Unsqueeze(ConversionOps):
    """Unsqueeze a tensor along `dim`."""

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
    """Squeeze a tensor along `dim`. Inverse of :class:`Unsqueeze`."""

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
    """Subtract 1 from a tensor (used for GGUF norm weight de-offset).

    The `-1` must run on the fp32 source values *before* the loader casts to
    the target dtype, otherwise the cast then subtract drops 1 ULP near
    `w = 1` and the Gemma/Nemotron norm weights end up ~5e-4 off vs HF.
    `load_checkpoint_state` pre-applies the subtraction in fp32 for arches
    that need it, so by the time the converter chain runs the value is
    already de-offset — make `convert` a no-op pass-through.
    """

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
        return {target_pattern: tensor}

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
    """Apply `log(-tensor)` (used for GGUF Mamba SSM-A de-transform)."""

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
    """Reverse Q-projection GGUF permutation. Reads `config.num_attention_heads` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if not hasattr(config, "num_attention_heads"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
    expected layout. Reads `config.num_attention_heads` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if not hasattr(config, "num_attention_heads"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
    """Reverse K-projection GGUF permutation. Reads `config.num_attention_heads` and
    `config.num_key_value_heads` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if not hasattr(config, "num_attention_heads"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
        if not hasattr(config, "num_attention_heads"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
    """Reverse Bloom QKV weight reshape. Reads `config.n_head` and `config.hidden_size` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        if not hasattr(config, "n_head") or not hasattr(config, "hidden_size"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
    """Reverse Bloom QKV bias reshape. Reads `config.n_head` and `config.hidden_size` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        if not hasattr(config, "n_head") or not hasattr(config, "hidden_size"):
            raise ValueError("Config does not have `num_attention_heads`, you can't use `ReversePermuteAttnQ`.")
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
